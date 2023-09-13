import time
import typing as typ
import warnings

import numpy as np
import torch
import transformers
import vod_types as vt
from loguru import logger

from src import vod_configs, vod_search

from .tokenizer_collate import TokenizerCollate
from .tools import (
    in_batch_negatives,
    sample,
    search,
    utils,
)

T = typ.TypeVar("T")

ROW_IDX_COL_NAME: str = "__row_idx__"


class RealmCollate(vt.Collate[typ.Any, torch.Tensor | list[int | float | str]]):
    """Collate function for retrieval-augmented language modeling tasks.

    This function is used to convert a list of examples into a batch.
    Steps:
        1. search & merge
        2. sample the sections
        3. fetch the content of each section from the huggingface `datasets.Dataset`
        4. tokenize the sections & querys
        5. cast & return the batch (`torch.Tensor`)
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    search_client: vod_search.HybridSearchClient
    sections: vt.DictsSequence
    config: vod_configs.RetrievalCollateConfig
    query_collate_fn: TokenizerCollate
    section_collate_fn: TokenizerCollate

    def __init__(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizerBase,
        search_client: vod_search.HybridSearchClient,
        sections: vt.DictsSequence,
        config: vod_configs.RetrievalCollateConfig,
        parameters: None | typ.MutableMapping = None,
    ):
        if "sparse" not in search_client.clients:
            raise ValueError(
                "The `sparse` client is required to lookup positive sections. "
                "Please add it to the `search_client` instance"
            )
        self.tokenizer = tokenizer
        self.search_client = search_client
        self.sections = sections
        self.config = config
        self.parameters = _validate_parameters(parameters or {}, search_client)
        self.query_collate_fn = TokenizerCollate.from_config(self.config, tokenizer=self.tokenizer, field="query")
        self.section_collate_fn = TokenizerCollate.from_config(self.config, tokenizer=self.tokenizer, field="section")

    def __call__(
        self,
        inputs: list[dict[str, typ.Any]],
        **kws: typ.Any,
    ) -> dict[str, torch.Tensor | list[int | float | str]]:
        """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch."""
        start_time = time.perf_counter()
        batch = utils.pack_examples(inputs)  # list[dict] -> dict[list]

        # Search within each client
        search_results, raw_scores = self.search(batch, top_k=self.config.prefetch_n_sections)
        diagnostics = {f"diagnostics.{key}": s for key, s in search_results.meta.items()}

        # Sample the sections given the positive ones and the pool of candidates
        with utils.BlockTimer(name="diagnostics.sample_sections_time", output=diagnostics):
            if self.config.n_sections is None:
                sections, sampled_raw = search_results, raw_scores
            else:
                sections, sampled_raw = sample.sample_sections(
                    search_results=search_results,
                    raw_scores=raw_scores,
                    n_sections=self.config.n_sections,
                    max_pos_sections=self.config.max_pos_sections,
                    temperature=float(self.config.do_sample),
                    max_support_size=self.config.support_size,  # <-- limit the candidate pool size
                )

        # Flatten sections (in-batch negative)
        if self.config.in_batch_negatives:
            sections, sampled_raw = in_batch_negatives.flatten_sections(
                sections,
                search_results=search_results,
                raw_scores=raw_scores,
                world_size=len(self.sections),
                padding=True,  # <-- padding is required for `torch.compile()` to compile a single graph.
            )

        # Replace negative indices with random ones
        #    this is requiered because `datasets.Dataset` doesn't support negative indices
        sections = utils.replace_negative_indices(sections, world_size=len(self.sections))

        # Fetch the content of each section from the huggingface `datasets.Dataset`
        flat_ids = sections.indices.flatten().tolist()
        flat_sections_content: dict[str, typ.Any] = self.sections[flat_ids]

        # Tokenize the sections and add them to the output
        with utils.BlockTimer(name="diagnostics.tokenize_time", output=diagnostics):
            tokenized_querys, tokenized_sections = _tokenize_fields(
                batch,
                flat_sections_content,
                sections=sections,
                query_collate_fn=self.query_collate_fn,
                section_collate_fn=self.section_collate_fn,
            )

        # Get query/section attributes (e.g., subset_id, retrieval_ids, etc.)
        attributes = _get_extra_attributes(
            batch,
            flat_sections_content,
            sections_shape=sections.indices.shape,
            config=self.config,
        )

        # Build the batch
        batch = {
            **tokenized_querys,
            **tokenized_sections,
            **_sections_to_dict(sections, sampled_raw, prefix="section.", as_torch=True),
            **attributes,
            **diagnostics,
            **{f"diagnostics.parameters.{k}": v for k, v in self.parameters.items()},
        }
        batch["diagnostics.collate_time"] = time.perf_counter() - start_time
        return batch

    def search(
        self,
        batch: dict[str, typ.Any],
        top_k: int,
    ) -> tuple[vod_search.RetrievalBatch, dict[str, np.ndarray]]:
        """Search the batch of queries and return the top `top_k` results."""
        # Get the query ids
        query_subset_ids = batch[self.config.subset_id_keys.query]

        # Get the query text
        query_text: list[str] = self.query_collate_fn.template.render_batch(batch)
        # Get the query vectors
        if self.search_client.requires_vectors:
            query_vectors = batch[self.config.vector_keys.query]
            if isinstance(query_vectors, list):
                query_vectors = np.stack(query_vectors)
        else:
            query_vectors = None

        # Async search the query text using `search_client.async_search`
        return search.multi_search(
            text=query_text,
            shards=batch[vod_configs.TARGET_SHARD_KEY],
            vector=query_vectors,
            subset_ids=query_subset_ids,
            section_ids=batch[self.config.section_id_keys.query],
            top_k=top_k,
            clients=self.search_client.clients,
            weights=dict(self.parameters),
        )


def _sections_to_dict(
    sections: vod_search.RetrievalBatch,
    raw_scores: None | dict[str, np.ndarray] = None,
    prefix: str = "",
    as_torch: bool = False,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Convert the sampled sections to a dictionary."""
    if sections.labels is None:
        raise ValueError("The sections must have labels.")
    output = {
        f"{prefix}idx": sections.indices,
        f"{prefix}score": sections.scores,
        f"{prefix}label": sections.labels > 0,
    }

    if raw_scores is not None:
        output.update({f"{prefix}{k}": v for k, v in raw_scores.items()})

    if as_torch:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fns = {
                f"{prefix}score": lambda x: torch.from_numpy(x).to(torch.float32),
                f"{prefix}label": lambda x: torch.from_numpy(x).to(torch.bool),
            }
            output = {k: fns.get(k, torch.from_numpy)(v) for k, v in output.items()}

    return output  # type: ignore


def _tokenize_fields(
    batch: dict[str, typ.Any],
    flat_sections_content: dict[str, typ.Any],
    *,
    sections: vod_search.RetrievalBatch,
    query_collate_fn: TokenizerCollate,
    section_collate_fn: TokenizerCollate,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # Tokenize the sections
    tokenized_sections = section_collate_fn(flat_sections_content)
    tokenized_sections = {k: v.view(*sections.indices.shape, -1) for k, v in tokenized_sections.items()}
    # Tokenize the queries
    tokenized_query = query_collate_fn(batch)
    return tokenized_query, tokenized_sections


def _get_extra_attributes(
    batch: dict[str, typ.Any],
    flat_sections_content: dict[str, typ.Any],
    *,
    sections_shape: tuple[int, int],
    config: vod_configs.RetrievalCollateConfig,
) -> dict[str, None | typ.Callable[[typ.Any], typ.Any]]:
    extras_keys_ops = {
        "id": None,
        "language": None,
        "subset_id": None,
        "subset_ids": None,
        "link": None,
        "dset_uid": None,
    }

    # Handle query attributes
    query_extras = {}
    query_extras["query.section_ids"] = batch[config.section_id_keys.query]
    for k, fn in extras_keys_ops.items():
        if k not in batch:
            continue
        v = batch[k]
        query_extras[f"query.{k}"] = fn(v) if fn is not None else v

    # Handle section attributes
    sections_extras = {}
    for k, fn in extras_keys_ops.items():
        if k not in flat_sections_content:
            continue
        v = flat_sections_content[k]
        v = fn(v) if fn is not None else v
        if isinstance(v, torch.Tensor):
            v = v.view(sections_shape)
        elif isinstance(v, np.ndarray):
            v = v.reshape(sections_shape)
        elif isinstance(v, list):
            v = utils.reshape_flat_list(v, sections_shape)
        sections_extras[f"section.{k}"] = v

    return {**query_extras, **sections_extras}


def _validate_parameters(
    parameters: typ.MutableMapping,
    search_client: vod_search.HybridSearchClient,
    default_value: float = 1.0,
) -> typ.MutableMapping:
    """Validate the parameters against the client names."""
    client_names = set(search_client.clients.keys())
    missing_clients = client_names - set(parameters.keys())
    if len(missing_clients):
        logger.warning(
            f"Missing weights for clients: `{missing_clients}`. "
            f"Fix this by referencing them in the `parameters` argument. "
            f"Replacing missing values with constant weights `1.0`."
        )
        for client in missing_clients:
            parameters[client] = default_value

    return parameters
