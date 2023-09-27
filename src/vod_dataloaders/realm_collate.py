import time
import typing as typ
import warnings

import numpy as np
import torch
import transformers
import vod_configs
import vod_search
import vod_types as vt
from loguru import logger

from .core import (
    in_batch_negatives,
    numpy_ops,
    sample,
    search,
    utils,
)
from .tokenizer_collate import TokenizerCollate

T = typ.TypeVar("T")
P = typ.ParamSpec("P")


ROW_IDX_COL_NAME: str = "__row_idx__"


class RealmCollate(vt.Collate[typ.Any, torch.Tensor | list[int | float | str]]):
    """Collate function for retrieval-augmented language modeling tasks.

    This function is used to convert a list of queries into a batch.
    For each queries, the function will search the search engine for the top results, and sample a subset.

    This function implements the following steps:
        1. search & merge
        2. sample the sections
        3. (optional) flatten sections (in-batch negative)
        4. fetch the content of each section from the huggingface `datasets.Dataset`
        5. tokenize the sections & queries
        6. cast & return the batch (`torch.Tensor`)
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    search_client: vod_search.HybridSearchClient
    config: vod_configs.RetrievalCollateConfig
    query_collate_fn: TokenizerCollate
    section_collate_fn: TokenizerCollate

    def __init__(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizerBase,
        search_client: vod_search.HybridSearchClient,
        config: vod_configs.RetrievalCollateConfig,
        parameters: None | typ.MutableMapping = None,
    ):
        self.tokenizer = tokenizer
        self.search_client = search_client
        self.config = config
        self.parameters = _validate_parameters(parameters or {}, search_client)
        self.query_collate_fn = TokenizerCollate.from_config(self.config, tokenizer=self.tokenizer, field="query")
        self.section_collate_fn = TokenizerCollate.from_config(self.config, tokenizer=self.tokenizer, field="section")

        if config.lookup_engine not in search_client.clients:
            raise ValueError(
                f"The `{config.lookup_engine}` client is required to lookup positive sections. "
                f"Please add it to the `search_client` instance"
            )

    @property
    def sections(self) -> vt.DictsSequence:
        """Get all indexed sections."""
        return self.search_client.sections

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
            samples = sample.sample_search_results(
                search_results=search_results,
                raw_scores=raw_scores,
                total=self.config.n_sections,
                max_pos_sections=self.config.max_pos_sections,
                temperature=float(self.config.do_sample),
                max_support_size=self.config.support_size,  # <-- limit the candidate pool size
            )

        # Flatten sections (in-batch negative)
        if self.config.in_batch_negatives:
            # NOTE: padding is required to output the same number of sections
            #       so `torch.compile()` to compile a single graph.
            samples = in_batch_negatives.flatten_samples(
                samples,
                padding=True,
            )

        # Replace negative indices with random ones
        #    this is required because `datasets.Dataset` doesn't support negative indices
        sections = numpy_ops.replace_negative_indices(samples.samples, world_size=len(self.sections))

        # Fetch the content of each section from the huggingface `datasets.Dataset`
        sections_shape = sections.indices.shape
        flat_ids = sections.indices.flatten().tolist()
        flat_sections_content: dict[str, list[typ.Any]] = self.sections[flat_ids]

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
            sections_shape=sections_shape,  # type: ignore
            config=self.config,
        )

        # Make the final batch and potentially cast attributes to `torch.Tensor``
        batch = {
            **tokenized_querys,
            **tokenized_sections,
            **_sections_to_dict(
                sections,
                sampling_log_weights=samples.log_weights,
                raw_scores=samples.raw_scores,
                prefix="section.",
                as_torch=True,
            ),
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
            try:
                query_vectors = batch[utils.VECTOR_KEY]
            except KeyError as exc:
                raise ValueError(
                    f"The search client `{type(self.search_client).__name__}` requires vectors. "
                    f"Please make sure indexing the provided dataset returns a dict with a key "
                    f"`{utils.VECTOR_KEY}` and value being a vector representation for that row. "
                    f"Found keys: `{list(batch.keys())}`."
                ) from exc
            if isinstance(query_vectors, list):
                query_vectors = np.stack(query_vectors)
        else:
            query_vectors = None

        # Async search the query text using `search_client.async_search`
        return search.async_hybrid_search(
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
    sampling_log_weights: None | np.ndarray = None,
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
        f"{prefix}score": sections.scores,
        f"{prefix}label": sections.labels > 0,
    }

    if raw_scores is not None:
        output.update({f"{prefix}{k}": v for k, v in raw_scores.items()})

    if sampling_log_weights is not None:
        output[f"{prefix}log_weight"] = sampling_log_weights

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
    sections_shape: tuple[int, int] | tuple[int],
    config: vod_configs.RetrievalCollateConfig,
) -> dict[str, None | dict[str, typ.Any]]:
    if len(sections_shape) > 2:  # noqa: PLR2004
        raise ValueError(f"Expected a 1D or 2D shape. Found {sections_shape}")

    # Define operators to apply to each extra key
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
        if len(sections_shape) == 2:  # noqa: PLR2004
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
