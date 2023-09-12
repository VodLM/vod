from __future__ import annotations

import asyncio
import math
import time
import warnings
from multiprocessing.managers import DictProxy
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import rich
import torch
import transformers
from loguru import logger
from vod_dataloaders import fast
from vod_tools import dstruct, pipes
from vod_tools.pipes.utils.misc import pack_examples

from src import vod_configs, vod_search

from .predict_collate import PredictCollate
from .utils import BlockTimer

T = TypeVar("T")

ROW_IDX_COL_NAME: str = "__row_idx__"
FLOAT_INF_THRES = 3e12  # <-- values above this threshold are considered as inf


class RetrievalCollate(pipes.Collate):
    """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch.

    Steps:
        1. search & merge
        2. sample the sections
        3. fetch the content of each section from the huggingface `datasets.Dataset`
        4. tokenize the sections & querys
        5. cast & return the batch (`torch.Tensor`)
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    search_client: vod_search.HybridSearchClient
    sections: dstruct.SizedDataset[dict[str, Any]]
    config: vod_configs.RetrievalCollateConfig
    query_collate_fn: PredictCollate
    section_collate_fn: PredictCollate

    def __init__(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizerBase,
        search_client: vod_search.HybridSearchClient,
        sections: dstruct.SizedDataset[dict[str, Any]],
        config: vod_configs.RetrievalCollateConfig,
        parameters: Optional[dict | DictProxy] = None,
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
        self.parameters = parameters or {}
        self.query_collate_fn = PredictCollate.from_config(self.config, tokenizer=self.tokenizer, field="query")
        self.section_collate_fn = PredictCollate.from_config(self.config, tokenizer=self.tokenizer, field="section")

        # Validate the parameters
        client_names = set(search_client.clients.keys())
        missing_clients = client_names - set(self.parameters.keys())
        if len(missing_clients):
            logger.warning(
                f"Missing weights for clients: `{missing_clients}`. "
                f"Fix this by referencing them in the `parameters` argument. "
                f"Replacing missing values with constant weights `1.0`."
            )
            for client in missing_clients:
                self.parameters[client] = 1.0

    def __call__(self, examples: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch."""
        start_time = time.perf_counter()
        batch = pack_examples(examples)  # list[dict] -> dict[list]

        # Search within each client
        search_results, raw_scores = self.search(batch, top_k=self.config.prefetch_n_sections)
        diagnostics = {f"diagnostics.{key}": s for key, s in search_results.meta.items()}

        # Sample the sections given the positive ones and the pool of candidates
        with BlockTimer(name="diagnostics.sample_sections_time", output=diagnostics):
            if self.config.n_sections is None:
                sections, sampled_raw = search_results, raw_scores
            else:
                sections, sampled_raw = sample_sections(
                    search_results=search_results,
                    raw_scores=raw_scores,
                    n_sections=self.config.n_sections,
                    max_pos_sections=self.config.max_pos_sections,
                    temperature=float(self.config.do_sample),
                    max_support_size=self.config.support_size,  # <-- limit the candidate pool size
                )

        # Flatten sections (in-batch negative)
        if self.config.in_batch_negatives:
            sections, sampled_raw = _gather_in_batch_negatives(
                sections,
                search_results=search_results,
                raw_scores=raw_scores,
                world_size=len(self.sections),
                padding=True,  # <-- padding is required for `torch.compile()` to compile a single graph.
            )

        # Replace negative indices with random ones
        #    this is requiered because `datasets.Dataset` doesn't support negative indices
        sections = _replace_negative_indices(sections, world_size=len(self.sections))

        # Fetch the content of each section from the huggingface `datasets.Dataset`
        flat_ids = sections.indices.flatten().tolist()
        flat_sections_content: dict[str, Any] = self.sections[flat_ids]

        # Tokenize the sections and add them to the output
        with BlockTimer(name="diagnostics.tokenize_time", output=diagnostics):
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
        batch: dict[str, Any],
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
        return _multi_search(
            text=query_text,
            shards=batch[vod_configs.TARGET_SHARD_KEY],
            vector=query_vectors,
            subset_ids=query_subset_ids,
            section_ids=batch[self.config.section_id_keys.query],
            top_k=top_k,
            clients=self.search_client.clients,
            weights=dict(self.parameters),
        )


def _multi_search(
    *,
    text: list[str],
    shards: list[str],
    vector: Optional[np.ndarray] = None,
    subset_ids: Optional[list[list[str]]] = None,
    section_ids: list[list[str]],
    top_k: int,
    clients: dict[str, vod_search.SearchClient],
    weights: dict[str, float],
) -> tuple[vod_search.RetrievalBatch, dict[str, np.ndarray]]:
    """Query a multisearch egine."""
    if "sparse" not in clients.keys():
        raise ValueError("The sparse client must be specified to lookup the positive sections.")

    # Create the search payloads - another payload is prepended to look up the positive sections
    lookup_payload = {
        "client": clients["sparse"],
        "vector": vector,
        "text": [""] * len(text),  # No search query is provided here. We are only looking for the `positive` sections
        "subset_ids": subset_ids,
        "ids": section_ids,
        "shard": shards,
        "top_k": top_k,
    }
    client_names = list(clients.keys())
    payloads = [
        {
            "client": clients[name],
            "vector": vector,
            "text": text,
            "subset_ids": subset_ids,
            "shard": shards,
            "top_k": top_k,
        }
        for name in client_names
    ]

    # Run the searches asynchronously
    meta = {}
    # Note - VL: I am not 100% sure this leverages asynchroneous execution effectively.
    #            This might something to look into for `asyncio` experts.
    with BlockTimer(name="search_time", output=meta):
        search_results = asyncio.run(_execute_search([lookup_payload] + payloads))

    # Unpack the results
    search_results = dict(zip(["lookup"] + client_names, search_results))

    # Discard the scores for the lookup client, discard the labels for all other clients
    search_results["lookup"].scores.fill(0.0)
    for name, result in search_results.items():
        if name == "lookup":
            continue
        result.labels = None

    # DEBUGGING - check for `-inf` scores in the `dense` results (hunting down problems with `faiss`)
    if "dense" in search_results:
        _check_inf_scores(search_results["dense"])

    # Retrieve the meta data
    for name, result in search_results.items():
        for key, value in result.meta.items():
            meta[f"{name}_{key}"] = value

    # normalize the scores
    fast.normalize_scores_(search_results, offset=1.0)

    # Combine the results and add meta data
    combined_results, raw_scores = fast.merge_search_results(
        search_results=search_results,
        weights={"lookup": 0.0, **weights},
    )
    combined_results.meta = meta
    raw_scores.pop("lookup")

    # Set -inf to the mask section (index -1) and replace the neg. indices with random indices
    is_masked = combined_results.indices < 0
    combined_results.scores[is_masked] = -np.inf
    for scores in raw_scores.values():
        scores[is_masked] = -np.inf

    return combined_results, raw_scores


def _replace_negative_indices(sections: vod_search.RetrievalBatch, world_size: int) -> vod_search.RetrievalBatch:
    """Replace negative indices with random ones."""
    is_negative = sections.indices < 0
    n_negative = is_negative.sum()
    if n_negative:
        sections.indices.setflags(write=True)
        sections.indices[is_negative] = np.random.randint(0, world_size, size=n_negative)
    return sections


def _check_inf_scores(r: vod_search.RetrievalBatch[np.ndarray]) -> None:
    """Check whether there is any `+inf` scores in the results.

    This sometimes happens with `faiss`. I still don't understand why.
    """
    is_inf = r.scores >= FLOAT_INF_THRES
    if is_inf.any():
        rich.print(r)
        warnings.warn(
            f"Found {is_inf.sum()} ({is_inf.sum() / is_inf.size:.2%}) inf scores in the search results.",
            stacklevel=2,
        )
        r.scores[is_inf] = np.nan


async def _execute_search(payloads: list[dict[str, Any]]) -> list[vod_search.RetrievalBatch]:
    """Execute the search asynchronously."""

    async def _async_search_one(args: dict[str, Any]) -> vod_search.RetrievalBatch[np.ndarray]:
        client = args.pop("client")
        start_time = time.perf_counter()
        result = await client.async_search(**args)
        result.meta["search_time"] = time.perf_counter() - start_time
        return result

    futures = [_async_search_one(payload) for payload in payloads]
    return await asyncio.gather(*futures)


def _sections_to_dict(
    sections: vod_search.RetrievalBatch,
    raw_scores: Optional[dict[str, np.ndarray]] = None,
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
    batch: dict[str, Any],
    flat_sections_content: dict[str, Any],
    *,
    sections: vod_search.RetrievalBatch,
    query_collate_fn: PredictCollate,
    section_collate_fn: PredictCollate,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # Tokenize the sections
    tokenized_sections = section_collate_fn(flat_sections_content)
    tokenized_sections = {k: v.view(*sections.indices.shape, -1) for k, v in tokenized_sections.items()}
    # Tokenize the queries
    tokenized_query = query_collate_fn(batch)
    return tokenized_query, tokenized_sections


def _get_extra_attributes(
    batch: dict[str, Any],
    flat_sections_content: dict[str, Any],
    *,
    sections_shape: tuple[int, ...],
    config: vod_configs.RetrievalCollateConfig,
) -> dict[str, None | Callable[[Any], Any]]:
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
    for k, fn in extras_keys_ops.items():
        if k not in batch:
            continue
        v = batch[k]
        query_extras[f"query.{k}"] = fn(v) if fn is not None else v

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        query_extras["query.section_ids"] = batch[config.section_id_keys.query]

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
            v = _reshape_flat_list(v, sections_shape)
        sections_extras[f"section.{k}"] = v

    return {**query_extras, **sections_extras}


def _reshape_flat_list(lst: list[T], shape: tuple[int, int]) -> list[list[T]]:
    """Reshape a list."""
    if len(shape) != 2:  # noqa: PLR2004
        raise ValueError(f"Expected a 2D shape. Found {shape}")
    if math.prod(shape) != len(lst):
        raise ValueError(f"Expected a list of length {math.prod(shape)}. Found {len(lst)}")
    return [lst[i : i + shape[1]] for i in range(0, len(lst), shape[1])]


def _gather_in_batch_negatives(
    samples: vod_search.RetrievalBatch,
    search_results: vod_search.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    world_size: int,
    padding: bool = True,
) -> tuple[vod_search.RetrievalBatch, dict[str, np.ndarray]]:
    """Merge all sections (positive and negative) as a flat batch."""
    unique_indices = np.unique(samples.indices)
    if padding:
        # pad the unique indices with random indices to a fixed size
        n_full = math.prod(samples.indices.shape)
        n_pad = n_full - unique_indices.shape[0]
        # We sample sections iid so there might be some collisions
        #   but in practice this is sufficiently unlikely,
        #   so it's not worth the extra computation to check.
        random_indices = np.random.randint(0, world_size, size=n_pad)
        unique_indices = np.concatenate([unique_indices, random_indices])

    # Repeat the unique indices for each section
    unique_indices_: np.ndarray = unique_indices[None, :].repeat(samples.indices.shape[0], axis=0)

    # Gather the scores from the `candidates` batch, set the NaNs to the minimum score
    scores = fast.gather_values_by_indices(unique_indices_, search_results.indices, search_results.scores)

    # Gather the labels from the `positives` batch, set NaNs to negatives
    if search_results.labels is None:
        raise ValueError("The `search_results` must have labels.")
    labels = fast.gather_values_by_indices(unique_indices_, search_results.indices, search_results.labels)
    labels[np.isnan(labels)] = -1

    # Other scores (client scores)
    flat_raw_scores = {}
    for key in raw_scores:
        flat_raw_scores[key] = fast.gather_values_by_indices(unique_indices_, search_results.indices, raw_scores[key])

    return (
        vod_search.RetrievalBatch(
            indices=unique_indices,
            scores=scores,
            labels=labels,
            allow_unsafe=True,
        ),
        flat_raw_scores,
    )


def sample_sections(
    *,
    search_results: vod_search.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    n_sections: int,
    max_pos_sections: Optional[int],
    temperature: float = 0,
    max_support_size: Optional[int] = None,
) -> tuple[vod_search.RetrievalBatch, dict[str, np.ndarray]]:
    """Sample the positive and negative sections."""
    samples = fast.sample(
        search_results=search_results,
        total=n_sections,
        n_positives=max_pos_sections,
        temperature=temperature,
        max_support_size=max_support_size,
    )

    # Sample the `raw_scores`
    sampled_raw_scores = {}
    for key, scores in raw_scores.items():
        sampled_raw_scores[key] = fast.gather_values_by_indices(samples.indices, search_results.indices, scores)

    # Set -inf to the mask section (index -1)
    is_masked = samples.indices < 0
    samples.scores.setflags(write=True)
    samples.scores[is_masked] = -np.inf
    for scores in sampled_raw_scores.values():
        scores.setflags(write=True)
        scores[is_masked] = -np.inf

    return samples, sampled_raw_scores
