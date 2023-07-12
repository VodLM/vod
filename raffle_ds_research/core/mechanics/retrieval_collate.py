from __future__ import annotations

import asyncio
import functools
import math
import time
import warnings
from multiprocessing.managers import DictProxy
from typing import Any, Optional, TypeVar

import numpy as np
import torch
import transformers
from loguru import logger

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics import fast
from raffle_ds_research.core.mechanics.post_filtering import PostFilter, post_filter_factory
from raffle_ds_research.core.mechanics.section_sampler import sample_sections
from raffle_ds_research.core.mechanics.utils import BlockTimer
from raffle_ds_research.tools import c_tools, dstruct, index_tools, pipes
from raffle_ds_research.tools.pipes.utils.misc import pack_examples

T = TypeVar("T")

ROW_IDX_COL_NAME: str = "__row_idx__"


class RetrievalCollate(pipes.Collate):
    """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch.

    Steps:
        1. search & merge
        2. post-filter
        3. sample the sections
        4. fetch the content of each section from the huggingface `datasets.Dataset`
        5. tokenize the sections & questions
        6. cast & return the batch (`torch.Tensor`)
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    search_client: index_tools.MultiSearchClient
    post_filter: Optional[PostFilter]
    sections: dstruct.SizedDataset[dict[str, Any]]
    config: core_config.RetrievalCollateConfig

    def __init__(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizerBase,
        search_client: index_tools.MultiSearchClient,
        sections: dstruct.SizedDataset[dict[str, Any]],
        config: core_config.RetrievalCollateConfig,
        parameters: Optional[dict | DictProxy] = None,
    ):
        if "bm25" not in search_client.clients:
            raise ValueError(
                "The `bm25` client is required to lookup positive sections. "
                "Please add it to the `search_client` argument"
            )
        self.tokenizer = tokenizer
        self.search_client = search_client
        self.sections = sections
        self.config = config
        self.parameters = parameters or {}

        # Build the post-filter
        if config.post_filter is not None:
            post_filter = post_filter_factory(
                config.post_filter,
                sections=sections,
                query_key=config.group_id_keys.query,
                section_key=config.group_id_keys.section,
            )
        else:
            post_filter = None
        self.post_filter = post_filter

        # Validate the parameters
        client_names = set(self.search_client.clients.keys())
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
        batch = pack_examples(examples)

        # Search within each client
        search_results, raw_scores = self.search(batch, top_k=self.config.prefetch_n_sections)
        diagnostics = {f"diagnostics.{key}": s for key, s in search_results.meta.items()}

        # Post-filtering sections based on the group hash
        if self.post_filter is not None:
            search_results, raw_scores = _post_filter(
                search_results,
                raw_scores,
                batch=batch,
                diagnostics=diagnostics,
                post_filter=self.post_filter,
            )

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
            sections, sampled_raw = gather_in_batch_negatives(
                sections,
                search_results=search_results,
                raw_scores=raw_scores,
                world_size=len(self.sections),
                padding=True,  # <-- padding is required for `torch.compile()` to compile a single graph.
                fill_score_offset=self.config.in_batch_neg_offset,
            )

        # Fetch the content of each section from the huggingface `datasets.Dataset`
        flat_ids = sections.indices.flatten().tolist()
        flat_sections_content: dict[str, Any] = self.sections[flat_ids]

        # Tokenize the sections and add them to the output
        with BlockTimer(name="diagnostics.tokenize_time", output=diagnostics):
            tokenized_sections = pipes.torch_tokenize_pipe(
                flat_sections_content,
                tokenizer=self.tokenizer,
                text_key="text",
                prefix_key="section.",
                max_length=self.config.section_max_length,
                truncation=True,
                padding="max_length",
            )
            tokenized_sections = {k: v.view(*sections.indices.shape, -1) for k, v in tokenized_sections.items()}

            # Tokenize the questions
            tokenized_question = pipes.torch_tokenize_pipe(
                batch,
                tokenizer=self.tokenizer,
                text_key="text",
                prefix_key="question.",
                max_length=self.config.question_max_length,
                truncation=True,
                padding="max_length",
            )

        # Get question/section attributes
        attributes = self._get_attributes(
            batch,
            flat_sections_content,
            sections_shape=sections.indices.shape,
        )

        # Debugging: proportion of in-domain sections (same group hash)
        q_group_hash = attributes["question.group_hash"][:, None]
        s_group_hash = attributes["section.group_hash"]
        if s_group_hash.ndim == 1:
            s_group_hash = s_group_hash[None, :]
        in_domain_prop = (q_group_hash == s_group_hash).float().mean(dim=-1).mean().item()
        diagnostics["diagnostics.in_domain_prop"] = in_domain_prop

        # Build the batch
        diagnostics["diagnostics.collate_time"] = time.perf_counter() - start_time
        batch = {
            **tokenized_question,
            **tokenized_sections,
            **_sampled_sections_to_dict(sections, sampled_raw, prefix="section.", as_torch=True),
            **attributes,
            **diagnostics,
            **{f"diagnostics.parameters.{k}": v for k, v in self.parameters.items()},
        }

        return batch

    def _get_attributes(
        self,
        batch: dict[str, Any],
        flat_sections_content: dict[str, Any],
        sections_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        as_tensor = functools.partial(_as_tensor, dtype=torch.long, replace={None: -1})
        extras_keys_ops = {"id": as_tensor, "answer_id": as_tensor, "kb_id": as_tensor, "group_hash": as_tensor}

        # Handle question attributes
        question_extras = {}
        for k, fn in extras_keys_ops.items():
            if k not in batch:
                continue
            question_extras[f"question.{k}"] = fn(batch[k])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            query_section_ids = torch.nested.nested_tensor(batch[self.config.section_id_keys.query])
            question_extras["question.section_ids"] = torch.nested.to_padded_tensor(query_section_ids, padding=-1)

        # Handle section attributes
        sections_extras = {}
        for k, fn in extras_keys_ops.items():
            if k not in flat_sections_content:
                continue
            v = fn(flat_sections_content[k])
            if isinstance(v, torch.Tensor):
                v = v.view(sections_shape)
            sections_extras[f"section.{k}"] = v

        return {**question_extras, **sections_extras}

    def search(
        self,
        batch: dict[str, Any],
        top_k: int,
    ) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
        """Search the batch of queries and return the top `top_k` results."""
        # Get the query ids
        query_group_ids = batch[self.config.group_id_keys.query]

        # Get the query text
        query_text = batch[self.config.text_keys.query]

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
            vector=query_vectors,
            group=query_group_ids,
            query_section_ids=batch[self.config.section_id_keys.query],
            top_k=top_k,
            clients=self.search_client.clients,
            weights=dict(self.parameters),
        )


def _multi_search(
    *,
    text: list[str],
    vector: Optional[np.ndarray] = None,
    group: Optional[list[str]] = None,
    query_section_ids: list[list[str | int]],
    top_k: int,
    clients: dict[str, index_tools.SearchClient],
    weights: dict[str, float],
) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    """Query a multisearch egine."""
    if "bm25" not in clients:
        raise ValueError("The BM25 client must be specified to lookup the positive sections.")

    # Create the search payloads - another payload is prepended to look up the positive sections
    lookup_payload = {
        "client": clients["bm25"],
        "vector": vector,
        "text": [""] * len(text),
        "group": group,
        "section_ids": query_section_ids,
        "top_k": top_k,
    }
    client_names = list(clients.keys())
    payloads = [
        {
            "client": clients[name],
            "vector": vector,
            "text": text,
            "group": group,
            "top_k": top_k,
        }
        for name in client_names
    ]

    # Run the searches asynchronously
    meta = {}
    with BlockTimer(name="search_time", output=meta):
        search_results = asyncio.run(_execute_search([lookup_payload] + payloads))

    # Unpack the results
    search_results = dict(zip(["lookup"] + client_names, search_results))
    search_results["lookup"].scores.fill(0.0)  # Discard the scores for the lookup

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

    # Set -inf to the mask section (index -1)
    is_masked = combined_results.indices < 0
    combined_results.scores[is_masked] = -np.inf
    for scores in raw_scores.values():
        scores[is_masked] = -np.inf

    return combined_results, raw_scores


async def _execute_search(payloads: list[dict[str, Any]]) -> list[index_tools.RetrievalBatch]:
    def search_fn(args: dict[str, Any]) -> index_tools.RetrievalBatch[np.ndarray]:
        client = args.pop("client")
        return client.search(**args)

    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            search_fn,
            payload,
        )
        for payload in payloads
    ]
    return await asyncio.gather(*futures)


def _post_filter(
    search_results: index_tools.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    *,
    post_filter: PostFilter,
    batch: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    n_not_inf = (~np.isinf(search_results.scores)).sum()
    all_scores = {"main": search_results.scores, **raw_scores}
    all_scores = post_filter(search_results.indices, all_scores, query=batch)
    search_results.scores = all_scores.pop("main")  # TODO : make new
    n_not_inf_after = (~np.isinf(search_results.scores)).sum()
    prop_filtered = (n_not_inf - n_not_inf_after) / search_results.scores.size
    diagnostics["diagnostics.post_filtered"] = prop_filtered
    return search_results, all_scores


def _sampled_sections_to_dict(
    sections: index_tools.RetrievalBatch,
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
            # cast to torch tensors
            output = {k: torch.from_numpy(v) for k, v in output.items()}
            output[f"{prefix}label"] = output[f"{prefix}label"].to(torch.bool)

    return output  # type: ignore


def _as_tensor(
    x: list | np.ndarray | torch.Tensor,
    dtype: torch.dtype,
    replace: Optional[dict[T, T]] = None,
) -> torch.Tensor:
    if replace is not None:
        if isinstance(x, list):
            x = [replace.get(i, i) for i in x]
        else:
            raise TypeError(f"Cannot use `replace` with type {type(x)}")

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(dtype=dtype)
    elif isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype)
    else:
        x = torch.tensor(x, dtype=dtype)

    return x


def gather_in_batch_negatives(
    samples: index_tools.RetrievalBatch,
    search_results: index_tools.RetrievalBatch,
    raw_scores: dict[str, np.ndarray],
    world_size: int,
    fill_score_offset: float = 0,
    padding: bool = True,
) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    """Merge all sections (positive and negative) as a flat batch."""
    unique_indices = np.unique(samples.indices)
    if padding:
        # pad the unique indices with random indices to a fixed size
        n_full = math.prod(samples.indices.shape)
        n_pad = n_full - unique_indices.shape[0]
        # We sample iid so there might be some collisions
        # but this is ok unlikely, so not worth the extra computation to check.
        random_indices = np.random.randint(0, world_size, size=n_pad)
        unique_indices = np.concatenate([unique_indices, random_indices])

    # Repeat the unique indices for each section
    unique_indices_ = unique_indices[None, :].repeat(samples.indices.shape[0], axis=0)

    # Gather the scores from the `candidates` batch, set the NaNs to the minimum score
    scores = c_tools.gather_by_index(unique_indices_, search_results.indices, search_results.scores)
    scores = _fill_nan_scores(
        scores,
        ref_scores=search_results.scores,
        fill_value_offset=fill_score_offset,
    )

    # Gather the labels from the `positives` batch, set NaNs to negatives
    labels = c_tools.gather_by_index(unique_indices_, search_results.indices, search_results.labels)
    labels[np.isnan(labels)] = -1

    # Other scores (client scores)
    flat_raw_scores = {}
    for key in raw_scores:
        flat_raw_scores[key] = c_tools.gather_by_index(unique_indices_, search_results.indices, raw_scores[key])
        flat_raw_scores[key] = _fill_nan_scores(
            flat_raw_scores[key],
            ref_scores=search_results.scores,
            fill_value_offset=fill_score_offset,
        )

    return (
        index_tools.RetrievalBatch(
            indices=unique_indices,
            scores=scores,
            labels=labels,
            allow_unsafe=True,
        ),
        flat_raw_scores,
    )


def _fill_nan_scores(scores: np.ndarray, ref_scores: np.ndarray, fill_value_offset: float = 0) -> np.ndarray:
    ref_scores_ = np.where(np.isfinite(ref_scores), ref_scores, np.nan)
    fill_value = np.nanmin(ref_scores_, axis=-1) + fill_value_offset
    scores = np.where(np.isnan(scores), fill_value[:, None], scores)
    return scores
