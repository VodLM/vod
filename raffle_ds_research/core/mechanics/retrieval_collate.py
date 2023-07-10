from __future__ import annotations

import asyncio
import functools
import math
import pathlib
import time
import warnings
from multiprocessing.managers import DictProxy
from typing import Any, Optional, TypeVar

import numpy as np
import rich
import torch
import transformers
from loguru import logger

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics.post_filtering import PostFilter, post_filter_factory
from raffle_ds_research.core.mechanics.section_sampler import SampledSections, sample_sections
from raffle_ds_research.core.mechanics.utils import fill_nans_with_min
from raffle_ds_research.tools import c_tools, dstruct, index_tools, pipes
from raffle_ds_research.tools.pipes.utils.misc import pack_examples

T = TypeVar("T")

ROW_IDX_COL_NAME: str = "__row_idx__"


class RetrievalCollate(pipes.Collate):
    """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch.

    Steps:
        1. search
        2. merge
        3. post-filter
        4. Gather the positive `section_ids`
        5. sample the sections given the positive ones and the pool of candidates'
        6. fetch the content of each section from the huggingface `datasets.Dataset`
        7. tokenize the sections & questions
        8. return the batch (`torch.Tensor`)
    """

    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    search_client: index_tools.MultiSearchClient
    post_filter: Optional[PostFilter]
    sections: dstruct.SizedDataset[dict[str, Any]]
    config: core_config.RetrievalCollateConfig
    _target_lookup: Optional[index_tools.LookupIndexbyGroup]
    _target_lookup_path: Optional[pathlib.Path]

    def __init__(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        search_client: index_tools.MultiSearchClient,
        sections: dstruct.SizedDataset[dict[str, Any]],
        config: core_config.RetrievalCollateConfig,
        parameters: Optional[dict | DictProxy] = None,
        target_lookup: index_tools.LookupIndexbyGroup | pathlib.Path,
    ):
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

        # Lookup table for the positive sections
        if isinstance(target_lookup, index_tools.LookupIndexbyGroup):
            self._target_lookup = target_lookup
            self._target_lookup_path = None
        elif isinstance(target_lookup, pathlib.Path):
            self._target_lookup_path = target_lookup
            self._target_lookup = None
        else:
            raise TypeError(f"Invalid type for `target_lookup`: {type(target_lookup)}")

    @property
    def target_lookup(self) -> index_tools.LookupIndexbyGroup:
        """Lazy loading of the target lookup table."""
        if self._target_lookup is None:
            if self._target_lookup_path is None:
                raise ValueError("Both `target_lookup` and `target_lookup_path` are `None`")
            self._target_lookup = index_tools.LookupIndexbyGroup.load(self._target_lookup_path)
        return self._target_lookup

    def __call__(self, examples: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch."""
        start_time = time.perf_counter()
        batch = pack_examples(examples)

        # Search within each client
        search_results = self.search(batch, top_k=self.config.prefetch_n_sections)
        diagnostics = {f"diagnostics.{key}_time": s.meta.get("time", None) for key, s in search_results.items()}

        # Merge the search results from the different clients
        candidate_samples, client_scores = weighted_merge_search_results(
            search_results,
            weights=self.parameters,
            fill_nan_scores=True,  # <-- replace NaNs for each client with the minimum score of that client
            fill_nan_scores_offset=-5,  # <-- offset to use when replacing NaNs with minimum scores + this offset
        )

        # Post-filtering sections based on the group hash
        if self.post_filter is not None:
            n_not_inf = (~np.isinf(candidate_samples.scores)).sum()
            candidate_samples = self.post_filter(candidate_samples, query=batch)
            n_not_inf_after = (~np.isinf(candidate_samples.scores)).sum()
            prop_filtered = (n_not_inf - n_not_inf_after) / candidate_samples.scores.size
            diagnostics["diagnostics.post_filtered"] = prop_filtered

        # Fetch the positive `section_ids`
        with BlockTimer(name="diagnostics.target_lookup_time", output=diagnostics):
            kb_ids = batch[self.config.group_id_keys.query]
            query_section_ids = batch[self.config.section_id_keys.query]
            positive_samples: index_tools.RetrievalBatch = self.target_lookup.search(query_section_ids, kb_ids)

        # Sample the sections given the positive ones and the pool of candidates
        with BlockTimer(name="diagnostics.sample_sections_time", output=diagnostics):
            max_support_size = self.config.prefetch_n_sections if self.config.n_sections is not None else None
            sections: SampledSections = sample_sections(
                candidates=candidate_samples,
                positives=positive_samples,
                n_sections=self.config.n_sections,
                max_pos_sections=self.config.max_pos_sections,
                do_sample=self.config.do_sample,
                other_scores=client_scores,
                max_support_size=max_support_size,  # <-- limit the max candidate pool size (deactivated when no sampl.)
            )

        # Flatten sections (in-batch negative)
        if self.config.in_batch_negatives:
            sections = gather_in_batch_negatives(
                sections,
                candidates=candidate_samples,
                positives=positive_samples,
                client_scores=client_scores,
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
            **_sampled_sections_to_dict(sections, prefix="section.", as_torch=True),
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
        **kwargs: Any,
    ) -> dict[str, index_tools.RetrievalBatch]:
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
        return asyncio.run(
            self.search_client.async_search(
                text=query_text,
                vector=query_vectors,
                label=query_group_ids,
                top_k=top_k,
            )
        )


def _sampled_sections_to_dict(
    sections: SampledSections, prefix: str = "", as_torch: bool = False
) -> dict[str, np.ndarray | torch.Tensor]:
    """Convert the sampled sections to a dictionary."""
    output = {f"{prefix}idx": sections.indices, f"{prefix}score": sections.scores, f"{prefix}label": sections.labels}

    if sections.other_scores is not None:
        output.update({f"{prefix}{k}": v for k, v in sections.other_scores.items()})

    if as_torch:
        output = {k: torch.from_numpy(v) for k, v in output.items()}
        output[f"{prefix}label"] = output[f"{prefix}label"].to(torch.bool)

    return output


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


def _ensure_match(
    *,
    batch: dict[str, Any],
    candidate_samples: index_tools.RetrievalBatch,
    features: dstruct.SizedDataset[dict[str, Any]],
    features_keys: list[str],
) -> index_tools.RetrievalBatch:
    """Filter the candidates sections based on the `config.ensure_match` keys."""
    batch_features = {key: np.asarray(batch[key]) for key in features_keys}
    section_indices = candidate_samples.indices.flatten().tolist()
    section_features = features[section_indices]

    keep_mask = None
    for key, batch_features_key in batch_features.items():
        section_features_key = section_features[key]
        section_features_key = np.asarray(section_features_key).reshape(batch_features_key.shape[0], -1)
        keep_mask_key = batch_features_key[:, None] == section_features_key
        keep_mask = keep_mask_key if keep_mask is None else keep_mask & keep_mask_key
    if keep_mask is None:
        raise ValueError("No features to match")

    candidate_samples.scores = np.where(keep_mask, candidate_samples.scores, -math.inf)
    return candidate_samples


def weighted_merge_search_results(
    candidates: dict[str, index_tools.RetrievalBatch],
    weights: dict[str, float],
    fill_nan_scores: bool = False,
    fill_nan_scores_offset: float = -1,
    top_k: Optional[int] = None,
) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    """Merge the candidate samples from multiple clients."""
    ordered_keys = list(candidates.keys())
    weights = {key: weights[key] for key in candidates}
    if len(candidates) == 1:
        key = ordered_keys[0]
        candidate = candidates[key]
        candidate_scores = candidate.scores
        candidate.scores = candidate_scores * weights[key]
        return candidate, {key: candidate_scores}
    if len(candidates) == 0:
        raise ValueError("No candidates to merge")

    candidate_samples = index_tools.merge_retrieval_batches([candidates[key] for key in ordered_keys])
    if top_k is not None:
        candidate_samples.indices[..., :top_k]
        candidate_samples.scores[..., :top_k]

    if fill_nan_scores:
        # replace nan scores with the minimum score
        candidate_samples.scores = fill_nans_with_min(
            values=candidate_samples.scores,
            offset_min_value=fill_nan_scores_offset,
            axis=1,
        )

    # Aggregate the scores
    aggregated_scores = np.full_like(candidate_samples.scores[..., 0], np.nan)
    client_scores = {}
    for i, key in enumerate(ordered_keys):
        weight = weights[key]
        client_scores[key] = candidate_samples.scores[..., i]
        weighted_client_scores_i = weight * client_scores[key]
        aggregated_scores = np.where(
            np.isnan(aggregated_scores),
            weighted_client_scores_i,
            aggregated_scores + np.where(np.isnan(weighted_client_scores_i), 0, weighted_client_scores_i),
        )

    # Replace the negative indices with -inf
    aggregated_scores = np.where(candidate_samples.indices < 0, -np.inf, aggregated_scores)
    candidate_samples.scores = aggregated_scores
    return candidate_samples, client_scores


def _sort_sections(
    candidate_samples: index_tools.RetrievalBatch,
    client_scores: dict[str, np.ndarray],
) -> tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    sort_ids = np.argsort(candidate_samples.scores, axis=-1)
    sort_ids = np.flip(sort_ids, axis=-1)
    candidate_samples.indices = np.take_along_axis(candidate_samples.indices, sort_ids, axis=-1)
    candidate_samples.scores = np.take_along_axis(candidate_samples.scores, sort_ids, axis=-1)
    for k, v in client_scores.items():
        client_scores[k] = np.take_along_axis(v, sort_ids, axis=-1)

    return candidate_samples, client_scores


class BlockTimer:
    """A context manager for timing code blocks."""

    def __init__(self, name: str, output: dict[str, Any]) -> None:
        self.name = name
        self.output = output

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *args: Any) -> None:
        self.output[self.name] = time.perf_counter() - self.start


def gather_in_batch_negatives(
    samples: SampledSections,
    candidates: index_tools.RetrievalBatch,
    positives: index_tools.RetrievalBatch,
    client_scores: dict[str, np.ndarray],
    world_size: int,
    fill_score_offset: float = 0,
    padding: bool = True,
) -> SampledSections:
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
    scores = c_tools.gather_by_index(unique_indices_, candidates.indices, candidates.scores)
    scores = _fill_nan_scores(
        scores,
        ref_scores=candidates.scores,
        fill_value_offset=fill_score_offset,
    )

    # Gather the labels from the `positives` batch, set NaNs to negatives
    labels = c_tools.gather_by_index(unique_indices_, positives.indices, positives.indices >= 0)
    labels = (~np.isnan(labels)) & (labels > 0)

    # Other scores (client scores)
    if samples.other_scores is not None:
        others = {}
        for key in samples.other_scores:
            others[key] = c_tools.gather_by_index(unique_indices_, candidates.indices, client_scores[key])
            others[key] = _fill_nan_scores(
                others[key],
                ref_scores=candidates.scores,
                fill_value_offset=fill_score_offset,
            )
    else:
        others = None

    return SampledSections(
        indices=unique_indices,
        scores=scores,
        labels=labels,
        other_scores=others,
    )


def _fill_nan_scores(scores: np.ndarray, ref_scores: np.ndarray, fill_value_offset: float = 0) -> np.ndarray:
    ref_scores_ = np.where(np.isfinite(ref_scores), ref_scores, np.nan)
    fill_value = np.nanmin(ref_scores_, axis=-1) + fill_value_offset
    scores = np.where(np.isnan(scores), fill_value[:, None], scores)
    return scores
