# pylint: disable=no-member
from __future__ import annotations

import copy
import dataclasses
import functools
import warnings
from typing import Any, Iterable, Optional, T, Tuple

import datasets
import numpy as np
import omegaconf
import pydantic
import rich
import torch
import transformers

from raffle_ds_research.core.dataset_builders.utils import numpy_gumbel_like, numpy_log_softmax
from raffle_ds_research.tools import c_tools, index_tools, pipes
from raffle_ds_research.tools.pipes.utils.misc import pack_examples

ROW_IDX_COL_NAME: str = "__row_idx__"
TEST_MODE = False


def _identity(x: T) -> T:
    return x


def sample_sections(
    *,
    positives: index_tools.RetrievalBatch,
    candidates: index_tools.RetrievalBatch,
    n_sections: int,
    max_pos_sections: int,
    do_sample: bool = False,
    other_scores: Optional[dict[str, np.ndarray]] = None,
    lookup_positive_scores: bool = True,
) -> SampledSections:
    """Sample the positive and negative sections.
    This function uses the Gumbel-Max trick to sample from the corresponding distributions.
    Gumbel-Max: https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    """

    if max_pos_sections is None:
        max_pos_sections = n_sections

    positives = copy.copy(positives)
    if lookup_positive_scores:
        # set the positive scores to be the scores from the pool of candidates
        positives.scores = c_tools.gather_by_index(
            queries=positives.indices,
            keys=candidates.indices,
            values=candidates.scores,
        )
        # replace NaNs with the minimum value along each dimension (each question)
        positives.scores = _fill_nans_with_min(offset_min_value=-1, values=positives.scores)
        # make sure that padded sections have -inf scores
        positives.scores = np.where(positives.indices < 0, -np.inf, positives.scores)
    else:
        positives.scores = np.where(np.isnan(positives.scores), 0, positives.scores)
        # make sure that padded sections have -inf scores
        positives.scores = np.where(positives.indices < 0, -np.inf, positives.scores)

    # gather the positive sections and apply perturbations
    positive_logits = numpy_log_softmax(positives.scores)
    if do_sample:
        positive_logits += numpy_gumbel_like(positive_logits)

    # gather the negative sections and apply perturbations
    negative_logits = numpy_log_softmax(candidates.scores)
    if do_sample:
        negative_logits += numpy_gumbel_like(negative_logits)

    # concat the positive and negative sections
    concatenated = c_tools.concat_search_results(
        a_indices=positives.indices,
        a_scores=positive_logits,
        b_indices=candidates.indices,
        b_scores=negative_logits,
        max_a=max_pos_sections,
        total=n_sections,
    )

    # set the labels to be `1` for the positive sections (revert label ordering)
    concatenated.labels = np.where(concatenated.labels == 0, 1, 0)

    # fetch the scores from the pool of negatives
    scores = c_tools.gather_by_index(
        queries=concatenated.indices,
        keys=candidates.indices,
        values=candidates.scores,
    )

    # also fetch the `other` scores (e.g., bm25, faiss) for tracking purposes
    if other_scores is not None:
        other_scores = {
            k: c_tools.gather_by_index(
                queries=concatenated.indices,
                keys=candidates.indices,
                values=v,
            )
            for k, v in other_scores.items()
        }

    output = SampledSections(
        indices=concatenated.indices,
        scores=scores,
        labels=concatenated.labels,
        other_scores=other_scores,
    )

    if (concatenated.labels.sum(axis=1) == 0).any():
        rich.print(
            dict(
                positive_indices=positives.indices,
                positive_logits=positive_logits,
                negative_indices=candidates.indices,
                negative_logits=negative_logits,
            )
        )
        rich.print(output)
        raise ValueError("No positive sections were sampled.")

    return output


def _fill_nans_with_min(values: np.ndarray, offset_min_value: Optional[float] = -1, axis: int = -1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_scores = np.nanmin(values, axis=axis, keepdims=True)
        min_scores = np.where(np.isnan(min_scores), 0, min_scores)
        if offset_min_value is not None:
            min_scores += offset_min_value  # make sure the min is lower than the rest
    values = np.where(np.isnan(values), min_scores, values)
    return values


class RetrievalCollate(pipes.Collate):
    """Collate function for retrieval tasks. This function is used to convert a list of examples into a batch.
    Steps:
        1. Pack the examples into a batch.
        2. Tokenize the questions
        3. Search the sections (bm25, faiss, etc.)
        4. Sample the sections (top_k
        5. Tokenize the sections"""

    _corpus: datasets.Dataset
    _section_id_lookup: index_tools.LookupIndexKnowledgeBase
    _kb_id_lookup: index_tools.LookupIndex
    tokenizer: transformers.PreTrainedTokenizer
    config: RetrievalCollateConfig

    def __init__(
        self,
        *,
        corpus: datasets.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        config: RetrievalCollateConfig,
    ):
        self._corpus = corpus
        self._section_id_lookup = index_tools.LookupIndexKnowledgeBase(
            corpus,
            key=config.section_id_keys.section,
            kb_key=config.kb_id_keys.section,
        )
        self._kb_id_lookup = index_tools.LookupIndex(
            corpus,
            key=config.kb_id_keys.section,
        )
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        batch = pack_examples(examples)

        # lookup the kb section ids
        kb_ids = batch[self.config.kb_id_keys.query]
        kb_samples: index_tools.RetrievalBatch = self._kb_id_lookup.search(kb_ids)
        if not self.config.search_enabled:
            # if not search service is enabled, use the sections from the same kb_id
            candidate_samples = kb_samples
            client_scores: dict[str, np.ndarray] = {}
        else:
            # fetch the query vectors
            if self.config.requires_vectors:
                question_ids = batch[ROW_IDX_COL_NAME]
                query_vectors = self.config.query_vectors[question_ids]
                if np.all(query_vectors == 0):
                    raise ValueError("Query vectors are all zeros.")
            else:
                query_vectors = None

            # search the indexes
            samples_: list[ClientResults] = []
            for cfg in self.config.enabled_clients:
                samples: index_tools.RetrievalBatch = cfg.client.search(
                    vector=query_vectors,
                    text=batch["text"],
                    label=batch[cfg.label_key],
                    top_k=self.config.prefetch_n_sections,
                )
                samples_.append(ClientResults(samples=samples, cfg=cfg))

            # concatenate the results
            candidate_samples, client_scores = _merge_candidate_samples(samples_)

        if self.config.mask_out_of_kb:
            # mask the sections that are not in the same kb_id.
            # this is a hack: `gather_by_index` returns `-inf` for missing values,
            #                  we set `1` for the sections that are found
            is_found = (
                c_tools.gather_by_index(
                    candidate_samples.indices,
                    kb_samples.indices,
                    np.ones_like(kb_samples.scores),
                )
                > 0
            )
            candidate_samples.scores = np.where(is_found, candidate_samples.scores, -np.inf)
            for k, v in client_scores.items():
                client_scores[k] = np.where(is_found, v, -np.inf)

        # tokenize the questions
        tokenized_question = pipes.torch_tokenize_pipe(
            batch,
            tokenizer=self.tokenizer,
            text_key="text",
            prefix_key="question.",
            max_length=self.config.question_max_length,
            truncation=True,
        )

        # fetch the positive `section_ids`
        query_section_ids = batch[self.config.section_id_keys.query]
        positive_samples: index_tools.RetrievalBatch = self._section_id_lookup.search(query_section_ids, kb_ids)

        # sample the sections given the positive ones and the pool of candidates
        sections: SampledSections = sample_sections(
            candidates=candidate_samples,
            positives=positive_samples,
            n_sections=self.config.n_sections,
            max_pos_sections=self.config.max_pos_sections,
            do_sample=self.config.do_sample,
            other_scores=client_scores,
        )

        # fetch the content of each section from the huggingface `datasets.Dataset`
        flat_ids = sections.indices.flatten().tolist()
        flat_sections_content: dict[str, Any] = self._corpus[flat_ids]  # type: ignore

        # tokenize the sections and add them to the output
        tokenized_sections = pipes.torch_tokenize_pipe(
            flat_sections_content,
            tokenizer=self.tokenizer,
            text_key="text",
            prefix_key="section.",
            max_length=self.config.section_max_length,
            truncation=True,
        )
        tokenized_sections = {k: v.view(*sections.indices.shape, -1) for k, v in tokenized_sections.items()}

        batch = {
            **tokenized_question,
            **tokenized_sections,
            **sections.to_dict(prefix="section.", as_torch=True),
            **self._get_extras(batch, flat_sections_content, sections_shape=sections.indices.shape),
        }

        return batch

    def _get_extras(
        self,
        batch: dict[str, Any],
        flat_sections_content: dict[str, Any],
        sections_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        __to_tensor = functools.partial(_to_tensor, dtype=torch.long, replace={None: -1})
        extras_keys = {"id": __to_tensor, "answer_id": __to_tensor, "kb_id": __to_tensor}

        # handle question attributes
        question_extras = {}
        for k, fn in extras_keys.items():
            if k not in batch:
                continue
            question_extras[f"question.{k}"] = fn(batch[k])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            query_section_ids = torch.nested.nested_tensor(batch[self.config.section_id_keys.query])
            question_extras["question.section_ids"] = torch.nested.to_padded_tensor(query_section_ids, padding=-1)

        # handle section attributes
        sections_extras = {}
        for k, fn in extras_keys.items():
            if k not in flat_sections_content:
                continue
            v = fn(flat_sections_content[k])
            if isinstance(v, torch.Tensor):
                v = v.view(sections_shape)
            sections_extras[f"section.{k}"] = v

        return {**question_extras, **sections_extras}


@dataclasses.dataclass(frozen=True)
class ClientResults:
    """A container for the results of a search client."""

    samples: index_tools.RetrievalBatch
    cfg: SearchClientConfig


def _merge_candidate_samples(
    candidates: Iterable[ClientResults],
) -> Tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    candidates = list(candidates)
    if len(candidates) == 1:
        candidate = candidates[0]
        return candidate.samples, {candidate.cfg.name: candidate.samples.scores}
    if len(candidates) == 0:
        raise ValueError("No candidates to merge")

    candidate_samples = index_tools.merge_retrieval_batches([c.samples for c in candidates])

    # replace nan scores with the minimum score
    candidate_samples.scores = _fill_nans_with_min(values=candidate_samples.scores, offset_min_value=-1, axis=1)

    # Aggregate the scores
    new_scores = np.zeros_like(candidate_samples.scores[..., 0])
    client_scores = {}
    for i, client_cfg in enumerate(c.cfg for c in candidates):
        client_scores_i = np.copy(candidate_samples.scores[..., i])
        new_scores += client_cfg.weight * client_scores_i
        client_scores[client_cfg.name] = client_scores_i

    # replace the negative indices with -inf
    new_scores = np.where(candidate_samples.indices < 0, -np.inf, new_scores)
    candidate_samples.scores = new_scores

    # sort by score - might not be necessary.
    sort_ids = np.argsort(candidate_samples.scores, axis=-1)
    sort_ids = np.flip(sort_ids, axis=-1)
    candidate_samples.indices = np.take_along_axis(candidate_samples.indices, sort_ids, axis=-1)
    candidate_samples.scores = np.take_along_axis(candidate_samples.scores, sort_ids, axis=-1)
    for k, v in client_scores.items():
        client_scores[k] = np.take_along_axis(v, sort_ids, axis=-1)

    return candidate_samples, client_scores


@dataclasses.dataclass(frozen=True)
class SampledSections:
    indices: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    other_scores: Optional[dict[str, np.ndarray]] = None

    def to_dict(self, prefix: str = "", as_torch: bool = False) -> dict[str, np.ndarray | torch.Tensor]:
        output = {
            f"{prefix}idx": self.indices,
            f"{prefix}score": self.scores,
            f"{prefix}label": self.labels,
        }

        if self.other_scores is not None:
            output.update({f"{prefix}{k}": v for k, v in self.other_scores.items()})

        if as_torch:
            output = {k: torch.from_numpy(v) for k, v in output.items()}
            output[f"{prefix}label"] = output[f"{prefix}label"].to(torch.bool)

        return output


def _to_tensor(x: Any, dtype: torch.dtype, replace: Optional[dict] = None) -> torch.Tensor:
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


class SearchClientConfig(pydantic.BaseModel):
    """Defines a configuration for a search client."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "forbid"

    name: str
    client: index_tools.SearchClient
    weight: float = 1.0
    label_key: str = "kb_id"

    @property
    def enabled(self) -> bool:
        """Whether the client is enabled or not."""
        return self.weight >= 0

    @property
    def requires_vectors(self) -> bool:
        """Whether the client requires vectors (embeddings) or not."""
        return self.client.requires_vectors


class KeyMap(pydantic.BaseModel):
    """Defines the name of the keys used on the query side and on the section side."""

    class Config:
        """Pydantic config."""

        extra = "forbid"

    query: str
    section: str


class RetrievalCollateConfig(pydantic.BaseModel):
    """Defines a configuration for the retrieval collate function."""

    _query_vectors: index_tools.VectorHandler = pydantic.PrivateAttr(None)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "forbid"

    # base config
    split: str = "train"
    n_sections: int = 10
    prefetch_n_sections: int = 100
    max_pos_sections: int = 3
    do_sample: bool = False
    question_max_length: int = 512
    section_max_length: int = 512
    mask_out_of_kb: bool = True

    # data required for indexing
    question_vectors: Optional[index_tools.VectorType] = None
    clients: list[SearchClientConfig] = []

    # keys for the lookup index
    section_id_keys: KeyMap = KeyMap(query="section_ids", section="id")
    kb_id_keys: KeyMap = KeyMap(query="kb_id", section="kb_id")

    @pydantic.validator("clients", pre=True)
    def _validate_clients(cls, values: list[SearchClientConfig | omegaconf.DictConfig]) -> list[SearchClientConfig]:
        def _parse(x: SearchClientConfig | omegaconf.DictConfig) -> SearchClientConfig:
            if isinstance(x, SearchClientConfig):
                return x
            if isinstance(x, (dict, omegaconf.DictConfig)):
                return SearchClientConfig(**x)
            raise ValueError(f"Invalid client config: type={type(x)}, value={x}")

        clients = [_parse(one_value) for one_value in values]
        clients = [one_client for one_client in clients if one_client.enabled]
        return clients

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.question_vectors is not None:
            self._query_vectors = index_tools.vector_handler(self.question_vectors)

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open ts.TensorStore object to make the state serializable."""
        state = super().__getstate__().copy()
        state["__private_attribute_values__"].pop("_query_vectors")
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        if self.question_vectors is not None:
            self._query_vectors = index_tools.vector_handler(self.question_vectors)

    def __del__(self) -> None:
        """Close the open ts.TensorStore object."""
        if self._query_vectors is not None:
            del self._query_vectors

    @property
    def query_vectors(self) -> index_tools.VectorHandler:
        """Return the vectors/embeddings of the queries."""
        return self._query_vectors

    @property
    def search_enabled(self) -> bool:
        """Return whether any of the clients is enabled."""
        return any(c.enabled for c in self.clients)

    @property
    def requires_vectors(self) -> bool:
        """Return whether any of the clients requires vectors."""
        return any(c.requires_vectors for c in self.clients)

    @property
    def enabled_clients(self) -> list[SearchClientConfig]:
        """Return the list of enabled clients."""
        return [c for c in self.clients if c.enabled]
