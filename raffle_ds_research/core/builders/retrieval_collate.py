from __future__ import annotations

import collections
import copy
import dataclasses
import warnings
from typing import Any, Callable, Iterable, Optional, Tuple

import datasets
import numpy as np
import omegaconf
import pydantic
import rich
import torch
import transformers

from raffle_ds_research.core.builders.utils import numpy_gumbel_like, numpy_log_softmax
from raffle_ds_research.tools import c_tools, index_tools, pipes
from raffle_ds_research.tools.pipes.utils.misc import pack_examples

ROW_IDX_COL_NAME: str = "__row_idx__"
TEST_MODE = False


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
    def __init__(
        self,
        *,
        corpus: datasets.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        config: RetrievalCollateConfig,
        **kwargs: Any,
    ):
        self.corpus = corpus
        self.lookup_index = pipes.LookupIndexPipe(corpus, keys=list(config.label_keys.values()))
        self.in_domain_lookup_index = pipes.LookupIndexPipe(corpus, keys=list(config.in_domain_keys.values()))
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        batch = pack_examples(examples)
        client_cfgs = list(self.config.enabled_clients)

        if len(client_cfgs) == 0:
            # if not search service is enabled, fetch the sections from the same kb_id
            neg_lookup_input = {v: batch[k] for k, v in self.config.in_domain_keys.items()}
            candidate_samples: index_tools.RetrievalBatch = _wrap_as_retrieval_batch(
                self.in_domain_lookup_index.search(neg_lookup_input),
                fill_value=0.0,
            )
            client_scores = {}
            candidate_samples_ = [candidate_samples]
        else:
            # fetch the query vectors
            if any(cfg.requires_vectors for cfg in client_cfgs):
                question_ids = batch[ROW_IDX_COL_NAME]
                query_vectors = self.config.query_vectors[question_ids]
            else:
                query_vectors = None

            # search the indexes
            candidate_samples_ = []
            for cfg in client_cfgs:
                samples: index_tools.RetrievalBatch = cfg.client.search(
                    vector=query_vectors,
                    text=batch["question"],
                    top_k=self.config.prefetch_n_sections,
                )
                candidate_samples_.append(samples)

            # concatenate the results
            candidate_samples, client_scores = _merge_candidate_samples(copy.deepcopy(candidate_samples_), client_cfgs)

            # ----------------------------------------------------------------
            # TESTING - TODO: remove
            if TEST_MODE:
                for i in range(len(batch["question"])):
                    for k, (pid, sco) in enumerate(zip(candidate_samples.indices[i], candidate_samples.scores[i])):
                        if pid < 0:
                            continue
                        pid_found = False
                        for j, ori_samples in enumerate(candidate_samples_):
                            cfg = client_cfgs[j]
                            ori_indices = ori_samples.indices[i].tolist()
                            ori_scores = ori_samples.scores[i].tolist()
                            agg_scores = client_scores[cfg.name][i].tolist()
                            agg_indices = candidate_samples.indices[i].tolist()
                            if pid not in ori_indices:
                                continue
                            pid_found = True
                            pid_loc = ori_indices.index(pid)
                            ori_scores_at_pid = ori_scores[pid_loc]
                            agg_scores_at_pid = agg_scores[k]
                            if not np.allclose(agg_scores_at_pid, ori_scores_at_pid):
                                rich.print(
                                    dict(
                                        i=i,
                                        j=j,
                                        client=cfg.name,
                                        pid=pid,
                                        aggr=[(agg_indices[t], agg_scores[t]) for t in range(len(agg_indices))],
                                        orig=[(ori_indices[t], ori_scores[t]) for t in range(len(ori_indices))],
                                    )
                                )
                                raise ValueError(
                                    f"[{i}, {j}; pid={pid}] score {agg_scores_at_pid} != {ori_scores_at_pid} in {cfg.name} index"
                                )
                        if not pid_found:
                            rich.print(
                                [
                                    (cfg.name, sample.indices[i].tolist())
                                    for cfg, sample in zip(client_cfgs, candidate_samples_)
                                ]
                                + [
                                    ("agg", candidate_samples.indices[i].tolist()),
                                ]
                            )
                            raise ValueError(f"[{i}; pid={pid}] not found in any index")
            # ----------------------------------------------------------------

        # tokenize the questions
        tokenized_question = pipes.torch_tokenize_pipe(
            batch,
            tokenizer=self.tokenizer,
            field="question",
            max_length=self.config.question_max_length,
            truncation=True,
        )

        # fetch the positive section ids (The `rule` allows keeping sections with match on either `id` or `answer_id`)
        pos_lookup_input = {v: batch[k] for k, v in self.config.label_keys.items()}
        pos_lookup_results = self.lookup_index.search(pos_lookup_input)
        positive_samples: index_tools.RetrievalBatch = _wrap_as_retrieval_batch(
            pos_lookup_results,
            is_positive_section=_matches_answer_xor_section,
            fill_value=np.nan,
        )

        # sample the sections given the positive ones and the pool of candidates
        sections: SampledSections = sample_sections(
            candidates=candidate_samples,
            positives=positive_samples,
            n_sections=self.config.n_sections,
            max_pos_sections=self.config.max_pos_sections,
            do_sample=self.config.do_sample,
            other_scores=client_scores,
        )

        # ----------------------------------------------------------------
        # TESTING - TODO: remove
        if TEST_MODE:
            for i in range(len(batch["question"])):
                if not sections.labels[i].any():
                    rich.print(
                        dict(
                            positive_samples=positive_samples,
                            sections_labels=sections.labels,
                        )
                    )
                    raise ValueError(f"No positive section found. i={i}")
                for pid, sco in zip(sections.indices[i], sections.scores[i]):
                    if pid < 0:
                        continue
                    pid_found = False
                    for j, ori_samples in enumerate(candidate_samples_):
                        cfg = client_cfgs[j]
                        if pid not in ori_samples.indices[i]:
                            if pid in positive_samples.indices[i]:
                                pid_found = True
                            else:
                                raise ValueError(f"pid {pid} not found in {cfg.name} index, nor in positive index")

                        else:
                            pid_found = True
                            pid_loc = ori_samples.indices[i].tolist().index(pid)

                    if not pid_found:
                        rich.print(
                            [
                                {"client": client_cfgs[j].name, "pids": sample.indices[i].tolist()}
                                for j, sample in enumerate(candidate_samples_)
                            ]
                        )
                        raise ValueError(f"pid {pid} not found in any index")
        # ----------------------------------------------------------------

        # fetch the content of each section
        flat_ids = sections.indices.flatten().tolist()
        flat_sections_content = self.corpus[flat_ids]  # type: ignore

        # tokenize the sections and add them to the output
        tokenized_sections = pipes.torch_tokenize_pipe(
            flat_sections_content,
            tokenizer=self.tokenizer,
            field="section",
            max_length=self.config.section_max_length,
            truncation=True,
        )
        tokenized_sections = {k: v.view(*sections.indices.shape, -1) for k, v in tokenized_sections.items()}

        # question extras
        questions_extras_keys = ["id", "section_id", "answer_id", "kb_id"]
        questions_extras = {
            f"question.{k}": _to_tensor(batch[k], dtype=torch.long, replace={None: -1}) for k in questions_extras_keys
        }

        # section extras
        section_extras_keys = ["id", "answer_id", "kb_id"]
        sections_extras = {
            f"section.{k}": _to_tensor(flat_sections_content[k], dtype=torch.long) for k in section_extras_keys
        }
        sections_extras = {k: v.view(*sections.indices.shape) for k, v in sections_extras.items()}

        batch = {
            **questions_extras,
            **tokenized_question,
            **tokenized_sections,
            **sections.to_dict(prefix="section.", as_torch=True),
            **sections_extras,
        }

        return batch


def _merge_candidate_samples(
    candidates: Iterable[index_tools.RetrievalBatch],
    client_cfgs: list[SearchClientConfig],
) -> Tuple[index_tools.RetrievalBatch, dict[str, np.ndarray]]:
    candidates = list(candidates)
    if len(candidates) == 1:
        return candidates[0], {client_cfgs[0].name: candidates[0].scores}
    elif len(candidates) == 0:
        raise ValueError("No candidates to merge")

    candidate_samples = index_tools.merge_retrieval_batches(candidates)

    # replace nan scores with the minimum score
    candidate_samples.scores = _fill_nans_with_min(values=candidate_samples.scores, offset_min_value=-1, axis=1)

    # Aggregate the scores
    new_scores = np.zeros_like(candidate_samples.scores[..., 0])
    client_scores = {}
    for i, client_cfg in enumerate(client_cfgs):
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


def _wrap_as_retrieval_batch(
    lookup_results: pipes.LookupSearchResults,
    is_positive_section: Optional[Callable[[pipes.LookupSearchResults], np.ndarray]] = None,
    fill_value: float = np.nan,
) -> index_tools.RetrievalBatch:
    """Wrap the lookup results as a `RetrievalBatch`."""
    if is_positive_section is None:
        is_positive_section = lookup_results.frequencies.sum(axis=-1) > 0
    else:
        is_positive_section = is_positive_section(lookup_results)

    # override the scores to NaN for positives, -inf for negatives
    scores = np.where(is_positive_section, fill_value, -np.inf)
    indices = np.where(is_positive_section, lookup_results.indices, -1)
    return index_tools.RetrievalBatch(indices=indices, scores=scores)


def _matches_answer_xor_section(results: pipes.LookupSearchResults) -> np.ndarray:
    """For each row, set mark the sections as positive.
    - where there is a match on `id` (section_id).
    - else where there is a match on `answer_id` (answer_id).
    """
    id_col_idx = results.labels.index("id")
    answer_id_col_idx = results.labels.index("answer_id")

    # frequencies: [batch_size, n_sections, n_labels]
    frequencies: np.ndarray = results.frequencies

    # compute the `id_defined` mask
    has_section_id_match = (frequencies[:, :, id_col_idx] > 0).any(axis=-1)
    is_positive_section = np.where(
        has_section_id_match[:, None],
        frequencies[:, :, id_col_idx] > 0,  # positive ids are the ones at the section level
        frequencies[:, :, answer_id_col_idx] > 0,  # positive ids are the ones at the answer level
    )

    if np.any(is_positive_section.sum(axis=-1) == 0):
        rich.print(results)
        raise ValueError("No positive section found")

    return is_positive_section


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
        arbitrary_types_allowed = True
        extra = "forbid"

    name: str
    client: index_tools.SearchClient
    weight: float = 1.0

    @property
    def enabled(self):
        return self.weight >= 0

    @property
    def requires_vectors(self):
        return self.client.requires_vectors


class RetrievalCollateConfig(pydantic.BaseModel):
    """Defines a configuration for the retrieval collate function."""

    _query_vectors: index_tools.VectorHandler = pydantic.PrivateAttr(None)

    class Config:
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

    # data required for indexing
    question_vectors: Optional[index_tools.VectorType] = None
    clients: list[SearchClientConfig] = []

    # keys for the lookup index
    label_keys: collections.OrderedDict[str, str] = {"answer_id": "answer_id", "section_id": "id"}
    in_domain_keys: collections.OrderedDict[str, str] = {"kb_id": "kb_id"}

    @pydantic.validator("clients", pre=True)
    def _validate_clients(cls, v):
        def _parse(w):
            if isinstance(w, SearchClientConfig):
                return w
            elif isinstance(w, (dict, omegaconf.DictConfig)):
                return SearchClientConfig(**w)
            else:
                raise ValueError(f"Invalid client config: {w}")

        clients = [_parse(c) for c in v]
        clients = [c for c in clients if c.enabled]
        return clients

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.question_vectors is not None:
            self._query_vectors = index_tools.vector_handler(self.question_vectors)

    def __getstate__(self):
        """Drop the open ts.TensorStore object to make the state serializable."""
        state = super().__getstate__().copy()
        state["__private_attribute_values__"].pop("_query_vectors")
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.question_vectors is not None:
            self._query_vectors = index_tools.vector_handler(self.question_vectors)

    def __del__(self):
        """Close the open ts.TensorStore object."""
        if self._query_vectors is not None:
            del self._query_vectors

    @property
    def query_vectors(self) -> index_tools.VectorHandler:
        return self._query_vectors

    @property
    def search_enabled(self) -> bool:
        return any(c.enabled for c in self.clients)

    @property
    def requires_vectors(self) -> bool:
        return any(c.requires_vectors for c in self.clients)

    @property
    def enabled_clients(self) -> list[SearchClientConfig]:
        return [c for c in self.clients if c.enabled]

    @pydantic.validator("label_keys", pre=True)
    def _validate_label_keys(cls, v: Any) -> collections.OrderedDict[str, str]:
        if isinstance(v, collections.OrderedDict):
            return v
        elif isinstance(v, dict):
            return collections.OrderedDict(v)
        else:
            raise TypeError(f"Expected dict or OrderedDict, got {type(v)}")
