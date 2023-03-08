from __future__ import annotations

import math
from typing import Any, Optional

import torch
import datasets
from jinja2 import Template as JinjaTemplate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from raffle_ds_research.core.builders.frank_builder import DEFAULT_TEMPLATES
from raffle_ds_research.core.data_models.supervised_retrieval import SupervisedRetrievalBatch
from raffle_ds_research.tools import dataset_builder, pipes


def preprocess_ms_marco(batch: dict, idx: Optional[list] = None, **kwargs) -> dict:
    """Preprocess a batch of raw `MS MARCO` into our format."""
    passages = batch["passages"]
    batch = {
        "question": batch["query"],
        "section": [r["passage_text"] for r in passages],
        "section.label": [r["is_selected"] for r in passages],
    }
    return batch


def filter_ms_marco(batch: dict, idx: Optional[list] = None, **kwargs) -> bool:
    """Filter out batches that have no positive examples."""
    return any(batch["section.label"])


class CollateMsMarco(pipes.Pipe):
    """Collate a list of `MS MARCO` examples into a batch."""

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    question_max_length: Optional[int] = 512
    section_max_length: Optional[int] = 512
    templates: Optional[dict] = DEFAULT_TEMPLATES
    max_sections: Optional[int] = None

    def _collate_egs(self, examples: list[dict], **kwargs) -> dict:
        """Implementation of the collate logic."""
        batch_size = len(examples)
        q_template = JinjaTemplate(self.templates["question"])
        s_template = JinjaTemplate(self.templates["section"])

        # Process the questions
        qs = [q_template.render(question=eg["question"]) for eg in examples]
        q_encodings = self.tokenizer(
            qs,
            return_tensors="pt",
            padding=True,
        )
        q_encodings = {k: v[..., : self.question_max_length] for k, v in q_encodings.items()}

        # Pad the sections to the max number and sample max `self.max_sections`
        max_sections_in_batch = max(len(eg["section"]) for eg in examples)
        examples = [_pad_sections(eg, max_sections_in_batch) for eg in examples]
        if self.max_sections:
            n_samples = min(max_sections_in_batch, self.max_sections)
            examples = [_sample_sections(eg, n_samples) for eg in examples]

        # Flatten, tokenize and truncate the sections. Finally, reshape them back.
        flat_sections = [s for eg in examples for s in eg["section"]]
        flat_sections = [s_template.render(content=s) for s in flat_sections]
        s_encodings = self.tokenizer(
            flat_sections,
            return_tensors="pt",
            padding=True,
        )
        s_encodings = {
            k: v.view(batch_size, -1, v.shape[-1])[..., : self.section_max_length] for k, v in s_encodings.items()
        }

        # Make the final batch
        batch = {
            "section.input_ids": s_encodings["input_ids"],
            "section.attention_mask": s_encodings["attention_mask"],
            "section.label": torch.tensor([eg.pop("section.label") for eg in examples], dtype=torch.bool),
            "section.score": torch.tensor(
                [eg.pop("section.score") for eg in examples],
                dtype=torch.float,
            ),
            "question.input_ids": q_encodings["input_ids"],
            "question.attention_mask": q_encodings["attention_mask"],
        }

        return batch

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        """No need for additional processing here."""
        return batch


class MsMarcoBuilder(dataset_builder.HfBuilder):
    """Builder for the `MS MARCO` dataset."""

    _validate_splits: list[str] = ["train", "validation"]

    def __init__(
        self,
        name: str = "ms_marco",
        subset_name: str = "v2.1",
        filter_questions_without_positive: bool = True,
        load_kwargs: Optional[dict] = None,
        prep_map_kwargs: Optional[dict] = None,
        subset_size: Optional[int | dict[str, int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        question_max_length: int = 512,
        section_max_length: int = 512,
        n_sections: Optional[int | dict[str, int]] = None,
        split: Optional[str] = None,
        templates: Optional[dict[str, str]] = None,
        language: str = "en",
    ):
        load_kwargs = load_kwargs or {}
        load_kwargs["name"] = subset_name
        super().__init__(
            name=name,
            load_kwargs=load_kwargs,
            validator=None,
            batch_validator=SupervisedRetrievalBatch,
        )
        if templates is None:
            templates = DEFAULT_TEMPLATES

        # store the parameters for preprocessing
        if prep_map_kwargs is None:
            prep_map_kwargs = dict(num_proc=4)
        elif isinstance(prep_map_kwargs, DictConfig):
            prep_map_kwargs = OmegaConf.to_container(prep_map_kwargs)
        prep_map_kwargs.update(dict(batched=True, with_indices=True))
        self.prep_map_kwargs = prep_map_kwargs
        self.subset_size = subset_size
        self.split = split
        self.language = language
        self.filter_questions_without_positive = filter_questions_without_positive

        # collate args
        self.tokenizer = tokenizer
        self.templates = templates
        self.question_max_length = question_max_length
        self.section_max_length = section_max_length
        if isinstance(n_sections, DictConfig):
            n_sections = OmegaConf.to_container(n_sections)
        self.n_sections = n_sections

    def _build_dset(self) -> datasets.DatasetDict:
        dataset = super()._build_dset()

        # sub-sample the dataset
        if self.subset_size is not None:
            dataset = self._take_subset(dataset)

        # convert into our own format
        dataset = dataset.map(
            preprocess_ms_marco,
            **self.prep_map_kwargs,
            remove_columns=[
                "passages",
                "query",
                "wellFormedAnswers",
            ],
            desc=f"Preprocess MS MARCO",
        )

        # filter out questions without positive
        if self.filter_questions_without_positive:
            n_questions = {s: len(d) for s, d in dataset.items()}
            kwargs = {**self.prep_map_kwargs, "batched": False}
            dataset = dataset.filter(
                filter_ms_marco,
                desc="Filtering out questions without positive passages (train set).",
                **kwargs,
            )
            n_questions_filtered = {s: n_questions[s] - len(d) for s, d in dataset.items()}
            logger.info(f"Filtered out {n_questions_filtered} questions without positive passages.")

        return dataset

    def get_collate_fn(self, split: Optional[str] = None):
        if isinstance(self.n_sections, dict):
            try:
                n_sections: int = self.n_sections[split]
            except KeyError:
                raise ValueError(f"Unknown split {split}. Known splits: `{self.n_sections.keys()}`.")
        else:
            n_sections = self.n_sections

        collate_fn = CollateMsMarco(
            tokenizer=self.tokenizer,
            question_max_length=self.question_max_length,
            section_max_length=self.section_max_length,
            max_sections=n_sections,
            templates=self.templates,
        )

        return collate_fn

    def get_corpus(self) -> Optional[datasets.Dataset]:
        return None


def _sample_sections(
    eg: dict[str, list],
    n_total: Optional[int],
) -> dict[str, list]:
    """Sample a subset of `n_total` sections."""
    if n_total is None:
        return eg

    def _cast_bool(x: Any) -> bool:
        if isinstance(x, int):
            if x not in {0, 1}:
                raise ValueError(f"Invalid label {x}. Expected 0 or 1.")
            return bool(x)

    def _sample(*, scores: list[float], indices: list[int], n: int) -> list[int]:
        if n:
            samples_ = torch.tensor(scores).softmax(dim=-1).multinomial(n, replacement=False).tolist()
            samples = [indices[i] for i in samples_]
        else:
            samples = []
        return samples

    eg = eg.copy()
    labels: list[bool] = [_cast_bool(x) for x in eg["section.label"]]
    scores: list[float] = eg["section.score"]
    if len(labels) != len(scores):
        raise ValueError(f"Number of labels ({len(labels)}) and scores ({len(scores)}) do not match.")
    if len(labels) < n_total:
        raise ValueError(f"Cannot sample {n_total} sections from {len(labels)} sections.")
    indexed_keys = {k for k in eg.keys() if k.startswith("section")}
    pos_indices = [i for i, l in enumerate(labels) if l]
    neg_indices = [i for i, l in enumerate(labels) if not l]
    if set(pos_indices).intersection(neg_indices):
        raise ValueError("There is an overlap between positive and negative indices.")

    # define the number of positive and negative sections to keep
    n_avail = len(pos_indices) + len(neg_indices)
    n_avail_positives = len(pos_indices)
    n_avail_negatives = len(neg_indices)
    n_positives = min(n_avail // 2, n_avail_positives)
    if n_positives + n_avail_negatives < n_total:
        n_positives = n_total - n_avail_negatives
    n_negatives = n_total - n_positives

    # sample the positives
    pos_samples = _sample(scores=[scores[i] for i in pos_indices], indices=pos_indices, n=n_positives)

    # sample the negatives
    neg_samples = _sample(scores=[scores[i] for i in neg_indices], indices=neg_indices, n=n_negatives)

    # final set of samples
    if set(pos_samples).intersection(neg_samples):
        raise ValueError("There is an neg_samples between positive and negative indices.")
    sampled_indices = pos_samples + neg_samples
    if len(sampled_indices) != n_total:
        raise ValueError(f"Expected {n_total} samples, got {len(sampled_indices)}.")
    for key in indexed_keys:
        eg[key] = [eg[key][i] for i in sampled_indices]

    return eg


def _pad_sections(eg: dict, max_n_sections: int) -> dict:
    """Pad the sections to the maximum number of sections."""
    eg = eg.copy()
    n_secs = len(eg["section"])
    if n_secs == max_n_sections:
        eg["section.score"] = [0.0] * max_n_sections
    else:
        n_to_add = max_n_sections - n_secs
        eg["section"] += [" "] * n_to_add
        eg["section.label"] += [False] * n_to_add
        eg["section.score"] = [0.0] * n_secs + [-math.inf] * n_to_add

    return eg
