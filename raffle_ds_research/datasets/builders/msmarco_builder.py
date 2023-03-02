from __future__ import annotations

import math
from typing import Optional

import torch
from datasets import DatasetDict as HfDatasetDict
from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from raffle_ds_research.datasets.builders.builder import HfBuilder
from raffle_ds_research.datasets.data_models.supervised_retrieval import SupervisedRetrievalBatch
from raffle_ds_research.tools import pipes


def preprocess_ms_marco(batch: dict, idx: Optional[list] = None, **kwargs) -> dict:
    passages = batch["passages"]
    batch = {
        "question": batch["query"],
        "section": [r["passage_text"] for r in passages],
        "section.label": [r["is_selected"] for r in passages],
    }
    return batch


class CollateMsMarco(pipes.Pipe):
    tokenizer: Optional[PreTrainedTokenizerBase] = Field(None, alias="tokenizer")
    question_max_length: Optional[int] = Field(512, alias="question_max_length")
    section_max_length: Optional[int] = Field(512, alias="section_max_length")

    def _collate_egs(self, examples: list[dict], **kwargs) -> dict:
        # process the questions
        q_encodings = self.tokenizer(
            [eg["question"] for eg in examples],
            return_tensors="pt",
            padding=True,
        )
        q_encodings = {k: v[..., : self.question_max_length] for k, v in q_encodings.items()}

        # resample the number of sections per questions if needed
        max_n_sections = max(len(eg["section"]) for eg in examples)
        for eg in examples:
            self._pad_sections(eg, max_n_sections)

        # process the sections
        flat_sections = [s for eg in examples for s in eg["section"]]
        s_encodings = self.tokenizer(
            flat_sections,
            return_tensors="pt",
            padding=True,
        )
        s_encodings = {
            k: v.view(-1, max_n_sections, v.shape[-1])[..., : self.section_max_length] for k, v in s_encodings.items()
        }

        # make the final batch
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

    @staticmethod
    def _pad_sections(eg: dict, max_n_sections: int) -> None:
        n_secs = len(eg["section"])
        if n_secs == max_n_sections:
            eg["section.score"] = [0.0] * max_n_sections
        else:
            n_to_add = max_n_sections - n_secs
            eg["section"] += [" "] * n_to_add
            eg["section.label"] += [False] * n_to_add
            eg["section.score"] = [0.0] * n_secs + [-math.inf] * n_to_add

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        return batch


class MsMarcoBuilder(HfBuilder):
    def __init__(
        self,
        name: str = "ms_marco",
        subset_name: str = "v2.1",
        load_kwargs: Optional[dict] = None,
        prep_map_kwargs: Optional[dict] = None,
        subset_size: Optional[int | dict[str, int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        question_max_length: int = 512,
        section_max_length: int = 512,
        top_k: Optional[int] = None,
        split: Optional[str] = None,
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

        # collate
        self.top_k = top_k
        self.question_max_length = question_max_length
        self.section_max_length = section_max_length
        self._collate_fn = CollateMsMarco(
            tokenizer=tokenizer,
            question_max_length=question_max_length,
            section_max_length=section_max_length,
        )

    def _build_dset(self) -> HfDatasetDict:
        dset = super()._build_dset()

        # sub-sample the dataset
        if self.subset_size is not None:
            dset = self._take_subset(dset)

        # convert into our own format
        dset = dset.map(
            preprocess_ms_marco,
            **self.prep_map_kwargs,
            remove_columns=[
                "passages",
                "query",
                "wellFormedAnswers",
            ],
            desc=f"Preprocess MS MARCO",
        )

        return dset

    def get_collate_fn(self):
        return self._collate_fn
