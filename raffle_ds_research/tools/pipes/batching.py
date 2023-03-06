from typing import Optional

import torch.utils.data
from datasets import Dataset as HfDataset
from pydantic import BaseModel, Field, PrivateAttr
from transformers import PreTrainedTokenizerBase

from ..utils.exceptions import dump_exceptions_to_file
from .pipe import Pipe
from .sampler import Sampler
from .utils.misc import keep_only_columns


class SupervisedBatcherExample(BaseModel):
    """Defines the input model for each example (row in the dataset)."""

    question_input_ids: list[int] = Field(
        ...,
        description="The input ids of the question.",
        alias="question.input_ids",
    )
    question_attention_mask: list[int] = Field(
        ...,
        description="The attention mask of the question.",
        alias="question.attention_mask",
    )
    section_pids: list[int] = Field(
        ...,
        description="The indices of the sections",
        alias="section.pid",
    )
    section_scores: list[float] = Field(
        ...,
        description="The scores to sample from.",
        alias="section.score",
    )
    section_labels: list[bool] = Field(
        ...,
        description="The labels of the sections",
        alias="section.label",
    )


class SupervisedBatcher(Pipe):
    """This pipe implements a `collate_fn` for supervised training.
    It is intended to be applied to a list of examples and return a batch.
    The function can be split into the following steps:

    1. in `_collate_egs`:
        a. pad and concatenate the questions
        b. gather the required question fields.
    2. in `_process_batch`:
        a. sample a subset of sections using the `Sampler` object
        b. fetch the sections from the `sections` dataset
        c. pad and concatenate the sections
        d. gather the required section fields along with the question fields.
    """

    tokenizer: PreTrainedTokenizerBase = Field(..., description="The tokenizer to use.")
    question_max_length: Optional[int] = Field(None, description="The max length of the question.")
    section_max_length: Optional[int] = Field(None, description="The max length of the document.")
    _example_model = PrivateAttr(SupervisedBatcherExample)
    _sections: HfDataset = PrivateAttr()
    _required_sections_columns: set[str] = PrivateAttr({"section.input_ids", "section.attention_mask"})
    sampler: Sampler = Field(..., description="The sampler to use.")

    def __init__(self, sections: HfDataset, **kwargs):
        super().__init__(**kwargs)
        if self._required_sections_columns is not None:
            sections = keep_only_columns(sections, self._required_sections_columns, strict=True)
        self._sections = sections

    @dump_exceptions_to_file
    def _collate_egs(self, egs: list[dict], **kwargs) -> dict[str, torch.Tensor]:
        """Collate a list of questions into a batch."""
        egs = [self._example_model(**eg).dict(by_alias=True) for eg in egs]
        q_tokens = [
            {
                "input_ids": eg.pop("question.input_ids"),
                "attention_mask": eg.pop("question.attention_mask"),
            }
            for eg in egs
        ]
        q_encodings = self.tokenizer.pad(
            q_tokens,
            return_tensors="pt",
        )
        q_encodings = {k: v[..., : self.question_max_length] for k, v in q_encodings.items()}

        batch = {
            "section.pid": torch.tensor([eg.pop("section.pid") for eg in egs]),
            "section.score": torch.tensor([eg.pop("section.score") for eg in egs]),
            "section.label": torch.tensor([eg.pop("section.label") for eg in egs]),
            "question.input_ids": q_encodings["input_ids"],
            "question.attention_mask": q_encodings["attention_mask"],
        }
        return batch

    @dump_exceptions_to_file
    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict[str, torch.Tensor]:
        """Sample the sections, fetch them from the dataset and pad them.s"""

        # sample sections
        samples = self.sampler(batch, idx=idx, keep_tensors=True, **kwargs)
        batch.update(samples)

        # fetch the section contents
        pids = batch["section.pid"]
        sections = self._sections[pids.view(-1)]

        # pad the sections
        sections_encodings = self.tokenizer.pad(
            {
                "input_ids": sections["section.input_ids"],
                "attention_mask": sections["section.attention_mask"],
            },
            return_tensors="pt",
        )
        sections_encodings = {
            f"section.{k}": v[..., : self.section_max_length].view(*pids.shape, -1)
            for k, v in sections_encodings.items()
        }

        # make the output
        output = {**batch, **sections_encodings}
        return output
