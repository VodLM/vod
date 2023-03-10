from __future__ import annotations

from functools import partial
from typing import Optional

import datasets
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from raffle_ds_research.tools import dataset_builder, pipes
from raffle_ds_research.tools.raffle_datasets import frank

QUESTION_TEMPLATE = "<extra_id_0>{{ question }}"
SECTION_TEMPLATE = "<extra_id_1>Title: " "{% if title %}{{ title }}{% else %}NO TITLE{% endif %}. " "{{ content }}"

DEFAULT_TEMPLATES = {
    "question": QUESTION_TEMPLATE,
    "section": SECTION_TEMPLATE,
}


class FrankRowModel(BaseModel):
    question: str
    answer_id: int
    section_id: Optional[int]
    knowledge_base_id: int
    question_input_ids: list[int] = Field(
        ...,
        alias="question.input_ids",
    )
    question_attention_mask: list[int] = Field(
        ...,
        alias="question.input_ids",
    )
    section_pid: list[int] = Field(
        ...,
        alias="section.pid",
    )
    section_score: list[float] = Field(
        ...,
        alias="section.score",
    )
    section_label: list[bool] = Field(
        ...,
        alias="section.label",
    )


class FrankBuilder(dataset_builder.HfBuilder):
    def __init__(
        self,
        language: str,
        split: frank.FrankSplitName,
        tokenizer: PreTrainedTokenizerBase,
        name: str = "frank",
        prep_map_kwargs: Optional[dict] = None,
        index_max_top_k: int = 100,
        n_sections: int = 32,
        question_max_length: Optional[int] = 512,
        section_max_length: Optional[int] = 512,
        templates: Optional[dict[str, str]] = None,
        subset_size: Optional[int | dict[str, int]] = None,
    ):
        super().__init__(
            name=name,
            load_kwargs={},
            validator=FrankRowModel,
        )

        if templates is None:
            templates = DEFAULT_TEMPLATES

        self.language = language
        self.split = split
        self.tokenizer = tokenizer
        self.index_max_top_k = index_max_top_k
        self.subset_size = subset_size

        # store the parameters for preprocessing
        if prep_map_kwargs is None:
            prep_map_kwargs = dict(num_proc=4)
        elif isinstance(prep_map_kwargs, DictConfig):
            prep_map_kwargs = OmegaConf.to_container(prep_map_kwargs)
        prep_map_kwargs.update(dict(batched=True, with_indices=True))
        self.prep_map_kwargs = prep_map_kwargs
        self.templates = templates

        # collate
        self.n_sections = n_sections
        self.question_max_length = question_max_length
        self.section_max_length = section_max_length

    def _load_frank_split(self, frank_split: frank.FrankSplitName) -> frank.HfFrankSplit:
        return frank.load_frank(self.language, split=frank_split)

    def _build_sections(self) -> datasets.Dataset:
        frank_split = self._load_frank_split(self.split)
        sections = frank_split.sections

        pipe = self._get_sections_preprocessing()
        sections = sections.map(
            pipe,
            **self.prep_map_kwargs,
            desc=f"Preprocessing Frank ({self.split}) sections",
        )
        return sections

    def _build_qa(self) -> datasets.DatasetDict:
        frank_split = self._load_frank_split(self.split)
        qa = frank_split.qa_splits

        # format and tokenize the questions
        pipe = self._get_qa_preprocessing()
        qa = qa.map(
            pipe,
            **self.prep_map_kwargs,
            desc=f"Preprocessing Frank ({self.split}) QA splits",
        )
        return qa

    def _build_dset(self) -> datasets.DatasetDict:
        qa = self._build_qa()
        sections = self._build_sections()

        # map the sections to the questions
        index_kwargs = dict(max_top_k=self.index_max_top_k, padding=True)
        index = pipes.LookupIndexPipe(sections, **index_kwargs)
        mapped_qa = qa.map(
            index,
            **self.prep_map_kwargs,
            desc=f"Indexing Frank ({self.split}) sections",
        )

        # sub-sample the dataset
        if self.subset_size is not None:
            mapped_qa = self._take_subset(mapped_qa)

        return mapped_qa

    def get_collate_fn(self, split: Optional[str] = None):
        sections = self._build_sections()
        sampler = pipes.RandomSampler(total=self.n_sections)
        collate_fn = pipes.SupervisedBatcher(
            tokenizer=self.tokenizer,
            sampler=sampler,
            sections=sections,
            question_max_length=self.question_max_length,
            section_max_length=self.section_max_length,
        )
        return collate_fn

    def _get_qa_preprocessing(self) -> pipes.Pipe:
        qa_pipe = pipes.Sequential(
            pipes=[
                partial(
                    pipes.template_pipe,
                    template=self.templates["question"],
                    input_keys=["question"],
                    output_key="question",
                ),
                partial(
                    pipes.tokenize_pipe,
                    tokenizer=self.tokenizer,
                    fied="question",
                    padding=False,
                    truncation=False,
                ),
            ]
        )
        return qa_pipe

    def _get_sections_preprocessing(self) -> pipes.Pipe:
        sections_pipe = pipes.Sequential(
            pipes=[
                pipes.TemplatePipe(
                    template=self.templates["section"],
                    input_keys=["title", "content"],
                    output_key="section",
                ),
                pipes.TokenizerPipe(
                    tokenizer=self.tokenizer,
                    input_key="section",
                    fn_kwargs=dict(padding=False, truncation=False),
                ),
            ]
        )
        return sections_pipe

    def get_corpus(self) -> Optional[datasets.Dataset]:
        return self._build_sections()
