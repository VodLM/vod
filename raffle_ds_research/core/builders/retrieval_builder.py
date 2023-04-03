from __future__ import annotations

import abc
import copy
from typing import Any, Optional, TypeVar, Union

import datasets
import numpy as np
import omegaconf
import pydantic
import transformers
from typing_extensions import Type

from raffle_ds_research.core.builders import retrieval_collate
from raffle_ds_research.tools import dataset_builder

QUESTION_TEMPLATE = "Question: {{ question }}"
SECTION_TEMPLATE = "{% if title %}Title: {{ title }}. Document: {% endif %}{{ content }}"
DEFAULT_TEMPLATES = {
    "question": QUESTION_TEMPLATE,
    "section": SECTION_TEMPLATE,
}


class RetrievalBuilderConfig(pydantic.BaseModel):
    """Defines a configuration for a retrieval dataset builder."""

    class Config:
        """Pydantic config for the `RetrievalBuilderConfig` class."""

        extra = pydantic.Extra.forbid
        arbitrary_types_allowed = True

    name: str
    subset_name: Optional[str] = None
    language: str = "en"
    tokenizer: Optional[Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None
    index_max_top_k: int = 100
    n_sections: int = 32
    question_max_length: Optional[int] = 512
    section_max_length: Optional[int] = 512
    templates: Optional[dict[str, str]] = DEFAULT_TEMPLATES
    hf_load_kwargs: dict[str, Any] = dict(num_proc=4)
    prep_map_kwargs: dict[str, Any] = dict()
    subset_size: Optional[Union[int, dict[str, int]]] = None

    @pydantic.validator("templates", pre=True)
    def _validate_templates(cls, templates: Any) -> dict[str, str]:
        """Converts templates to a dict if they are an OmegaConf DictConfig."""
        if isinstance(templates, omegaconf.DictConfig):
            templates = omegaconf.OmegaConf.to_container(templates)
        return templates

    @pydantic.validator("prep_map_kwargs", pre=True)
    def _validate_prep_map_kwargs(cls, prep_map_kwargs: Any) -> dict[str, Any]:
        """Converts prep_map_kwargs to a dict if they are an OmegaConf DictConfig."""
        if isinstance(prep_map_kwargs, omegaconf.DictConfig):
            prep_map_kwargs = omegaconf.OmegaConf.to_container(prep_map_kwargs)

        return prep_map_kwargs

    @pydantic.validator("hf_load_kwargs", pre=True)
    def _validate_hf_load_kwargs(cls, hf_load_kwargs: Any) -> dict[str, Any]:
        """Converts hf_load_kwargs to a dict if they are an OmegaConf DictConfig."""
        if isinstance(hf_load_kwargs, omegaconf.DictConfig):
            hf_load_kwargs = omegaconf.OmegaConf.to_container(hf_load_kwargs)
        return hf_load_kwargs

    @pydantic.validator("subset_size", pre=True)
    def _validate_subset_size(cls, subset_size: Any) -> Union[int, dict[str, int]]:
        if isinstance(subset_size, omegaconf.DictConfig):
            subset_size = omegaconf.OmegaConf.to_container(subset_size)
        return subset_size


RetrievalCfg = TypeVar("RetrievalCfg", bound=RetrievalBuilderConfig)


class QuestionModel(pydantic.BaseModel):
    """Model a question for retrieval datasets."""

    id: int
    text: str
    section_ids: list[int]
    kb_id: int


class SectionModel(pydantic.BaseModel):
    """Model a section for retrieval datasets."""

    content: str
    title: str
    id: int
    kb_id: int


class RetrievalBuilder(dataset_builder.DatasetBuilder[datasets.DatasetDict, RetrievalCfg]):
    """Base class for retrieval dataset builders."""

    _builder_config: Type[RetrievalCfg] = RetrievalBuilderConfig
    _collate_config = retrieval_collate.RetrievalCollateConfig

    def __init__(self, config: RetrievalCfg = None, **kwargs: Any):
        if config is None:
            config = self._builder_config(**kwargs)
        if not isinstance(config, self._builder_config):
            if isinstance(config, pydantic.BaseModel):
                config = config.dict()
                config = self._builder_config(**config)
            else:
                raise TypeError(f"Expected config of type {self._builder_config}, got {type(config)}")

        self.config = config

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Returns the tokenizer for the dataset builder."""
        return self.config.tokenizer

    def _prep_map_kwargs(self, **overrides: Any) -> dict[str, Any]:
        always_on = dict(batched=True, with_indices=True)
        return {**self.config.prep_map_kwargs, **overrides, **always_on}

    def __call__(self) -> datasets.DatasetDict:
        corpus = self.get_corpus()
        self._validate_corpus(corpus)
        dset = self._build_dset()
        self._validate_dset(dset)
        dset = self._take_subset(dset)
        dset = self._add_row_indices(dset)
        return dset

    def _build_dset(self) -> datasets.DatasetDict:
        dset = datasets.load_dataset(
            path=self.config.name,
            name=self.config.subset_name,
            **self.config.hf_load_kwargs,
        )
        return dset

    @abc.abstractmethod
    def get_corpus(self) -> datasets.Dataset:
        raise NotImplementedError()

    def get_collate_fn(
        self, config: Optional[retrieval_collate.RetrievalCollateConfig] = None
    ) -> retrieval_collate.RetrievalCollate:
        if config is None:
            config = retrieval_collate.RetrievalCollateConfig()
        if isinstance(config, (dict, omegaconf.DictConfig)):
            config = retrieval_collate.RetrievalCollateConfig(**config)

        sections = self.get_corpus()
        collate_fn = retrieval_collate.RetrievalCollate(
            corpus=sections,
            tokenizer=self.config.tokenizer,
            config=config,
        )
        return collate_fn

    def _take_subset(self, dset: datasets.DatasetDict) -> datasets.DatasetDict:
        """Take a subset of the dataset."""
        if self.config.subset_size is None:
            return dset

        # pylint: disable=no-member
        rgn = np.random.RandomState(0)

        # define the subset size for each split
        subset_size = copy.copy(self.config.subset_size)
        if isinstance(subset_size, int):
            subset_size = {split: subset_size for split in dset.keys()}
        elif isinstance(subset_size, dict):
            assert set(subset_size.keys()) == set(dset.keys())
        else:
            raise TypeError(f"subset_size must be an int or a dict, not {type(self.config.subset_size)}")

        # sample the subsets
        new_mapped_qa = {}
        for split, size in subset_size.items():
            ids = rgn.choice(list(range(len(dset[split]))), size=size, replace=False)
            new_mapped_qa[split] = dset[split].select(ids)

        return datasets.DatasetDict(new_mapped_qa)

    def _add_row_indices(self, dset: dataset_builder.DorDD) -> dataset_builder.DorDD:
        """Add row index to each row."""
        prep_map_kwargs = self._prep_map_kwargs(batched=False, desc="Adding row indices")
        dset = dset.map(_add_row_idx, **prep_map_kwargs)
        return dset

    def _validate_dset(self, dset: datasets.DatasetDict) -> None:
        for d in dset.values():
            for i in range(10):
                row = d[i]
                QuestionModel(**row)

    def _validate_corpus(self, corpus: datasets.Dataset) -> None:
        for i in range(10):
            row = corpus[i]
            SectionModel(**row)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.__repr__()})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.__str__()})"


def _add_row_idx(_: dict, idx: int) -> dict[str, int]:
    return {retrieval_collate.ROW_IDX_COL_NAME: idx}
