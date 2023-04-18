from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Optional, Union

import datasets
import numpy as np
import omegaconf
import pydantic
import transformers
from torch.utils import data as torch_data

from raffle_ds_research.core.dataset_builders import retrieval_collate
from raffle_ds_research.core.dataset_builders.dataloader_sampler import DataloaderSampler
from raffle_ds_research.tools import dataset_builder, pipes, raffle_datasets

QUESTION_TEMPLATE = "Question: {{ question }}"
SECTION_TEMPLATE = "{% if title %}Title: {{ title }}. Document: {% endif %}{{ content }}"
DEFAULT_TEMPLATES = {
    "question": QUESTION_TEMPLATE,
    "section": SECTION_TEMPLATE,
}

_DEFAULT_SPLITS = ["train", "validation"]


@dataclasses.dataclass
class DatasetArgs:
    """Parse dataset names like `frank.A.en.pos` into a structured object."""

    name: str
    subset_name: Optional[str]
    language: str
    only_positive_sections: bool = False

    @classmethod
    def parse(cls, x: str) -> "DatasetArgs":
        if x.endswith(".pos"):
            only_positive_sections = True
            x = x.replace(".pos", "")
        else:
            only_positive_sections = False

        parts = x.split(".")
        if len(parts) == 2:
            name, language = parts
            subset_name = None
        elif len(parts) == 3:
            name, subset_name, language = parts
        else:
            raise ValueError(f"Invalid dataset name: {x}")
        return cls(
            name=name,
            subset_name=subset_name,
            language=language,
            only_positive_sections=only_positive_sections,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            name=self.name,
            subset_name=self.subset_name,
            language=self.language,
            only_positive_sections=self.only_positive_sections,
        )


class RetrievalBuilderConfig(pydantic.BaseModel):
    """Defines a configuration for a retrieval dataset builder."""

    class Config:
        """Pydantic config for the `RetrievalBuilderConfig` class."""

        extra = pydantic.Extra.forbid
        arbitrary_types_allowed = True

    splits: list[str] = _DEFAULT_SPLITS
    tokenizer: Optional[Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None
    index_max_top_k: int = 100
    n_sections: int = 32
    question_max_length: Optional[int] = 512
    section_max_length: Optional[int] = 512
    templates: Optional[dict[str, str]] = DEFAULT_TEMPLATES
    hf_load_kwargs: dict[str, Any] = dict(num_proc=4)
    prep_map_kwargs: dict[str, Any] = dict()
    subset_size: Optional[Union[int, dict[str, int]]] = None
    include_only_positive_sections: bool = False
    dl_sampler: Optional[DataloaderSampler] = None

    @pydantic.validator("splits", pre=True)
    def _validate_splits(cls, splits: Any) -> list[str]:
        """Converts splits to a list if they are an OmegaConf ListConfig."""
        if isinstance(splits, omegaconf.ListConfig):
            splits = omegaconf.OmegaConf.to_container(splits)
        if splits is None:
            splits = _DEFAULT_SPLITS
        return splits

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


class RetrievalBuilder(dataset_builder.DatasetBuilder[datasets.DatasetDict, RetrievalBuilderConfig]):
    """Base class for retrieval dataset builders."""

    _builder_config = RetrievalBuilderConfig
    _collate_config = retrieval_collate.RetrievalCollateConfig
    _load_fn: Callable[[], raffle_datasets.RetrievalDataset]
    name: str

    def __init__(
        self,
        loader: Callable[[], raffle_datasets.RetrievalDataset],
        name: str,
        config: dict | RetrievalBuilderConfig = None,
        **kwargs: Any,
    ):
        if config is None:
            config = self._builder_config(**kwargs)
        elif len(kwargs) > 0:
            raise ValueError(f"Received both config and kwargs: {kwargs}")

        if not isinstance(config, self._builder_config):
            if isinstance(config, pydantic.BaseModel):
                config = config.dict()
            else:
                raise TypeError(f"Expected config of type {self._builder_config}, got {type(config)}")

        self._load_fn = loader
        self.config = config
        self.name = name

    @classmethod
    def from_name(cls, name: str, **config: Any) -> "RetrievalBuilder":
        """Returns a dataset builder from a dataset name."""
        names = name.split("+") if "+" in name else [name]
        parts = []
        for part_name in sorted(names):
            args = DatasetArgs.parse(part_name)
            loader = raffle_datasets.DatasetLoader(
                name=args.name,
                subset_name=args.subset_name,
                language=args.language,
                only_positive_sections=args.only_positive_sections,
            )
            parts.append(loader)

        if len(parts) > 1:
            loader = raffle_datasets.ConcatenatedDatasetLoader(parts)
        else:
            loader = parts[0]
        return cls(loader=loader, name=name, **config)

    @property
    def splits(self) -> list[str]:
        """Returns the split names."""
        return self.config.splits

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Returns the tokenizer for the dataset builder."""
        return self.config.tokenizer

    def _prep_map_kwargs(self, **overrides: Any) -> dict[str, Any]:
        always_on = dict(batched=True, with_indices=True)
        return {**self.config.prep_map_kwargs, **overrides, **always_on}

    def get_corpus(self) -> datasets.Dataset:
        dset = self._load_fn()
        section_prep = pipes.Partial(
            pipes.template_pipe,
            template=self.config.templates["section"],
            input_keys=["title", "content"],
            output_key="text",
        )
        sections = dset.sections.map(
            section_prep,
            **self._prep_map_kwargs(desc=f"Preprocessing {self.name} sections"),
        )
        return sections

    def __call__(self) -> datasets.DatasetDict:
        dset = self._load_fn().qa_splits
        self._validate_questions(dset)

        # optionally take a subset of the splits
        found_splits = set(dset.keys())
        if not found_splits.issuperset(self.splits):
            raise ValueError(f"Expected splits {self.splits}, found {found_splits}")
        elif set(self.splits) != found_splits:
            dset = datasets.DatasetDict({split: dset[split] for split in self.splits})

        # Optionally take a subset of rows
        dset = self._take_subset(dset)

        # add the row index to each row
        dset = self._add_row_indices(dset)
        return dset

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

    def get_sampler(self, split: str = "train") -> Optional[torch_data.Sampler]:
        """Returns a sampler for the given split. The sampler add inverse-frequency weights to each label value."""

        if self.config.dl_sampler is None:
            return None

        dset = self()[split]
        return self.config.dl_sampler(dset)

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

    def _validate_questions(self, dset: datasets.DatasetDict) -> None:
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
