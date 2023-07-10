from __future__ import annotations

import functools
import hashlib
import pickle
import string
from typing import Any, Callable, Optional

import datasets
import numpy as np
import pydantic
import typing_extensions
from loguru import logger
from typing_extensions import Self, Type, TypeVar

from raffle_ds_research.core import config as core_config
from raffle_ds_research.tools import pipes, raffle_datasets

T = typing_extensions.TypeVar("T")


def _tokenize(text: str) -> list[str]:
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    return text.split()


def _filter_row_by_min_tokens(
    row: dict[str, Any],
    idx: Optional[int] = None,  # noqa: ARG001
    *,
    min_tokens: int,
    key: str,
    **kwargs: Any,
) -> bool:
    text = row[key]
    tokens = _tokenize(text)
    return len(tokens) >= min_tokens


class DatasetFactory:
    """Factory class to generate datasets."""

    loader: Callable[[], raffle_datasets.RetrievalDataset]
    config: core_config.DatasetFactoryConfig

    def __init__(
        self,
        loader: Callable[[], raffle_datasets.RetrievalDataset],
        config: dict | core_config.DatasetFactoryConfig,
    ):
        self.loader = loader
        self.config = _cast_pydantic_model(config, core_config.DatasetFactoryConfig)

    @classmethod
    def from_config(cls: Type[Self], config: dict | core_config.DatasetFactoryConfig) -> Self:
        """Instantiate a sequence of builders from a list of name."""
        config = _cast_pydantic_model(config, core_config.DatasetFactoryConfig)
        if config.name is None:
            raise ValueError("A name must be provided to instantiate a dataset builder.")
        loader = raffle_datasets.RetrievalDatasetLoader(name=config.name)
        return cls(loader=loader, config=config)

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self.config.name

    @property
    def split(self) -> str:
        """Return the split of the dataset."""
        return self.config.split

    def _prep_map_kwargs(self, **overrides: Any) -> dict[str, Any]:
        always_on = {"batched": True, "with_indices": True}
        return {**self.config.prep_map_kwargs, **always_on, **overrides}

    def get_sections(self) -> datasets.Dataset:
        """Returns the document sections for the dataset."""
        locator = f"{self.config.name}:{self.config.split}:sections"
        section_prep = pipes.Partial(
            pipes.template_pipe,
            template=self.config.templates["section"],
            input_keys=["title", "content", "language", "kb_id"],
            output_key="text",
        )

        # load the raw dataset and validate
        dset = self.loader()
        _validate_sections(dset.sections)

        # Filter the sections
        if self.config.min_section_tokens is not None:
            n_sections = len(dset.sections)
            dset.sections = dset.sections.filter(
                functools.partial(
                    _filter_row_by_min_tokens,
                    min_tokens=self.config.min_section_tokens,
                    key="content",
                ),
                **self._prep_map_kwargs(
                    batched=False,
                    desc=f"{locator}: Filtering sections",
                ),
            )
            logger.debug(f"Filtered {n_sections - len(dset.sections)} sections.")

        # preprocess the sections
        sections = dset.sections.map(
            section_prep,
            **self._prep_map_kwargs(desc=f"{locator}: Preprocessing sections"),
        )

        # add the group hash
        sections = _compute_group_hashes(
            sections,
            keys=self.config.group_keys,
            output_key=self.config.group_hash_key,
            **self._prep_map_kwargs(desc=f"{locator}: Computing group hashes", batched=False),
        )

        if self.config.filter_unused_sections:
            n_sections = len(dset.sections)
            sections = _filter_unused_sections(sections, self.get_qa_split())
            logger.debug(f"Filtered {n_sections - len(sections)} unused sections.")

        return sections

    def get_qa_split(self) -> datasets.Dataset:
        """Build and return the dataset."""
        locator = f"{self.config.name}:{self.config.split}:qa_split"
        qa_prep = pipes.Partial(
            pipes.template_pipe,
            template=self.config.templates["question"],
            input_keys=["text", "language", "kb_id"],
            output_key="text",
        )

        # load the raw dataset and validate
        dset = self.loader()
        qa_split = dset.qa_splits[self.config.split]
        _validate_questions(qa_split)

        # preprocess the questions
        qa_split = qa_split.map(
            qa_prep,
            **self._prep_map_kwargs(desc=f"{locator}: Preprocessing questions"),
        )

        # add the group hash
        qa_split = _compute_group_hashes(
            qa_split,
            keys=self.config.group_keys,
            output_key=self.config.group_hash_key,
            **self._prep_map_kwargs(desc=f"{locator}: Computing group hashes", batched=False),
        )

        # Optionally take a subset of rows
        qa_split = _take_subset(qa_split, self.config.subset_size)

        return qa_split

    def __call__(self, what: str = "qa_split") -> datasets.Dataset:
        """Build and return the dataset."""
        if what in {"qa_split", "question", "questions"}:
            return self.get_qa_split()
        if what in {"section", "sections"}:
            return self.get_sections()

        raise ValueError(f"Invalid value for `what`: {what}")

    def __repr__(self) -> str:
        """Returns a string representation of the dataset builder."""
        return f"{self.__class__.__name__}(config={self.config.__repr__()})"

    def __str__(self) -> str:
        """Returns a string representation of the dataset builder."""
        return f"{self.__class__.__name__}(config={self.config.__str__()})"


class _LazySectionLookupFilter:
    _lookup: Optional[set[int]]

    def __init__(
        self,
        qa_split: datasets.Dataset,
        qa_key: str = "section_ids",
        section_key: str = "id",
    ) -> None:
        self._qa_split = qa_split
        self._qa_key = qa_key
        self._section_key = section_key
        self._lookup = None

    def __call__(self, row: dict[str, Any]) -> bool:
        if self._lookup is None:
            self._lookup = {pid for row in self._qa_split for pid in row[self._qa_key]}  # type: ignore

        return row[self._section_key] in self._lookup


def _filter_unused_sections(sections: datasets.Dataset, qa_split: datasets.Dataset) -> datasets.Dataset:
    """Filter out sections that are not used in the QA split."""
    filter_op = _LazySectionLookupFilter(qa_split)
    return sections.filter(filter_op, desc="Filtering unused sections")


M = TypeVar("M", bound=pydantic.BaseModel)


def _cast_pydantic_model(config: dict | pydantic.BaseModel | M, model: Type[M]) -> M:
    if isinstance(config, model):
        return config

    if isinstance(config, pydantic.BaseModel):
        config = config.dict()

    return model(**config)


def _take_subset(dset: datasets.Dataset, subset_size: None | int) -> datasets.Dataset:
    """Take a subset of the dataset."""
    if subset_size is None:
        return dset

    rgn = np.random.RandomState(0)

    # sample the subsets
    ids = rgn.choice(list(range(len(dset))), size=subset_size, replace=False)
    return dset.select(ids)


class QuestionModel(pydantic.BaseModel):
    """Model a question for retrieval datasets."""

    id: int
    text: str
    section_ids: list[int]
    kb_id: int
    language: str


class SectionModel(pydantic.BaseModel):
    """Model a section for retrieval datasets."""

    content: str
    title: str
    id: int
    kb_id: int
    language: str


def _validate_questions(dset: datasets.Dataset) -> None:
    for i in range(core_config._N_VALID_SAMPLES):
        row = dset[i]
        QuestionModel(**row)


def _validate_sections(corpus: datasets.Dataset) -> None:
    for i in range(core_config._N_VALID_SAMPLES):
        row = corpus[i]
        SectionModel(**row)


class KeyHasher:
    """Hash key values into a single `int64`."""

    def __init__(self, keys: list[str], output_key: str = "group_hash") -> None:
        self.keys = keys
        self.output_key = output_key

    def __call__(self, row: dict[str, Any], idx: Optional[int] = None, **kwds: Any) -> dict[str, Any]:  # noqa: ARG002
        """Hash the keys."""
        subrow = [(k, row[k]) for k in sorted(self.keys)]
        h = hashlib.sha256(pickle.dumps(subrow, 1))
        obj_hash = h.hexdigest()
        np_int_hash = np.int64(int(obj_hash, 16) % np.iinfo(np.int64).max)
        return {self.output_key: np_int_hash}


def _compute_group_hashes(
    dataset: datasets.Dataset,
    keys: list[str],
    output_key: str,
    **kws: Any,
) -> datasets.Dataset:
    """Compute group hashes based on some list of `keys`."""
    hasher = KeyHasher(keys=keys, output_key=output_key)
    return dataset.map(hasher, **kws)
