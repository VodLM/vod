from __future__ import annotations

import functools
from functools import partial
from typing import Literal, Optional

import datasets

from raffle_ds_research.core.dataset_builders import retrieval_builder
from raffle_ds_research.tools import pipes
from raffle_ds_research.tools.raffle_datasets import frank


class FrankBuilderConfig(retrieval_builder.RetrievalBuilderConfig):
    """Configures a dataset builder for Frank."""

    name: Literal["frank"] = "frank"
    subset_name: frank.FrankSplitName = "A"


class FrankBuilder(retrieval_builder.RetrievalBuilder[FrankBuilderConfig]):
    """Builds a Frank dataset for retrieval."""

    def _load_frank_split(self, frank_split: frank.FrankSplitName) -> frank.HfFrankPart:
        return frank.load_frank(
            self.config.language, split=frank_split, only_positive_sections=self.config.include_only_positive_sections
        )

    def _build_dset(self) -> datasets.DatasetDict:
        frank_split = self._load_frank_split(self.config.subset_name)
        return frank_split.qa_splits

    def get_corpus(self) -> datasets.Dataset:
        frank_split = self._load_frank_split(self.config.subset_name)
        sections = frank_split.sections.map(
            self._get_sections_preprocessing(),
            **self._prep_map_kwargs(desc=f"Preprocessing Frank ({self.config.subset_name}) sections"),
        )
        return sections

    def _get_sections_preprocessing(self) -> pipes.Pipe | functools.partial:
        section_prep = partial(
            pipes.template_pipe,
            template=self.config.templates["section"],
            input_keys=["title", "content"],
            output_key="text",
        )
        return section_prep
