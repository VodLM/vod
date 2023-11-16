import pydantic
import vod_configs as vcfg
from loguru import logger
from typing_extensions import Self


class TrainValQueries(vcfg.StrictModel):
    """Models the training and validation queries."""

    train: list[vcfg.QueriesDatasetConfig]
    val: list[vcfg.QueriesDatasetConfig]


class SectionsDatasets(vcfg.StrictModel):
    """Models the sections."""

    sections: list[vcfg.SectionsDatasetConfig]


class TrainDatasets(vcfg.StrictModel):
    """Defines the training datasets."""

    queries: TrainValQueries
    sections: SectionsDatasets

    @pydantic.model_validator(mode="after")
    def _validate_links(self: Self) -> Self:
        """Check that the queries are pointing to valid sections."""
        section_ids = [s.identifier for s in self.sections.sections]
        linked_queries = {sid: [] for sid in section_ids}
        if len(linked_queries) != len(section_ids):
            raise ValueError(
                f"Duplicate section identifiers found: `{section_ids}`. "
                f"Make sure to assign each section dataset with a uniquer identifier."
            )

        # Assign queries to sections
        for query in self.queries.train + self.queries.val:
            if query.link not in section_ids:
                raise ValueError(
                    f"Query `{query.identifier}` points to invalid section ID `{query.link}`. "
                    f"Available section IDs: `{section_ids}`"
                )
            linked_queries[query.link].append(query.identifier)

        # Check that all sections have at least one query
        # Drop the sections that have no queries
        for sid in list(linked_queries.keys()):
            if not linked_queries[sid]:
                logger.warning(f"Section `{sid}` has no queries; dropping it.")
                self.sections.sections = [s for s in self.sections.sections if s.identifier != sid]

        return self


class BenchmarkDataset(vcfg.StrictModel):
    """Defines a benchmark."""

    queries: vcfg.QueriesDatasetConfig
    sections: vcfg.SectionsDatasetConfig


class ExperimentDatasets(vcfg.StrictModel):
    """Definse all datasets, including the base search config."""

    training: TrainDatasets
    benchmark: list[BenchmarkDataset]
