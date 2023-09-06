import datasets
from vod_datasets.rosetta import models
from vod_datasets.tests import utils

from src.vod_datasets.rosetta.adapters.squad import SquadQueryAdapter, SquadQueryWithContextsAdapter


def test_squad_as_queries() -> None:
    """Test parsing the SQuaD dataset."""
    return utils.test_parse_dataset(
        datasets.load_dataset("squad_v2", split="validation[:10]"),  # type: ignore
        adapter_cls=SquadQueryAdapter,
        output_model=models.QueryModel,
    )  # type: ignore


def test_squad_as_queries_with_context() -> None:
    """Test parsing the SQuaD dataset."""
    return utils.test_parse_dataset(
        datasets.load_dataset("squad_v2", split="validation[:10]"),  # type: ignore
        adapter_cls=SquadQueryWithContextsAdapter,
        output_model=models.QueryWithContextsModel,
    )
