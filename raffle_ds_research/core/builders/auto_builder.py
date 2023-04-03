from typing import Any

from raffle_ds_research.core.builders import retrieval_builder
from raffle_ds_research.core.builders.frank_builder import FrankBuilder
from raffle_ds_research.core.builders.test_builder import TestBuilder

BUILDERS = {
    "frank": FrankBuilder,
    "test": TestBuilder,
}


def auto_builder(**config: Any) -> retrieval_builder.RetrievalBuilder:
    """Builds a dataset from a configuration."""
    try:
        builder_name = config["name"]
    except KeyError as exc:
        raise ValueError(f"Builder `name` not specified in config. Found keys: {config.keys()}") from exc

    builder_cls = BUILDERS[builder_name]
    return builder_cls(**config)
