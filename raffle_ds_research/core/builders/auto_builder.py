from typing import Any

from raffle_ds_research.core.builders import FrankBuilder, retrieval_builder
from raffle_ds_research.core.builders.test_builder import TestBuilder

BUILDERS = {
    "frank": FrankBuilder,
    "test": TestBuilder,
}


def auto_builder(**config: Any) -> retrieval_builder.RetrievalBuilder:
    """Builds a dataset from a config."""
    try:
        builder_name = config["name"]
    except KeyError:
        raise ValueError(f"Builder `name` not specified in config. Found keys: {config.keys()}")

    builder_cls = BUILDERS[builder_name]
    return builder_cls(**config)
