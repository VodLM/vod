import copy
import typing as typ

import omegaconf as omg
import vod_configs as vcfg
from hydra.utils import instantiate
from typing_extensions import Type

from .datasets import (
    BenchmarkDataset,
    ExperimentDatasets,
    SectionsDatasets,
    TrainDatasets,
    TrainValQueries,
)
from .utils import (
    resolve_configs_list,
    to_dicts_list,
)

T = typ.TypeVar("T")
K = typ.TypeVar("K")
DsetCfg = typ.TypeVar("DsetCfg", bound=vcfg.QueriesDatasetConfig | vcfg.SectionsDatasetConfig)
HybridSearchFactory = typ.TypeVar(
    "HybridSearchFactory", bound=vcfg.HybridSearchFactoryConfig | vcfg.HybridSearchFactoryDiff
)


def _parse_base_options(
    config: typ.Mapping[str, typ.Any],
    base_options: None | vcfg.DatasetOptions = None,
) -> vcfg.DatasetOptions:
    base_options = base_options or vcfg.DatasetOptions()
    if "options" in config:
        base_options = base_options + vcfg.DatasetOptionsDiff(**config["options"])
    return base_options


def _parse_hybrid_search_factory(
    config: typ.Mapping[str, typ.Any], cls: Type[HybridSearchFactory]
) -> HybridSearchFactory:
    """Parse a dictionary / omegaconf configuration into a structured dict."""
    engines_configs = copy.deepcopy(config)
    if "engines" in engines_configs:
        config = engines_configs["engines"]

    # Parse the search engines configs
    factories_by_backend = {
        vcfg.HybridSearchFactoryConfig: vcfg._FactoryConfigsByBackend,
        vcfg.HybridSearchFactoryDiff: vcfg._FactoryDiffByBackend,
    }[cls]
    engines_configs = _parse_multi_search(config, factories_by_backend)

    if len(config) == 0:
        raise ValueError(f"Attempting to initialize a `{cls.__name__}` without engines.")

    return cls(engines=config)  # type: ignore


def _parse_base_search(
    config: typ.Mapping[str, typ.Any],
    base_search: vcfg.HybridSearchFactoryConfig | vcfg.SearchFactoryDefaults,
) -> vcfg.SearchFactoryDefaults | vcfg.HybridSearchFactoryConfig:
    if "search" in config:
        search_diff = _parse_hybrid_search_factory(config["search"], cls=vcfg.HybridSearchFactoryDiff)
        base_search = base_search + search_diff
    return base_search


def parse_dataset_config(
    config: typ.Mapping[str, typ.Any],
    *,
    cls: Type[DsetCfg],
    base_search: None | vcfg.HybridSearchFactoryConfig | vcfg.SearchFactoryDefaults = None,
    base_options: None | vcfg.DatasetOptions = None,
) -> DsetCfg:
    """Parse a config dictionary or dataset name into a structured config."""
    if isinstance(config, omg.DictConfig):
        params: dict[str, typ.Any] = omg.OmegaConf.to_container(config, resolve=True)  # type: ignore
    else:
        params: dict[str, typ.Any] = copy.deepcopy(config)  # type: ignore

    # Instantiate the `DatasetLoader` if any
    params["name_or_path"] = instantiate(params["name_or_path"])

    # parse base options and search
    params["options"] = _parse_base_options(params, base_options=base_options)
    if issubclass(cls, vcfg.SectionsDatasetConfig):
        params["search"] = _parse_base_search(params, base_search=base_search or vcfg.SearchFactoryDefaults())

    return cls(**params)


def _parse_dataset_configs_list(
    config: typ.Mapping[str, typ.Any] | list[typ.Mapping[str, typ.Any]], cls: Type[DsetCfg], **kws: typ.Any
) -> list[DsetCfg]:
    configs_list = to_dicts_list(config)
    # Resolve dynamic configurations (e.g. `__vars__`)
    configs_list = resolve_configs_list(configs_list)  # type: ignore
    # Parse each configuration
    return [parse_dataset_config(cfg, cls=cls, **kws) for cfg in configs_list]


def _parse_multi_search(
    config: typ.Mapping[str, typ.Any],
    sub_cls_by_backend: dict[K, Type[T]],
) -> dict[K, T]:
    if isinstance(config, omg.DictConfig):
        engine_params: dict[str, typ.Any] = omg.OmegaConf.to_container(config, resolve=True)  # type: ignore
    else:
        engine_params: dict[str, typ.Any] = copy.deepcopy(config)  # type: ignore

    engine_configs = {}
    for engine_name, cfg in engine_params.items():
        try:
            backend = cfg["backend"]
        except KeyError as exc:
            raise KeyError(
                f"Backend must be configured. Found configuration keys `{list(cfg.keys())}`. Missing=`backend`"
            ) from exc
        try:
            sub_cls = sub_cls_by_backend[backend]
        except KeyError as exc:
            raise KeyError(f"Unknown backend `{backend}`. Known backends: {list(sub_cls_by_backend.keys())}") from exc

        # Instantiate the specific engine
        engine_configs[engine_name] = sub_cls(**cfg)

    return engine_configs


def _parse_train_datasets(
    config: typ.Mapping[str, typ.Any],
    *,
    base_search: vcfg.HybridSearchFactoryConfig | vcfg.SearchFactoryDefaults,
    base_options: None | vcfg.DatasetOptions = None,
) -> TrainDatasets:
    """Parse dict or omegaconf.DictConfig into a structured config."""
    base_options = _parse_base_options(config, base_options=base_options)
    base_search = _parse_base_search(config, base_search=base_search)

    # Parse the sections
    sections = SectionsDatasets(
        sections=_parse_dataset_configs_list(
            config["sections"],
            cls=vcfg.SectionsDatasetConfig,
            base_search=base_search,
            base_options=base_options,
        ),
    )

    # Parse the queries
    queries_dsets = config["queries"]
    queries = TrainValQueries(
        train=_parse_dataset_configs_list(
            queries_dsets["train"],
            cls=vcfg.QueriesDatasetConfig,
            base_search=base_search,
            base_options=base_options,
        ),
        val=_parse_dataset_configs_list(
            queries_dsets["val"],
            cls=vcfg.QueriesDatasetConfig,
            base_search=base_search,
            base_options=base_options,
        ),
    )

    # Implicitely link the queries to the sections when there is only one section dataset
    if len(sections.sections) == 1:
        for query in queries.train + queries.val:
            with vcfg.AllowMutations(query):
                query.link = sections.sections[0].identifier

    return TrainDatasets(
        queries=queries,
        sections=sections,
    )


def _parse_benchmark_dataset(
    config: typ.Mapping[str, typ.Any],
    *,
    base_search: vcfg.HybridSearchFactoryConfig | vcfg.SearchFactoryDefaults,
    base_options: None | vcfg.DatasetOptions = None,
) -> BenchmarkDataset:
    """Parse dict or omegaconf.DictConfig into a structured config."""
    base_options = _parse_base_options(config, base_options=base_options)
    base_search = _parse_base_search(config, base_search=base_search)

    # Parse the sections
    sections = parse_dataset_config(
        config["sections"],
        cls=vcfg.SectionsDatasetConfig,
        base_search=base_search,
        base_options=base_options,
    )

    # Parse the queries
    queries = parse_dataset_config(
        config["queries"],
        cls=vcfg.QueriesDatasetConfig,
        base_search=base_search,
        base_options=base_options,
    )

    # Implicitely link the queries to the sections when there is only one section dataset
    with vcfg.AllowMutations(queries):
        queries.link = sections.identifier

    return BenchmarkDataset(
        queries=queries,
        sections=sections,
    )


def parse_experiment_datasets(
    config: typ.Mapping[str, typ.Any], base_options: None | vcfg.DatasetOptions = None
) -> ExperimentDatasets:
    """Parse the experiment datasets configuration."""
    search_defaults = vcfg.SearchFactoryDefaults(**config["search_defaults"])
    base_options = _parse_base_options(config, base_options=base_options)
    base_search = _parse_base_search(config, base_search=search_defaults)

    # Resolve dynamic configurations (i.e. `__vars__`)
    benchmark_configs = to_dicts_list(config["benchmark"])
    benchmark_configs = resolve_configs_list(benchmark_configs)

    return ExperimentDatasets(
        training=_parse_train_datasets(
            config["training"],
            base_options=base_options,
            base_search=base_search,
        ),
        benchmark=[
            _parse_benchmark_dataset(
                cfg,
                base_options=base_options,
                base_search=base_search,
            )
            for cfg in benchmark_configs
        ],
    )
