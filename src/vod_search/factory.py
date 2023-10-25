import ast
import copy
import os
import pathlib
import typing as typ

import datasets
import faiss
import lightning as L
import numpy as np
import torch
import vod_configs
import vod_types as vt
from loguru import logger
from vod_configs.utils.es_body import validate_es_body
from vod_search import base, es_search, faiss_search, qdrant_search
from vod_search.socket import find_available_port
from vod_tools import fingerprint
from vod_tools.misc.template import Template

from .base import ShardName
from .hybrid_search import HyrbidSearchMaster
from .sharded_search import ShardedSearchMaster

D = typ.TypeVar("D", bound=vt.Sequence)


def build_search_index(  # noqa: PLR0913
    index_type: str,
    *,
    sections: None | vt.DictsSequence,
    vectors: None | vt.Sequence[np.ndarray],
    config: dict[str, typ.Any],
    cache_dir: str | pathlib.Path,
    skip_setup: bool = False,
    barrier_fn: None | typ.Callable[[str], None] = None,
    serve_on_gpu: bool = False,
    free_resources: bool = False,
) -> base.SearchMaster:
    """Build a search Master client."""
    if sections is not None and vectors is not None and len(sections) != len(vectors):
        raise ValueError(f"Sections and vectors must have the same length. Found: {len(sections)} != {len(vectors)}")
    if index_type == "faiss":
        if vectors is None:
            raise ValueError("Must provide vectors for `faiss` index")
        return build_faiss_index(
            vectors=vectors,
            config=vod_configs.FaissFactoryConfig(**config),
            cache_dir=cache_dir,
            skip_setup=skip_setup,
            barrier_fn=barrier_fn,
            serve_on_gpu=serve_on_gpu,
            free_resources=free_resources,
        )
    if index_type == "elasticsearch":
        if sections is None:
            raise ValueError("Must provide sections for `elasticsearch` index")
        return build_elasticsearch_index(
            sections=sections,
            config=vod_configs.ElasticsearchFactoryConfig(**config),
            skip_setup=skip_setup,
            free_resources=free_resources,
        )

    if index_type == "qdrant":
        if vectors is None:
            raise ValueError("Must provide vectors for `qdrant` index")
        if sections is None:
            raise ValueError("Must provide sections for `qdrant` index")
        return build_qdrant_index(
            vectors=vectors,
            sections=sections,
            config=vod_configs.QdrantFactoryConfig(**config),
            skip_setup=skip_setup,
            free_resources=free_resources,
        )

    raise ValueError(f"Unknown index type `{index_type}`")


def build_elasticsearch_index(
    sections: vt.DictsSequence,
    config: dict | vod_configs.ElasticsearchFactoryConfig,
    skip_setup: bool = False,
    free_resources: bool = False,
) -> es_search.ElasticSearchMaster:
    """Build a sparse elasticsearch index."""
    if isinstance(config, dict):
        config = vod_configs.ElasticsearchFactoryConfig(**config)

    # Validate the ES body and use it to compute the index fingerprint
    es_body = validate_es_body(config.es_body, language=config.language)
    index_fingerprint = "-".join(
        [
            f"{fingerprint.fingerprint(sections)}",
            f"{config.fingerprint(exclude=['es_body'])}",
            f"{fingerprint.Hasher.hash(es_body)}",
        ]
    )
    template = Template(config.section_template)
    logger.info(
        f"Init. elasticsearch index `{index_fingerprint}`, "
        f"template: `{template.template}`, "
        f"section_id_key: `{config.section_id_key}`,"
        f"subset_id: `{config.subset_id_key}`"
    )
    row = next(iter(sections))
    if not template.is_valid(row):
        raise ValueError(f"Invalid template `{template.template}` for row with keys `{list(row.keys())}`")
    texts = (template.render(row) for row in iter(sections))
    if config.section_id_key is not None and config.section_id_key not in row:
        raise ValueError(f"Could not find `{config.section_id_key}` in `{row.keys()}`")
    sections_ids = (row[config.section_id_key] for row in iter(sections)) if config.section_id_key else None
    return es_search.ElasticSearchMaster(
        texts=texts,
        subset_ids=_fetch_subset_ids(sections, config.subset_id_key),
        section_ids=sections_ids,
        host=config.host,
        port=config.port,
        index_name=f"vod-{index_fingerprint}",
        input_size=len(sections),
        persistent=config.persistent,
        es_body=es_body,
        language=config.language,
        exist_ok=True,
        skip_setup=skip_setup,
        free_resources=free_resources,
    )


def build_faiss_index(
    vectors: vt.Sequence[np.ndarray],
    *,
    config: vod_configs.FaissFactoryConfig | dict,
    cache_dir: str | pathlib.Path,
    skip_setup: bool = False,
    barrier_fn: None | typ.Callable[[str], None] = None,
    serve_on_gpu: bool = False,
    free_resources: bool = False,
) -> faiss_search.FaissMaster:
    """Build a faiss index."""
    if isinstance(config, dict):
        config = vod_configs.FaissFactoryConfig(**config)
    if not torch.cuda.is_available():
        serve_on_gpu = False
    index_fingerprint = f"{fingerprint.fingerprint(vectors)}-{config.fingerprint()}"
    logger.info(
        f"Init. faiss index `{index_fingerprint}`, "
        f"factory: `{config.factory}`, "
        f"metric: `{vod_configs.FAISS_METRICS_INV[config.metric]}`, "
        f"train_size: `{config.train_size}`"
    )
    index_path = pathlib.Path(cache_dir, "indices", f"{index_fingerprint}.faiss")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_setup:
        if not index_path.exists():
            logger.info(f"Building faiss index at `{index_path}`")
            index = faiss_search.build_faiss_index(
                vectors=vectors,
                factory_string=config.factory,
                faiss_metric=config.metric,
                train_size=config.train_size,
                gpu_config=config.gpu,
            )
            logger.info(f"Saving faiss index to `{index_path}`")
            faiss.write_index(index, str(index_path))
            if not index_path.exists():
                raise FileNotFoundError(
                    f"{_rank_info()}Could not find index at `{index_path}` right after building it."
                )
        else:
            logger.info(f"Loading existing faiss index from `{index_path}`")

    if barrier_fn is not None:
        barrier_fn(f"faiss build: `{index_path.name}`")

    if not index_path.exists():
        raise FileNotFoundError(f"{_rank_info()}Could not find faiss index at `{index_path}`.")

    return faiss_search.FaissMaster(
        index_path=index_path,
        nprobe=config.nprobe,
        logging_level=config.logging_level,
        host=config.host,
        port=config.port,
        skip_setup=skip_setup,
        serve_on_gpu=serve_on_gpu,
        free_resources=free_resources,
    )


def build_qdrant_index(
    vectors: vt.Sequence[np.ndarray],
    sections: vt.DictsSequence,
    config: vod_configs.QdrantFactoryConfig | dict,
    skip_setup: bool = False,
    free_resources: bool = False,
) -> qdrant_search.QdrantSearchMaster:
    """Build a dense Qdrant index."""
    if isinstance(config, dict):
        config = vod_configs.QdrantFactoryConfig(**config)
    index_fingerprint = f"{fingerprint.fingerprint(vectors)}-{config.fingerprint()}"
    logger.info(
        f"Init. Qdrant index `{index_fingerprint}`, "
        f"vector: `{[len(vectors), *vectors[0].shape]}`, "
        f"subset_id_key: `{config.subset_id_key}`"
    )

    return qdrant_search.QdrantSearchMaster(
        vectors=vectors,
        subset_ids=_fetch_subset_ids(sections, config.subset_id_key),
        host=config.host,
        port=config.port,
        grpc_port=config.grpc_port,
        index_name=f"vod-{index_fingerprint}",
        persistent=config.persistent,
        qdrant_body=config.qdrant_body,
        search_params=config.search_params,
        exist_ok=True,
        skip_setup=skip_setup,
        force_single_collection=config.force_single_collection,
        free_resources=free_resources,
    )


def _fetch_subset_ids(sections: vt.DictsSequence, subset_id_key: None | str) -> None | typ.Generator[str, None, None]:
    if subset_id_key and subset_id_key in sections[0]:
        subset_ids = (row[subset_id_key] for row in iter(sections))
    else:
        if subset_id_key and subset_id_key not in sections[0]:
            logger.info(
                f"Could not find subset ID key `{subset_id_key}` in row `{sections[0].keys()}`. "
                "Filtering by subset ID will be disabled. This may be intentional."
            )
        subset_ids = None
    return subset_ids


def _init_dense_search_engine(
    sections: None | vt.DictsSequence,
    vectors: vt.Sequence[np.ndarray],
    config: vod_configs.FaissFactoryConfig | vod_configs.QdrantFactoryConfig,
    cache_dir: str | pathlib.Path,
    skip_setup: bool,
    barrier_fn: None | typ.Callable[[str], None],
    serve_on_gpu: bool,
    free_resources: bool = False,
) -> qdrant_search.QdrantSearchMaster | faiss_search.FaissMaster:
    if isinstance(config, vod_configs.FaissFactoryConfig):
        return build_faiss_index(
            vectors=vectors,
            config=config,
            cache_dir=cache_dir,
            skip_setup=skip_setup,
            barrier_fn=barrier_fn,
            serve_on_gpu=serve_on_gpu,
            free_resources=free_resources,
        )
    if isinstance(config, vod_configs.QdrantFactoryConfig):
        if sections is None:
            raise ValueError("`sections` must be provided when using `Qdrant` search engine")
        return build_qdrant_index(
            vectors=vectors,
            sections=sections,
            config=config,
            skip_setup=skip_setup,
            free_resources=free_resources,
        )

    raise TypeError(f"Unknown dense factory config type `{type(config)}`")


def build_hybrid_search_engine(  # noqa: C901, PLR0912, PLR0913
    *,
    sections: None | dict[ShardName, vt.DictsSequence],
    vectors: None | dict[ShardName, vt.Sequence[np.ndarray]],
    configs: dict[ShardName, vod_configs.HybridSearchFactoryConfig],
    cache_dir: str | pathlib.Path,
    dense_enabled: bool = True,
    sparse_enabled: bool = True,
    skip_setup: bool = False,
    barrier_fn: None | typ.Callable[[str], None] = None,
    serve_on_gpu: bool = False,
    free_resources: None | bool = None,
    fabric: None | L.Fabric = None,
    resolve_ports: bool = True,
) -> HyrbidSearchMaster:
    """Build a hybrid search engine."""
    if free_resources is None:
        free_resources = ast.literal_eval(os.environ.get("FREE_SEARCH_RESOURCES", "True"))
        free_resources = bool(free_resources)
    if skip_setup:
        free_resources = False

    # Get the list of shard names
    shard_names = _get_list_of_shard_names(configs, sections=sections, vectors=vectors)

    # Compute the dataset offsets given the list of shard names
    offsets: dict[ShardName, int] = _infer_and_validate_offsets(sections, vectors, shard_names)

    # Resolve missing ports
    if resolve_ports:
        if fabric is None:
            logger.debug("No fabric provided. Ports may not be synced across ranks.")
        configs = {shard: _resolve_ports(configs[shard], fabric=fabric) for shard in shard_names}

    servers: dict[str, ShardedSearchMaster] = {}
    if dense_enabled:
        dense_shards: dict[str, base.SearchMaster] = {}
        for shard_name in shard_names:
            if vectors is None:
                raise ValueError("`vectors` must be provided if `dense_enabled`")
            if configs[shard_name].engines.get("dense", None) is None:
                raise ValueError("`dense` must be provided if `dense_enabled`")

            dense_shards[shard_name] = _init_dense_search_engine(
                sections=sections[shard_name] if sections is not None else None,
                vectors=vectors[shard_name],
                config=configs[shard_name].engines["dense"],  # type: ignore
                cache_dir=cache_dir,
                skip_setup=skip_setup,
                barrier_fn=barrier_fn,
                serve_on_gpu=serve_on_gpu,
                free_resources=False,  # <- let the HyrbidSearchMaster handle this
            )
        servers["dense"] = ShardedSearchMaster(
            shards=dense_shards,
            offsets=offsets,
            skip_setup=skip_setup,
            free_resources=False,  # <- let the HyrbidSearchMaster handle this
        )

    if sparse_enabled:
        sparse_shards: dict[str, base.SearchMaster] = {}
        for shard_name in shard_names:
            if sections is None:
                raise ValueError("`sections` must be provided if `sparse_enabled`")
            if configs[shard_name].engines.get("sparse", None) is None:
                raise ValueError("`sparse` must be provided if `sparse_enabled`")

            sparse_shards[shard_name] = build_elasticsearch_index(
                sections=sections[shard_name],
                config=configs[shard_name].engines["sparse"],  # type: ignore
                skip_setup=skip_setup,
                free_resources=False,  # <- let the HyrbidSearchMaster handle this
            )

        servers["sparse"] = ShardedSearchMaster(
            shards=sparse_shards,
            offsets=offsets,
            skip_setup=skip_setup,
            free_resources=False,  # <- let the HyrbidSearchMaster handle this
        )

    if len(servers) == 0:
        raise ValueError("No search servers were enabled.")

    # Concatenate the sections
    concatenated_sections = _concatenate_dsets(
        [sections[shard] for shard in shard_names],  # type: ignore
    )

    return HyrbidSearchMaster(
        servers=servers,  # type: ignore
        skip_setup=skip_setup,
        free_resources=free_resources,
        shard_list=shard_names,
        sections=concatenated_sections,
    )


def _rank_info() -> str:
    rank = os.getenv("RANK", None)
    winfo = f"[{rank}] " if rank is not None else ""
    return winfo


def _resolve_ports(
    config: vod_configs.HybridSearchFactoryConfig,
    fabric: None | L.Fabric,
) -> vod_configs.HybridSearchFactoryConfig:
    """Resolve missing ports."""
    engines: dict[str, vod_configs.SingleSearchFactoryConfig] = copy.copy(config.engines)
    for key, engine in engines.items():
        if engine.port < 0:
            new_port = find_available_port()
            if fabric is not None:
                # Sync the port across all ranks
                new_port = fabric.broadcast(new_port, 0)
            engines[key] = engine.model_copy(update={"port": new_port})

    return config.model_copy(update={"engines": engines})


def _infer_offsets(x: dict[ShardName, typ.Sized], shard_names: list[ShardName]) -> dict[ShardName, int]:
    """Infer the offsets of a list of SizedDatasets."""
    if len(x) == 0:
        return {}
    offsets = [0, *np.cumsum([len(x[name]) for name in shard_names])[:-1]]
    return dict(zip(shard_names, offsets))


def _infer_and_validate_offsets(
    sections: None | dict[ShardName, typ.Sized],
    vectors: None | dict[ShardName, typ.Sized],
    shard_names: list[ShardName],
) -> dict[ShardName, int]:
    if sections is not None and vectors is not None:
        sec_offsets = _infer_offsets(sections, shard_names)
        vec_offsets = _infer_offsets(vectors, shard_names)
        if sec_offsets != vec_offsets:
            raise ValueError(
                f"The offsets of `sections` and `vectors` must match. Found Offsets: {sec_offsets} != {vec_offsets}"
            )
        return sec_offsets
    if sections is not None:
        return _infer_offsets(sections, shard_names)
    if vectors is not None:
        return _infer_offsets(vectors, shard_names)

    raise ValueError("Must provide either `sections` or `vectors`")


def _get_list_of_shard_names(
    configs: dict[ShardName, vod_configs.HybridSearchFactoryConfig],
    sections: None | dict[ShardName, vt.DictsSequence] = None,
    vectors: None | dict[ShardName, vt.Sequence[np.ndarray]] = None,
) -> list[ShardName]:
    """Get the list of shard names."""
    shard_names = list(configs.keys())
    if sections is not None and set(shard_names) != set(sections.keys()):
        raise ValueError("The keys of `sections` and `configs` must match.")
    if vectors is not None and set(shard_names) != set(vectors.keys()):
        raise ValueError("The keys of `vectors` and `configs` must match.")
    return shard_names


def _concatenate_dsets(parts: list[D]) -> D:
    """Concatenate a list of datasets."""
    if len(parts) > 1:
        if all(isinstance(p, datasets.Dataset) for p in parts):
            return datasets.concatenate_datasets(parts)  # type: ignore
        return vt.ConcatenatedSequences(parts)  # type: ignore

    return parts[0]
