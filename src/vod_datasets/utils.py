import os

import datasets
import fsspec
import gcsfs


def init_gcloud_filesystem() -> fsspec.AbstractFileSystem:
    """Initialize a GCS filesystem."""
    try:
        token = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError as exc:
        raise RuntimeError("Missing `GOOGLE_APPLICATION_CREDENTIALS` environment variables. ") from exc
    try:
        project = os.environ["GCLOUD_PROJECT_ID"]
    except KeyError as exc:
        raise RuntimeError("Missing `GCLOUD_PROJECT_ID` environment variables. ") from exc
    return gcsfs.GCSFileSystem(token=token, project=project)


def _fetch_queries_split(queries: datasets.DatasetDict, split: None | str) -> datasets.Dataset | datasets.DatasetDict:
    if split is None or split in {"all"}:
        return queries

    normalized_split = {
        "val": "validation",
    }.get(split, split)

    return queries[normalized_split]
