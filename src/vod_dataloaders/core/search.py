import asyncio
import copy
import time
import typing as typ
import warnings

import numpy as np
import rich
import tenacity
import vod_search as vs
import vod_types as vt
from vod_dataloaders.core import merge, normalize

from .utils import BlockTimer

FLOAT_INF_THRES = 3e12  # <-- values above this threshold are considered as +inf
LOOKUP_CLIENT_NAME = "lookup"


def async_hybrid_search(  # noqa: PLR0913
    *,
    text: list[str],
    shards: list[str],
    vector: None | np.ndarray = None,
    subset_ids: None | list[list[str]] = None,
    section_ids: list[list[str]],
    top_k: int,
    clients: dict[str, vs.SearchClient],
    weights: dict[str, float],
    lookup_engine_name: str = "sparse",
) -> tuple[vt.RetrievalBatch, dict[str, np.ndarray]]:
    """Query a hybrid search engine using `asyncio` and merge the search results into one.

    NOTE: The `lookup_engine_name` is used to lookup the golden/positive sections.
    """
    meta = {}
    if lookup_engine_name not in clients:
        raise ValueError(f"The `{lookup_engine_name}` client must be specified to lookup the golden/positive sections.")

    # Create the search payloads
    # NOTE: The `Lookup` payload is prepended to search the `lookup_engine_name` for the golden sections.
    lookup_payload = {
        "client": clients[lookup_engine_name],
        "vector": vector,
        "text": [""] * len(text),  # No search query is provided here. We are only looking for the `golden` sections
        "subset_ids": subset_ids,
        "ids": section_ids,
        "shard": shards,
        "top_k": top_k,
    }
    client_names = list(clients.keys())
    payloads = [
        {
            "client": clients[name],
            "vector": vector,
            "text": text,
            "subset_ids": subset_ids,
            "shard": shards,
            "top_k": top_k,
        }
        for name in client_names
    ]

    # Run the searches asynchronously
    with BlockTimer(name="search_time", output=meta):
        search_results = asyncio.run(_execute_search([lookup_payload] + payloads))

    # Unpack the results
    search_results = dict(zip([LOOKUP_CLIENT_NAME] + client_names, search_results))

    # Merge the search results into one
    search_results, raw_scores = _merge_search_results(search_results, weights)

    # Assign the `meta` dict to the `search_results` and return
    search_results.meta.update(meta)
    return search_results, raw_scores


def _merge_search_results(
    search_results: dict[str, vt.RetrievalBatch],
    weights: dict[str, float],
) -> tuple[vt.RetrievalBatch, dict[str, np.ndarray]]:
    """Merge the search results into one.

    This implementation is dependent of `async_hybrid_search` and requires the `lookup` client to be specified.
    """
    meta = {}
    if LOOKUP_CLIENT_NAME not in search_results:
        raise ValueError(f"The `{LOOKUP_CLIENT_NAME}` client must be specified to lookup the golden/positive sections.")

    # NOTE: Discard the scores for the `lookup` client, discard the labels for all other clients
    search_results[LOOKUP_CLIENT_NAME].scores.fill(0.0)
    for name, result in search_results.items():
        if name == LOOKUP_CLIENT_NAME:
            continue
        result.labels = None

    # DEBUG - check for `-inf` scores in the `dense` results (hunting down problems with `faiss`)
    if "dense" in search_results:
        _debug_inf_scores(search_results["dense"])

    # Retrieve the meta data for each search results, and add them with a prefix to the `meta` output dict
    for name, result in search_results.items():
        for key, value in result.meta.items():
            meta[f"{name}_{key}"] = value

    # Normalize the scores (make sure to substract the minimum score from all scores)
    # this is required for the scores to be in comparable ranges before applying the merge function.
    normalize.normalize_search_scores_(search_results, offset=0.0)

    # Combine the results using the weights. The combined scores
    # are a weighted sum of the input scores.
    # NOTE: The `lookup` client is weighted with `0.0` so it doesn't
    #       contribute to the final scores.
    # NOTE: The labels are only read from the `lookup` client since they
    #       have been set to `None` for all other clients.
    combined_results, raw_scores = merge.merge_search_results(
        search_results=search_results,
        weights={LOOKUP_CLIENT_NAME: 0.0, **weights},
    )
    raw_scores.pop(LOOKUP_CLIENT_NAME)

    # Assign the `meta` dict to the `combined_results` and return
    combined_results.meta = meta
    return combined_results, raw_scores


async def _execute_search(payloads: list[dict[str, typ.Any]]) -> list[vt.RetrievalBatch]:
    """Execute the search asynchronously."""

    @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_random_exponential(min=1, max=60))
    async def _async_search_one(args: dict[str, typ.Any]) -> vt.RetrievalBatch:
        """Execute a single search with retries.

        NOTE: make sure to copy args before passing them to this function,
              so we don't modify the original dict and ensure safe retries.
        """
        args = copy.copy(args)  # shallow copy to avoid modifying the original dict
        client = args.pop("client")
        start_time = time.perf_counter()
        result = await client.async_search(**args)
        result.meta["search_time"] = time.perf_counter() - start_time
        return result

    futures = [_async_search_one(payload) for payload in payloads]
    return await asyncio.gather(*futures)


def _debug_inf_scores(r: vt.RetrievalBatch) -> None:
    """Check whether there is any `+inf` scores in the results.

    NOTE -VL: This sometimes happens with `faiss`. I still don't understand why.
    """
    is_inf = r.scores >= FLOAT_INF_THRES
    if is_inf.any():
        rich.print(r)
        warnings.warn(
            f"Found {is_inf.sum()} ({is_inf.sum() / is_inf.size:.2%}) inf scores in the search results.",
            stacklevel=2,
        )
        r.scores[is_inf] = np.nan
