import asyncio
import time
import typing as typ
import warnings

import numpy as np
import rich
import vod_search as vs
import vod_types as vt
from vod_dataloaders.tools.fast import merge_search_results, normalize_scores_

from .utils import BlockTimer

FLOAT_INF_THRES = 3e12  # <-- values above this threshold are considered as inf


def multi_search(
    *,
    text: list[str],
    shards: list[str],
    vector: None | np.ndarray = None,
    subset_ids: None | list[list[str]] = None,
    section_ids: list[list[str]],
    top_k: int,
    clients: dict[str, vs.SearchClient],
    weights: dict[str, float],
) -> tuple[vt.RetrievalBatch, dict[str, np.ndarray]]:
    """Query a multisearch egine."""
    if "sparse" not in clients.keys():
        raise ValueError("The sparse client must be specified to lookup the positive sections.")

    # Create the search payloads - another payload is prepended to look up the positive sections
    lookup_payload = {
        "client": clients["sparse"],
        "vector": vector,
        "text": [""] * len(text),  # No search query is provided here. We are only looking for the `positive` sections
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
    meta = {}
    # Note - VL: I am not 100% sure this leverages asynchroneous execution effectively.
    #            This might something to look into for `asyncio` experts.
    with BlockTimer(name="search_time", output=meta):
        search_results = asyncio.run(_execute_search([lookup_payload] + payloads))

    # Unpack the results
    search_results = dict(zip(["lookup"] + client_names, search_results))

    # Discard the scores for the lookup client, discard the labels for all other clients
    search_results["lookup"].scores.fill(0.0)
    for name, result in search_results.items():
        if name == "lookup":
            continue
        result.labels = None

    # DEBUGGING - check for `-inf` scores in the `dense` results (hunting down problems with `faiss`)
    if "dense" in search_results:
        _check_inf_scores(search_results["dense"])

    # Retrieve the meta data
    for name, result in search_results.items():
        for key, value in result.meta.items():
            meta[f"{name}_{key}"] = value

    # normalize the scores
    normalize_scores_(search_results, offset=1.0)

    # Combine the results and add meta data
    combined_results, raw_scores = merge_search_results(
        search_results=search_results,
        weights={"lookup": 0.0, **weights},
    )
    combined_results.meta = meta
    raw_scores.pop("lookup")

    # Set -inf to the mask section (index -1) and replace the neg. indices with random indices
    is_masked = combined_results.indices < 0
    combined_results.scores[is_masked] = -np.inf
    for scores in raw_scores.values():
        scores[is_masked] = -np.inf

    return combined_results, raw_scores


async def _execute_search(payloads: list[dict[str, typ.Any]]) -> list[vt.RetrievalBatch]:
    """Execute the search asynchronously."""

    async def _async_search_one(args: dict[str, typ.Any]) -> vt.RetrievalBatch:
        client = args.pop("client")
        start_time = time.perf_counter()
        result = await client.async_search(**args)
        result.meta["search_time"] = time.perf_counter() - start_time
        return result

    futures = [_async_search_one(payload) for payload in payloads]
    return await asyncio.gather(*futures)


def _check_inf_scores(r: vt.RetrievalBatch) -> None:
    """Check whether there is any `+inf` scores in the results.

    This sometimes happens with `faiss`. I still don't understand why.
    """
    is_inf = r.scores >= FLOAT_INF_THRES
    if is_inf.any():
        rich.print(r)
        warnings.warn(
            f"Found {is_inf.sum()} ({is_inf.sum() / is_inf.size:.2%}) inf scores in the search results.",
            stacklevel=2,
        )
        r.scores[is_inf] = np.nan
