# pylint: disable=missing-function-docstring
from __future__ import annotations

import collections
import tempfile

import numpy as np
import rich
from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics.dataset_factory import DatasetFactory
from raffle_ds_research.core.mechanics.search_engine import build_search_engine
from raffle_ds_research.tools import arguantic, index_tools
from tqdm import tqdm
from transformers import AutoTokenizer


class BenchFrank(arguantic.Arguantic):
    """Arguments for the script."""

    n_points: int = 100
    n_retrieval: int = 500
    bs: int = 10
    text_key: str = "text"
    group_key: str = "group_hash"
    section_id_key: str = "id"
    query_section_ids_key: str = "section_ids"
    metric_top_k: int = 3

    # Dataset info
    dset: str = "frank.B.en"
    split: str = "val"


def _reshape(x: list, bs: int, n_retrieval: int) -> list:
    """Reshape the flat values."""
    output = []
    for i in range(bs):
        output.append(x[i * n_retrieval : (i + 1) * n_retrieval])
    return output


def benchmark_frank(args: BenchFrank) -> dict:
    """Benchmark Frank using bm25."""
    rich.print(args)

    factory = DatasetFactory.from_config(
        {
            "name": args.dset,
            "split": args.split,
            "tokenizer": AutoTokenizer.from_pretrained("google/mt5-base"),
            "prep_map_kwargs": {"num_proc": 4},
        }
    )
    qas = factory.get_qa_split()
    sections = factory.get_sections()
    rich.print(sections)

    # Search and benchmark
    benchmark = collections.defaultdict(list)
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        build_search_engine(
            sections=sections,
            vectors=None,
            config=core_config.SearchConfig(
                text_key=args.group_key,
                group_key=args.group_key,
                bm25=index_tools.Bm25FactoryConfig(
                    text_key=args.text_key,
                    group_key=args.group_key,
                    section_id_key=args.section_id_key,
                ),
            ),
            cache_dir=tmpdir,
            faiss_enabled=False,
            bm25_enabled=True,
            skip_setup=False,
            close_existing_es_indices=False,
        ) as master,
    ):
        search_client = master.get_client()

        for i in tqdm(range(0, len(qas), args.bs)):
            qbatch = qas[i : i + args.bs]
            bs = len(qbatch[args.text_key])
            search_results = search_client.search(
                text=qbatch[args.text_key],
                group=qbatch[args.group_key],
                section_ids=qbatch[args.query_section_ids_key],
                top_k=args.n_retrieval,
            )
            search_results = search_results["bm25"]
            if i == 0:
                rich.print(search_results)
            indices = [int(k) for k in search_results.indices.reshape(-1)]
            q_secs = sections[indices]
            q_secs = {k: _reshape(v, bs, args.n_retrieval) for k, v in q_secs.items()}

            # sort the sections
            for j in range(bs):
                targets = qbatch["section_ids"][j]
                q_sec_j = {k: v[j] for k, v in q_secs.items()}
                scores = search_results.scores[j]
                scores = np.where(np.isnan(scores), -np.inf, scores)
                section_ids = [int(k) for k in q_sec_j["id"]]
                labels = [(s in targets) for s in section_ids]

                # get the top labels
                sorted_labels = [lab for _, lab in sorted(zip(scores, labels), reverse=True, key=lambda pair: pair[0])]
                top_labels = sorted_labels[: args.metric_top_k]

                # update the metrics
                hit = any(t for t in top_labels)
                benchmark["hits"].append(float(hit))

    return {k: sum(v) / len(v) for k, v in benchmark.items()}


if __name__ == "__main__":
    bench = benchmark_frank(BenchFrank())
    rich.print(bench)
