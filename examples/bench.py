# pylint: disable=missing-function-docstring
from __future__ import annotations

import collections
import tempfile

import rich
from transformers import AutoTokenizer

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core.mechanics.dataset_factory import DatasetFactory
from raffle_ds_research.core.mechanics.search_engine import build_search_engine
from raffle_ds_research.tools import arguantic
from raffle_ds_research.tools.utils.progress import IterProgressBar


class BenchFrank(arguantic.Arguantic):
    """Arguments for the script."""

    n_points: int = 100
    n_retrieval: int = 10
    bs: int = 10
    text_key: str = "text"
    group_key: str = "group_hash"
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
                bm25={},
            ),
            cache_dir=tmpdir,
            faiss_enabled=False,
            bm25_enabled=True,
            skip_setup=False,
            close_existing_es_indices=False,
        ) as master,
    ):
        search_client = master.get_client()

        prog = IterProgressBar()
        pbar = prog.add_task("Benchmarking", total=len(qas) // args.bs, info="Searching")
        for i in range(0, len(qas), args.bs):
            qbatch = qas[i : i + args.bs]
            bs = len(qbatch[args.text_key])
            search_results = search_client.search(
                text=qbatch[args.text_key],
                label=qbatch[args.group_key],
                top_k=args.n_retrieval,
            )
            search_results = search_results["bm25"]
            indices = [int(k) for k in search_results.indices.reshape(-1)]
            q_secs = sections[indices]
            q_secs = {k: _reshape(v, bs, args.n_retrieval) for k, v in q_secs.items()}

            # sort the sections
            for j in range(bs):
                targets = qbatch["section_ids"][j]
                q_sec_j = {k: v[j] for k, v in q_secs.items()}
                scores = search_results.scores[j]
                section_ids = [int(k) for k in q_sec_j["id"]]
                labels = [(s in targets) for s in section_ids]

                # get the top labels
                sorted_labels = [lab for _, lab in sorted(zip(scores, labels), reverse=True, key=lambda pair: pair[0])]
                top_labels = sorted_labels[: args.metric_top_k]

                # update the metrics
                hit = any(t for t in top_labels)
                benchmark["hits"].append(float(hit))

            # compute the metrics
            hit_rate = sum(benchmark["hits"]) / len(benchmark["hits"])
            prog.update(pbar, advance=1, info=f"Hit rate: {hit_rate:.3f}")

    return {k: sum(v) / len(v) for k, v in benchmark.items()}


if __name__ == "__main__":
    bench = benchmark_frank(BenchFrank())
    rich.print(bench)
