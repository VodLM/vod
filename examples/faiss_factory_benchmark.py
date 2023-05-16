import itertools
import json
import pathlib

import rich
from loguru import logger

from examples.faiss_factory import Args, build_and_eval_index

if __name__ == "__main__":
    output_file = pathlib.Path("faiss_factory_benchmark.jsonl")
    logger.info(f"Writing results to `{output_file.absolute()}`")
    if output_file.exists():
        output_file.unlink()
    base_config = {
        "train_size": 1_000_000,
        "verbose": False,
    }

    parameter_groups = [
        [
            {"nvecs": 1_000_000},
            {"nvecs": 10_000_000},
        ],
        [{"use_float16": True}, {"use_float16": False}],
        [{"serve_on_gpu": False}, {"serve_on_gpu": True}],
        [
            {"factory": "IVFauto,Flat"},
            {"factory": "IVFauto,PQ32x8"},
            {"factory": "OPQ32,IVFauto,PQ32x8"},
            {"factory": "OPQ32_512,IVFauto,PQ32x8"},
        ],
    ]
    # multiply the parameters into a flat list
    parameters = [{k: v for g in v for k, v in g.items()} for v in itertools.product(*parameter_groups)]
    rich.print(parameters)

    for p in parameters:
        rich.print(p)
        args = Args(**base_config, **p)
        try:
            r = build_and_eval_index(args)
            r["status"] = "success"
        except Exception as e:
            rich.print(f"[red]Failed to run with {p}")
            r = {"status": "failed", "error": str(e), "args": args.dict()}
        with output_file.open("a") as f:
            f.write(json.dumps(r) + "\n")
        rich.print(r)

    logger.info(f"Results written to `{output_file.absolute()}`")
