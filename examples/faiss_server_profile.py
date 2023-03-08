import itertools
from pathlib import Path

import rich
import seaborn as sns
from matplotlib import pyplot as plt

from examples.faiss_server import profile_faiss_server, ProfileArgs

sns.set(style="darkgrid")
colors = sns.color_palette()
markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "X", "D", "d"]

if __name__ == "__main__":
    base_args = {
        "n_calls": 1_000,
        "top_k": 100,
        "batch_size": 10,
        "index_factory": "Flat",
        "verbose": False,
    }
    args = {
        "dataset_size": [100, 1_000, 10_000, 100_000],
        "vector_size": [128, 512, 1024],
    }
    benchmarks = []
    arg_keys = list(args.keys())
    arg_permutations = list(itertools.product(*(args[k] for k in arg_keys)))
    for exp_arg in arg_permutations:
        exp_arg = dict(zip(arg_keys, exp_arg))
        rich.print(f"> [magenta bold]Running experiment with args:[/] {exp_arg}")
        exp_arg.update(base_args)
        benchmark = profile_faiss_server(ProfileArgs(**exp_arg))
        benchmark = {k: benchmark[k] for k in sorted(benchmark)}
        benchmark_str = {k: f"{v:.2f}" for k, v in benchmark.items()}
        benchmark_str = f"Benchmark({', '.join(f'{k}={v}' for k, v in benchmark_str.items())})"
        rich.print(f"  └── {benchmark_str}")
        benchmarks.append(
            {
                "args": exp_arg,
                "benchmark": benchmark,
            }
        )

    # make a grid plot with one column per `vector_size` value and a single row.
    # each plot will be a line plot with one line per `dataset_size` value as x-axis and time as y-axis.
    # each method will be a different color.
    vector_sizes = set(b["args"]["vector_size"] for b in benchmarks)
    methods = list(sorted(set(c for b in benchmarks for c in b["benchmark"].keys())))
    fig, axes = plt.subplots(1, len(vector_sizes), figsize=(len(vector_sizes) * 4, 4), sharey=True)  # type: ignore
    for i, vector_size in enumerate(vector_sizes):
        ax = axes[i]
        ax.set_xscale("log")
        ax.set_title(f"vector_size={vector_size}, batch_size={base_args['batch_size']}, top_k={base_args['top_k']}")
        ax.set_xlabel("dataset_size")
        if i == 0:
            ax.set_ylabel(f"time (ms/batch)")
        for j, method in enumerate(methods):
            x = []
            y = []
            for b in benchmarks:
                if b["args"]["vector_size"] == vector_size:
                    x.append(b["args"]["dataset_size"])
                    y.append(b["benchmark"][method])
            ax.plot(x, y, label=method, marker=markers[j], color=colors[j])

        if i == len(vector_sizes) - 1:
            ax.legend()

    output_path = Path("assets", "faiss_server_profile.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
