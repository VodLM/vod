import timeit

import numpy as np
import rich

from raffle_ds_research.tools import c_tools

if __name__ == "__main__":
    n_points = 30
    max_n_unique = 10
    n_types = 3
    bach_size = 3
    batch_pids = np.random.randint(0, max_n_unique, size=(bach_size, n_points), dtype=np.uint64)
    batch_labels = np.random.randint(0, n_types, size=(bach_size, n_points), dtype=np.uint64)
    rich.print(
        {
            "batch_pid": batch_pids,
            "batch_label": batch_labels,
        }
    )

    batch_upids, batch_ulabels = c_tools.unique_by_label(
        batch_pids,
        batch_labels,
        n_labels=n_types,
        max_n_unique=max_n_unique,
    )
    rich.print(
        {
            "batch_upids": batch_upids,
            "batch_ulabels": batch_ulabels,
            "batch_upids.shape": batch_upids.shape,
        }
    )

    # timeit
    rtime = timeit.timeit(
        "c_tools.unique_by_label(batch_pids, batch_labels, n_labels=n_types, max_n_unique=max_n_unique)",
        globals=globals(),
        number=1_000,
    )
    rich.print(f"Time per call: {rtime / 10:.3f} ms (total: {rtime:.3f} s)")
