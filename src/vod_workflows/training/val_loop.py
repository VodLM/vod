import collections

import lightning as L
import numpy as np
import torch
import vod_models
from rich import progress
from torch.utils import data as torch_data
from vod_workflows.utils import helpers

from .utils import format_metric_value, format_pbar_info


@torch.no_grad()
def validation_loop(
    ranker: vod_models.Ranker,
    fabric: L.Fabric,
    trainer_state: helpers.TrainerState,
    val_dl: torch_data.DataLoader,
    n_steps: int,
    pbar: progress.Progress,
) -> dict[str, float | torch.Tensor]:
    """Run a validation loop."""
    ranker.eval()
    val_pbar = pbar.add_task("Validation", total=n_steps, info=f"{n_steps} steps")
    metrics = collections.defaultdict(list)
    for i, batch in enumerate(val_dl):
        output = ranker(batch, mode="evaluate")
        pbar.update(val_pbar, advance=1, taskinfo=format_pbar_info(trainer_state, output))
        for k, v in output.items():
            metrics[k].append(format_metric_value(v))
        if i >= n_steps:
            break

    # aggregate metrics
    metrics = {k: np.mean(v) for k, v in metrics.items()}

    # log metrics
    fabric.log_dict(
        metrics={f"val/{k.replace('.', '/')}": v for k, v in metrics.items()},
        step=trainer_state.step,
    )

    pbar.remove_task(val_pbar)
    ranker.train()
    return dict(metrics)
