import lightning as L
import torch
import vod_models
from rich import progress
from torch.utils import data as torch_data
from vod_workflows.utils.trainer_state import TrainerState

from .utils import RunningAverage, format_pbar_info


@torch.no_grad()
def validation_loop(
    module: vod_models.VodSystem,
    fabric: L.Fabric,
    state: TrainerState,
    val_dl: torch_data.DataLoader,
    n_steps: int,
    pbar: progress.Progress,
) -> dict[str, float | torch.Tensor]:
    """Run a validation loop."""
    fabric.call("on_validation_start", fabric=fabric, module=module)
    val_pbar = pbar.add_task("Validation", total=n_steps, info=f"{n_steps} steps")
    agg_metrics = RunningAverage()
    for i, batch in enumerate(val_dl):
        # Evaluate the module
        fabric.call("on_validation_batch_start", fabric=fabric, module=module, batch=batch, batch_idx=i)
        output = module(batch, mode="evaluate")
        fabric.call("on_validation_batch_end", fabric=fabric, module=module, batch=batch, output=output, batch_idx=i)

        # Update progress bar and store metrics
        pbar.update(val_pbar, refresh=True, advance=1, taskinfo=format_pbar_info(state, output))
        agg_metrics.update(output)
        if i >= n_steps:
            break

    # Cleanup and return
    pbar.remove_task(val_pbar)
    fabric.call("on_validation_end", fabric=fabric, module=module)
    return agg_metrics.get()
