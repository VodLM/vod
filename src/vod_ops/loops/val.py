import typing as typ

import lightning as L
import torch
import vod_models
from rich import progress
from torch.utils import data as torch_data
from vod_models.monitoring import RetrievalMonitor
from vod_ops.utils.format import format_pbar_info
from vod_ops.utils.trainer_state import TrainerState


@torch.no_grad()
def validation_loop(
    module: vod_models.VodSystem,
    fabric: L.Fabric,
    state: TrainerState,
    val_dl: torch_data.DataLoader,
    n_steps: int,
    pbar: progress.Progress,
) -> typ.Mapping[str, torch.Tensor]:
    """Run a validation loop."""
    fabric.call("on_validation_start", fabric=fabric, module=module)
    val_pbar = pbar.add_task("Validation", total=n_steps, info=f"{n_steps} steps")
    monitor = RetrievalMonitor(state.config.metrics)
    monitor.to(dtype=torch.float64, device=fabric.device)
    for i, batch in enumerate(val_dl):
        # Evaluate the module
        fabric.call("on_validation_batch_start", fabric=fabric, module=module, batch=batch, batch_idx=i)
        output = module(batch, mode="evaluate")
        fabric.call("on_validation_batch_end", fabric=fabric, module=module, batch=batch, output=output, batch_idx=i)

        # Update progress bar and store metrics
        pbar.update(val_pbar, refresh=True, advance=1, taskinfo=format_pbar_info(state, output))
        monitor.update(batch=batch, model_output=output)
        if i >= n_steps:
            break

    # Cleanup and return
    pbar.remove_task(val_pbar)
    fabric.call("on_validation_end", fabric=fabric, module=module)
    return monitor.compute(synchronize=True)
