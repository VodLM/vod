import typing as typ

import torch
from vod_ops.utils.trainer_state import TrainerState

T = typ.TypeVar("T")


def format_metric_value(v: T) -> T:
    """Format a metric value so it is safe for logging.."""
    if isinstance(v, torch.Tensor):
        return v.detach().mean().cpu()

    return v


def format_pbar_info(
    state: TrainerState,
    train_metrics: None | typ.Mapping[str, typ.Any] = None,
    val_metrics: None | typ.Mapping[str, typ.Any] = None,
    keys: None | list[str] = None,
) -> str:
    """Format the metrics for the progress bar."""
    keys = keys or ["loss"]
    desc = (
        f"{state.step}/{state.next_period_start_step} ({state.config.max_steps}) "
        f"• epoch={1+state.epoch} "
        f"• grad-acc={state.config.accumulate_grad_batches}"
    )
    if train_metrics or val_metrics:
        suppl = []
        if train_metrics is not None:
            for k in keys:
                if k in train_metrics:
                    suppl.append(f"train/{k}={train_metrics[k]:.2f}")

        if val_metrics is not None:
            for k in keys:
                if k in val_metrics:
                    suppl.append(f"val/{k}={val_metrics[k]:.2f}")

        desc = f"[yellow]{' '.join(suppl)}[/yellow] • {desc}"

    return desc
