import typing as typ

import torch
from vod_workflows.utils import helpers

T = typ.TypeVar("T")


def format_metric_value(
    v: T,
) -> T:
    """Format a metric value so it is safe for logging.."""
    if isinstance(v, torch.Tensor):
        return v.detach().mean().cpu()

    return v


def format_pbar_info(
    state: helpers.TrainerState,
    train_metrics: None | dict[str, typ.Any] = None,
    eval_metrics: None | dict[str, typ.Any] = None,
    keys: None | list[str] = None,
) -> str:
    """Format the metrics for the progress bar."""
    keys = keys or ["loss"]
    desc = (
        f"{1+state.step}/{state.period_max_steps} ({state.max_steps}) "
        f"• epoch={1+state.epoch} "
        f"• grad-acc={state.accumulate_grad_batches}"
    )
    if train_metrics or eval_metrics:
        suppl = []
        if train_metrics is not None:
            for k in keys:
                if k in train_metrics:
                    suppl.append(f"train/{k}={train_metrics[k]:.2f}")

        if eval_metrics is not None:
            for k in keys:
                if k in eval_metrics:
                    suppl.append(f"val/{k}={eval_metrics[k]:.2f}")

        desc = f"[yellow]{' '.join(suppl)}[/yellow] • {desc}"

    return desc
