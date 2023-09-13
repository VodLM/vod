import collections
import typing as typ
from multiprocessing.managers import DictProxy

import lightning as L
import numpy as np
import rich
import torch
import vod_models
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from rich import progress
from torch.utils import data as torch_data
from vod_tools import pipes
from vod_tools.misc.progress import IterProgressBar
from vod_workflows.utils import helpers, io
from vod_workflows.utils.chrono import Chrono

from .callbacks import OnFirstBatchCallback


def training_loop(  # noqa: C901, PLR0915
    *,
    ranker: vod_models.Ranker,
    optimizer: torch.optim.Optimizer,
    trainer_state: helpers.TrainerState,
    fabric: L.Fabric,
    train_dl: torch_data.DataLoader,
    val_dl: torch_data.DataLoader,
    scheduler: None | torch.optim.lr_scheduler._LRScheduler = None,
    checkpoint_path: None | str = None,
    on_first_batch_fn: None | OnFirstBatchCallback = None,
    pbar_keys: None | list[str] = None,
    parameters: None | DictProxy = None,
) -> helpers.TrainerState:
    """Train a ranker."""
    _check_ranker_and_optimizer(ranker, optimizer)
    optimizer.zero_grad()
    ranker.train()

    if fabric.is_global_zero:
        rich.print(trainer_state)

    try:
        # infer the number of training and valid steps
        n_train_steps, n_val_steps = _infer_num_steps(state=trainer_state, fabric=fabric, val_dl=val_dl)
        with IterProgressBar(disable=not fabric.is_global_zero) as pbar:
            train_pbar = pbar.add_task(
                f"Period {1+trainer_state.pidx}",
                total=n_train_steps,
                info=_pbar_info(trainer_state, keys=pbar_keys),
            )
            eval_metrics = None
            chrono = None
            local_step = 0
            max_steps = trainer_state.period_max_steps or trainer_state.max_steps
            while trainer_state.step < max_steps:
                for batch in train_dl:
                    if trainer_state.step >= max_steps:
                        break

                    if on_first_batch_fn is not None and local_step == 0:
                        on_first_batch_fn(fabric, batch, ranker)

                    # Forward pass
                    is_accumulating = (1 + local_step) % trainer_state.accumulate_grad_batches != 0
                    step_metrics = ranker.gradients.forward_backward(
                        batch=batch,
                        fwd_fn=ranker,
                        fabric=fabric,
                        loss_scaler=1 / trainer_state.accumulate_grad_batches,
                        no_backward_sync=is_accumulating,
                    )

                    # Log the training metrics
                    if trainer_state.step % trainer_state.log_interval == 0:
                        fabric.log_dict(
                            metrics={
                                "trainer/epoch": float(trainer_state.epoch),
                                **{f"train/{k.replace('.', '/')}": v for k, v in step_metrics.items()},
                                **{f"parameter/{k}": v for k, v in _extract_learning_rates(optimizer).items()},
                            },
                            step=trainer_state.step,
                        )

                    # Optimization & eval step
                    if not is_accumulating:
                        # Clip the gradients
                        if trainer_state.gradient_clip_val is not None:
                            fabric.clip_gradients(ranker, optimizer, max_norm=trainer_state.gradient_clip_val)

                        # Run an optimization step, reset the gradients and update the learning rate
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                        # Update the chrono, the trainer state and the progress bar
                        if chrono is not None:
                            chrono.stop()
                        trainer_state.step += 1

                        # Update the parameters
                        if parameters is not None:
                            parameters.update(trainer_state.get_parameters())

                        # Update the progress bar
                        pbar.update(
                            train_pbar,
                            advance=1,
                            speed=chrono.get_avg_laps_per_second() if chrono is not None else None,
                            info=_pbar_info(
                                trainer_state,
                                train_metrics=step_metrics,
                                eval_metrics=eval_metrics,
                                keys=pbar_keys,
                            ),
                        )

                        # Validation
                        if (1 + trainer_state.step) % trainer_state.val_check_interval == 0:
                            optimizer.zero_grad()
                            eval_metrics = _validation_loop(
                                ranker=ranker,
                                fabric=fabric,
                                trainer_state=trainer_state,
                                val_dl=val_dl,
                                n_steps=n_val_steps,
                                pbar=pbar,
                            )
                            if checkpoint_path is not None:
                                logger.debug("Saving checkpoint to `{}`", checkpoint_path)
                                io.save_training_state(
                                    checkpoint_path=checkpoint_path,
                                    fabric=fabric,
                                    model=ranker,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    trainer_state=trainer_state,
                                )

                        # Start the chono
                        if chrono is None:
                            chrono = Chrono()
                        chrono.start()

                    local_step += 1
                trainer_state.epoch += 1
    except KeyboardInterrupt:
        logger.warning(
            f"Training period {1+trainer_state.pidx} (step={trainer_state.step}) "
            f"interrupted by user (KeyboardInterrupt)."
        )

    optimizer.zero_grad()

    # Save and Synch the model parameters
    if checkpoint_path is not None:
        logger.info("End of period. Saving and re-loading model from checkpoint (parameter sync).")
        io.save_training_state(
            checkpoint_path=checkpoint_path,
            fabric=fabric,
            model=ranker,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
        io.load_training_state(
            checkpoint_path=checkpoint_path,
            fabric=fabric,
            model=ranker,
        )
    logger.info(f"End of period. Model hash: `{pipes.fingerprint_torch_module(None, ranker)}`")
    return trainer_state


def _check_ranker_and_optimizer(ranker: vod_models.Ranker, optimizer: torch.optim.Optimizer) -> None:
    if not fabric_wrappers.is_wrapped(ranker):
        raise RuntimeError("Ranker must be wrapped by lightning `Fabric`.")
    if not fabric_wrappers.is_wrapped(optimizer):
        raise RuntimeError("Optimizer must be wrapped by lightning `Fabric`.")


def _infer_num_steps(
    *,
    state: helpers.TrainerState,
    fabric: L.Fabric,
    val_dl: torch_data.DataLoader,
) -> tuple[int, int]:
    max_steps = state.period_max_steps or state.max_steps
    n_train_steps = max_steps - state.step
    if state.n_max_eval is None:
        n_val_steps = len(val_dl)
    else:
        eff_eval_bs = fabric.world_size * val_dl.batch_size  # type: ignore
        n_val_steps = min(len(val_dl), max(1, -(-state.n_max_eval // eff_eval_bs)))
    return n_train_steps, n_val_steps


@torch.no_grad()
def _validation_loop(
    ranker: vod_models.Ranker,
    fabric: L.Fabric,
    trainer_state: helpers.TrainerState,
    val_dl: torch_data.DataLoader,
    n_steps: int,
    pbar: progress.Progress,
) -> dict[str, float | torch.Tensor]:
    ranker.eval()
    val_pbar = pbar.add_task("Validation", total=n_steps, info=f"{n_steps} steps")
    metrics = collections.defaultdict(list)
    for i, batch in enumerate(val_dl):
        output = ranker(batch, mode="evaluate")
        pbar.update(val_pbar, advance=1, taskinfo=_pbar_info(trainer_state, output))
        for k, v in output.items():
            metrics[k].append(_format_metric(v))
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


def _format_metric(v: typ.Any) -> typ.Any:  # noqa: ANN401
    if isinstance(v, torch.Tensor):
        return v.detach().mean().cpu()

    return v


def _extract_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {f"lr_{i}": param_group["lr"] for i, param_group in enumerate(optimizer.param_groups)}


def _pbar_info(
    state: helpers.TrainerState,
    train_metrics: None | dict[str, typ.Any] = None,
    eval_metrics: None | dict[str, typ.Any] = None,
    keys: None | list[str] = None,
) -> str:
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
                    suppl.append(f"train/{k}={train_metrics[k]:.3f}")

        if eval_metrics is not None:
            for k in keys:
                if k in eval_metrics:
                    suppl.append(f"val/{k}={eval_metrics[k]:.3f}")

        desc = f"[yellow]{' '.join(suppl)}[/yellow] • {desc}"

    return desc
