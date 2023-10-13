import typing as typ

import lightning as L
import torch
import vod_models
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger
from torch.utils import data as torch_data
from vod_models.monitoring import RetrievalMonitor
from vod_ops.utils import io
from vod_ops.utils.chrono import Chrono
from vod_ops.utils.format import format_pbar_info
from vod_ops.utils.trainer_state import TrainerState
from vod_tools import fingerprint
from vod_tools.misc.progress import IterProgressBar

from .val import validation_loop


def training_loop(  # noqa: C901, PLR0915
    state: TrainerState,
    *,
    module: vod_models.VodSystem,
    optimizer: torch.optim.Optimizer,
    fabric: L.Fabric,
    train_dl: torch_data.DataLoader,
    val_dl: torch_data.DataLoader,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    parameters: None | typ.MutableMapping[str, typ.Any] = None,
) -> TrainerState:
    """Train a ranker."""
    _check_frabric_wrapping(module, optimizer)
    optimizer.zero_grad()
    module.train()
    fabric.call("on_train_start", fabric=fabric, module=module)
    try:
        # infer the number of training and valid steps
        n_train_steps, n_val_steps = _infer_train_val_steps(state=state, fabric=fabric, val_dl=val_dl)
        with IterProgressBar(disable=not fabric.is_global_zero, redirect_stderr=False) as pbar:
            train_pbar = pbar.add_task(
                f"Period {1+state.pidx}",
                total=n_train_steps,
                auto_refresh=False,
                info=format_pbar_info(state, keys=state.config.pbar_keys),
            )
            train_metrics = None
            val_metrics = None
            chrono = None
            loop_step = 0
            pidx = state.pidx
            period_last_step = state.next_period_start_step
            state.must_stop = False
            optim_steps_counter = 0
            monitor = RetrievalMonitor(state.config.metrics)
            monitor.to(dtype=torch.float64, device=fabric.device)
            while not state.must_stop:
                for batch in train_dl:
                    if state.step >= state.next_period_start_step or state.must_stop:
                        break

                    # Callback - start of batch
                    fabric.call(
                        "on_train_batch_start",
                        fabric=fabric,
                        module=module,
                        batch=batch,
                        batch_idx=loop_step,
                    )

                    # Forward/backward pass
                    is_accumulating = (1 + loop_step) % state.config.accumulate_grad_batches != 0
                    model_output = _forward_backward(
                        batch=batch,
                        fwd_fn=module,
                        fabric=fabric,
                        loss_scaler=1 / state.config.accumulate_grad_batches,
                        no_backward_sync=is_accumulating,
                        fwd_kws={"mode": "evaluate", "compute_metrics": True},
                    )

                    # Callback - start of batch
                    fabric.call(
                        "on_train_batch_end",
                        fabric=fabric,
                        module=module,
                        batch=batch,
                        output=model_output,
                        batch_idx=loop_step,
                    )

                    # Update the training metrics
                    monitor.update(batch=batch, model_output=model_output)

                    # Optimization, logging, eval, and checkpointing
                    if not is_accumulating:
                        # Clip the gradients
                        if state.config.gradient_clip_val is not None:
                            fabric.clip_gradients(module, optimizer, max_norm=state.config.gradient_clip_val)

                        # Run an optimization step, reset the gradients and update the learning rate
                        optimizer.step()
                        optim_steps_counter += 1
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                        # Update the chrono
                        if chrono is not None:
                            chrono.stop()

                        # Log the training metrics
                        if state.step % state.config.log_interval == 0:
                            train_metrics = monitor.compute()  # Synchronize aggregators and compute metrics
                            fabric.log_dict(
                                metrics={
                                    # Log Trainer info
                                    "trainer/epoch": float(state.epoch),
                                    "trainer/period": float(pidx),
                                    **{f"trainer/diagnostics/{k}": v for k, v in batch.get("diagnostics", {}).items()},
                                    **{f"trainer/parameters/{k}": v for k, v in (parameters or {}).items()},
                                    # Log Model/Optimizer info
                                    "train/loss": model_output["loss"].detach(),
                                    **{f"train/{k}": v for k, v in train_metrics.items()},
                                    **{
                                        "train/model/diagnostics/{k}": v
                                        for k, v in model_output.get("diagnostics", {}).items()  # type: ignore
                                    },
                                    **{f"optimizer/{k}": v for k, v in _extract_learning_rates(optimizer).items()},
                                },
                                step=state.step,
                            )

                        # Increment the global step
                        state.step += 1
                        if state.step >= period_last_step:
                            state.must_stop = True

                        # Validation and checkpointing
                        if state.must_stop or state.step % min(n_train_steps, state.config.val_check_interval) == 0:
                            module.eval()
                            val_metrics = validation_loop(
                                module=module,
                                fabric=fabric,
                                state=state,
                                val_dl=val_dl,
                                n_steps=n_val_steps,
                                pbar=pbar,
                            )
                            module.train()

                            # Log the valuation metrics
                            fabric.log_dict(
                                metrics={f"val/{k.replace('.', '/')}": v for k, v in val_metrics.items()},
                                step=state.step,
                            )

                            # Save the model and training state
                            if state.config.checkpoint_path is not None:
                                io.save_training_state(
                                    checkpoint_path=state.config.checkpoint_path,
                                    fabric=fabric,
                                    model=module,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    trainer_state=state,
                                )

                        # Update the parameters given the state
                        if parameters is not None:
                            parameters.update(state.get_parameters())

                        # Update the progress bar
                        pbar.update(
                            train_pbar,
                            refresh=True,
                            completed=optim_steps_counter,
                            speed=chrono.get_avg_laps_per_second() if chrono is not None else None,
                            info=format_pbar_info(
                                state,
                                train_metrics=train_metrics,
                                val_metrics=val_metrics,
                                keys=state.config.pbar_keys,
                            ),
                        )

                        # Start the chono
                        if chrono is None:
                            chrono = Chrono()
                        chrono.start()

                    loop_step += 1

                # Update the Epoch counter
                state.epoch += 1
    except KeyboardInterrupt:
        logger.warning(
            f"Training period {1+state.pidx} (step={state.step}) " f"interrupted by user (KeyboardInterrupt)."
        )

    optimizer.zero_grad()

    # Save and Synch the model parameters
    # TODO: remove this
    if state.config.checkpoint_path is not None:
        logger.info("End of period. Syncing parameters.")
        io.save_training_state(
            checkpoint_path=state.config.checkpoint_path,
            fabric=fabric,
            model=module,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=state,
        )
        io.load_training_state(
            checkpoint_path=state.config.checkpoint_path,
            fabric=fabric,
            module=module,
        )

    fabric.call("on_train_end", fabric=fabric, module=module)
    logger.info(f"End of period. Model hash: `{fingerprint.fingerprint_torch_module(module)}`")
    return state


def _check_frabric_wrapping(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not fabric_wrappers.is_wrapped(model):
        raise RuntimeError("Ranker must be wrapped by lightning `Fabric`.")
    if not fabric_wrappers.is_wrapped(optimizer):
        raise RuntimeError("Optimizer must be wrapped by lightning `Fabric`.")


def _forward_backward(
    batch: typ.Mapping[str, torch.Tensor],
    *,
    fwd_fn: typ.Callable[[typ.Mapping[str, torch.Tensor]], typ.Mapping[str, torch.Tensor]],
    fabric: L.Fabric,
    loss_scaler: None | float = None,
    no_backward_sync: bool = False,
    fwd_kws: None | dict = None,
    backward_kws: None | dict[str, typ.Any] = None,
    loss_key: str = "loss",
) -> typ.Mapping[str, torch.Tensor]:
    """Run a forward pass followed by a backward pass."""
    fwd_kws = fwd_kws or {}
    fwd_out = fwd_fn(batch, **fwd_kws)

    # Compute the loss
    loss = fwd_out[loss_key]
    if loss_scaler is not None:
        loss *= loss_scaler

    # backward pass
    backward_kws = backward_kws or {}
    with fabric.no_backward_sync(fwd_fn, enabled=no_backward_sync):  # type: ignore
        fabric.backward(loss, **backward_kws)

    return fwd_out


def _infer_train_val_steps(
    *,
    state: TrainerState,
    fabric: L.Fabric,
    val_dl: torch_data.DataLoader,
) -> tuple[int, int]:
    max_steps = state.next_period_start_step or state.config.max_steps
    n_train_steps = max_steps - state.step
    if state.config.n_max_eval is None:
        n_val_steps = len(val_dl)
    else:
        eff_eval_bs = fabric.world_size * val_dl.batch_size  # type: ignore
        n_val_steps = min(len(val_dl), max(1, -(-state.config.n_max_eval // eff_eval_bs)))
    return n_train_steps, n_val_steps


def _extract_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {f"lr_{i}": param_group["lr"] for i, param_group in enumerate(optimizer.param_groups)}
