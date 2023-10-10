import functools
import pathlib
import tempfile
import typing as typ

import lightning as L
import rich
import torch
import transformers
import vod_configs
from loguru import logger
from vod_tools import pretty
from vod_workflows.utils.trainer_state import TrainerState

from .base import Callback

try:
    import wandb
except ImportError:
    wandb = None

P = typ.ParamSpec("P")
T = typ.TypeVar("T")


def safe_exec_decorator(fn: typ.Callable[P, T]) -> typ.Callable[P, None | T]:
    """A wrapper to safely execute a function and log exceptions using loguru."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None | T:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred in {fn.__name__}: {e}")
            return None

    return wrapper


class PprintBatch(Callback):
    """Pretty print the first training batch."""

    def __init__(
        self,
        *,
        output_file: None | str | pathlib.Path = None,
        log_to_wandb: bool = False,
        training: bool = True,
        validation: bool = False,
        testing: bool = False,
        is_retrieval: bool = False,
        tokenizer_encoder: transformers.PreTrainedTokenizerBase | vod_configs.TokenizerConfig | dict[str, typ.Any],
    ) -> None:
        super().__init__()
        self.output_file = pathlib.Path(output_file) if output_file is not None else None
        self.log_to_wandb = log_to_wandb
        self.training = training
        self.validation = validation
        self.testing = testing
        self.is_retrieval = is_retrieval

        # Encoder tokenizer for retrieval
        if not isinstance(tokenizer_encoder, (transformers.PreTrainedTokenizerBase, vod_configs.TokenizerConfig)):
            tokenizer_encoder = vod_configs.TokenizerConfig(**tokenizer_encoder)
        if isinstance(tokenizer_encoder, vod_configs.TokenizerConfig):
            tokenizer_encoder = tokenizer_encoder.instantiate()
        self.tokenizer_encoder = tokenizer_encoder

    @safe_exec_decorator
    def _on_batch_start(
        self,
        *,
        fabric: L.Fabric,
        module: torch.nn.Module,  # noqa:
        batch: dict[str, typ.Any],
        batch_idx: TrainerState,
    ) -> None:
        if batch_idx == 0 and fabric.is_global_zero:
            console = rich.console.Console(record=True)
            pretty.pprint_batch(batch, header="Training batch", console=console)
            if self.is_retrieval:
                pretty.pprint_retrieval_batch(
                    batch,
                    tokenizer=self.tokenizer_encoder,
                    skip_special_tokens=True,
                    console=console,
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    logfile = self.output_file or pathlib.Path(tmpdir) / "output.html"
                    console.save_html(str(logfile))
                    if self.log_to_wandb and wandb is not None and wandb.run is not None:
                        wandb.log({"data/retrieval-batch": wandb.Html(logfile.read_text(encoding="utf-8"))})

    def on_train_batch_start(self, *args: typ.Any, **kws: typ.Any) -> None:  # noqa: D102
        if self.training:
            self._on_batch_start(*args, **kws)

    def on_validation_batch_start(self, *args: typ.Any, **kws: typ.Any) -> None:  # noqa: D102
        if self.validation:
            self._on_batch_start(*args, **kws)

    def on_test_batch_start(self, *args: typ.Any, **kws: typ.Any) -> None:  # noqa: D102
        if self.testing:
            self._on_batch_start(*args, **kws)
