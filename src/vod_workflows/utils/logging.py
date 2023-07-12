from __future__ import annotations

import pathlib
from typing import Any, Iterable, Optional

import loguru
import rich.console
import transformers
from lightning.fabric.loggers import Logger as FabricLogger
from raffle_ds_research.tools import pipes
from raffle_ds_research.utils.pretty import print_metric_groups
from rich import terminal_theme


def log(
    metrics: dict[str, Any],
    loggers: Iterable[FabricLogger],
    step: Optional[int] = None,
    console: bool = False,
    header: Optional[str] = None,
) -> None:
    """Log metrics to the trainer loggers and optionally to the console."""
    for logger in loggers:
        logger.log_metrics(metrics, step=step)

    if console:
        print_metric_groups(metrics, header=header)


def log_retrieval_batch(
    batch: dict[str, Any],
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_sections: int = 10,
    locator: str = "train",
) -> None:
    """Log a retrieval batch to wandb."""
    try:
        console = rich.console.Console(record=True)
        pipes.pprint_retrieval_batch(
            batch,
            header=f"{locator} retrieval batch",
            tokenizer=tokenizer,
            console=console,
            skip_special_tokens=True,
            max_sections=max_sections,
        )
        # console.print(
        #     "." * os.get_terminal_size().columns + "\n"
        # )  # <- this is a hack to make sure the console is flushed
        html_path = pathlib.Path("retrieval-batch.html")
        console.save_html(str(html_path), theme=terminal_theme.MONOKAI)

        import wandb

        wandb.log({f"trainer/{locator}/batch": wandb.Html(html_path.open())})
    except Exception as e:
        loguru.logger.debug(f"Could not log batch to wandb: {e}")
