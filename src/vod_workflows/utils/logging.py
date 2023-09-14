import pathlib
from typing import Any, Iterable, Optional

import loguru
import rich.console
import transformers
from lightning.fabric.loggers.logger import Logger as FabricLogger
from rich import terminal_theme
from vod_tools import pretty


def log(
    metrics: dict[str, Any],
    loggers: Iterable[FabricLogger],
    step: Optional[int] = None,
    console: bool = False,
    header: None | str = None,
) -> None:
    """Log metrics to the trainer loggers and optionally to the console."""
    for logger in loggers:
        logger.log_metrics(metrics, step=step)

    if console:
        pretty.pprint_metric_dict(metrics, header=header)


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
        pretty.pprint_retrieval_batch(
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
