import rich
import torch
import transformers
from rich import table as rich_table

from .format import human_format_nb


def pprint_parameters_stats(
    model: torch.nn.Module | transformers.PreTrainedModel,
    header: None | str = None,
    console: None | rich.console.Console = None,
) -> None:
    """Print the fraction of parameters for each `dtype` in the model."""
    dtype_counts = {}
    dtype_trainable_counts = {}
    total_params = 0
    total_trainable_params = 0

    for param in model.parameters():
        dtype = param.dtype
        num_params = param.numel()
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + num_params
        total_params += num_params

        if param.requires_grad:
            dtype_trainable_counts[dtype] = dtype_trainable_counts.get(dtype, 0) + num_params
            total_trainable_params += num_params

    # Compute the fraction of (trainable) parameters for each dtype
    dtype_fractions = {dtype: count / total_params for dtype, count in dtype_counts.items()}
    dtype_trainable_fractions = {
        dtype: dtype_trainable_counts.get(dtype, 0) / count for dtype, count in dtype_counts.items()
    }

    # Create a table using rich
    table = rich_table.Table(
        show_header=True,
        title=header,
    )
    table.add_column("Dtype", style="bold cyan")
    table.add_column("Number of Parameters", style="green")
    table.add_column("Fraction of Parameters", style="yellow")
    table.add_column("Fraction of Trainable Parameters", style="magenta")

    for dtype in dtype_counts:
        count = dtype_counts[dtype]
        fraction = dtype_fractions[dtype]
        trainable_fraction = dtype_trainable_fractions[dtype]
        table.add_row(
            str(dtype),
            human_format_nb(count),
            f"{fraction:.2%}",
            f"{trainable_fraction:.2%}",
        )

    # Add a row for total parameters
    if len(dtype_counts) > 1:
        table.add_section()
        table.add_row(
            "Total",
            human_format_nb(total_params),
            f"{1.0:.2%}",
            f"{total_trainable_params/total_params:.2%}",
        )

    console = console or rich.console.Console()
    console.print(table)
