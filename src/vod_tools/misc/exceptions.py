import os
import typing as typ
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import stackprinter
from loguru import logger

T = typ.TypeVar("T")
P = typ.ParamSpec("P")


def dump_exceptions_to_file(func: typ.Callable[P, T]) -> typ.Callable[P, T]:
    """Decorator to catch exceptions and dump them to a file. Useful for debugging with multiprocessing."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            output = func(*args, **kwargs)
        except Exception as exc:
            log_file = Path(".exceptions")
            log_file /= datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            log_file /= datetime.now(tz=timezone.utc).strftime("%H-%M")
            log_file /= f"exception-{type(exc).__name__}-{func.__name__}-{os.getpid()}.txt"
            log_file.parent.mkdir(exist_ok=True, parents=True)
            logger.warning(f"Error in {type(func).__name__}. " f"See full stack in {log_file.absolute()}")
            with log_file.open("w") as f:
                f.write(stackprinter.format())

                # log args and kwargs
                _sep = "-" * 80
                header = f"{_sep}\n=== PARAMETERS ===\n{_sep}"
                f.write("\n\n" + header + "\n\n")
                for i, arg in enumerate(args):
                    f.write(f"args[{i}]: {type(arg)}\n{arg}\n\n")
                for k, v in kwargs.items():
                    f.write(f"kwargs[{k}]: {type(v)}\n{v}\n\n")
            raise exc

        return output

    return wrapper
