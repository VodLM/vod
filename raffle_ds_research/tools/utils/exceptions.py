from __future__ import annotations

import os
from datetime import datetime
from functools import wraps
from pathlib import Path

import stackprinter
from loguru import logger


def dump_exceptions_to_file(func):
    """Decorator to catch exceptions and dump them to a file.
    Useful for debugging with multiprocessing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
        except Exception as e:
            log_file = Path(".exceptions")
            log_file /= datetime.now().strftime("%Y-%m-%d")
            log_file /= datetime.now().strftime("%H-%M")
            log_file /= f"exception-{type(e).__name__}-{func.__name__}-{os.getpid()}.txt"
            log_file.parent.mkdir(exist_ok=True, parents=True)
            logger.warning(f"Error in {type(func).__name__}. " f"See full stack in {log_file.absolute()}")
            with open(log_file, "w") as f:
                f.write(stackprinter.format())

                # log args and kwargs
                _sep = "-" * 80
                header = f"{_sep}\n=== PARAMETERS ===\n{_sep}"
                f.write("\n\n" + header + "\n\n")
                for i, arg in enumerate(args):
                    f.write(f"args[{i}]: {type(arg)}\n{arg}\n\n")
                for k, v in kwargs.items():
                    f.write(f"kwargs[{k}]: {type(v)}\n{v}\n\n")
            raise e

        return output

    return wrapper
