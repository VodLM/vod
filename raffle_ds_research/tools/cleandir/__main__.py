from __future__ import annotations

import os
import shutil

import loguru
import pathlib
import lightning.pytorch as pl


class CleanableDirectory:
    """A context manager that creates a temporary directory and deletes it when exiting the context."""

    path: pathlib.Path

    def __init__(self, path: str | pathlib.Path, *, delete_existing: bool = False):
        self.delete_existing = delete_existing
        self.path = pathlib.Path(path)

    def __enter__(self) -> pathlib.Path:
        """Create the temporary directory and return its path."""
        self._create(self.path)
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN
        """Cleanup the temporary directory."""
        self._cleanup(self.path)

    @pl.utilities.rank_zero_only
    def _create(self, path: pathlib.Path) -> None:
        if self.delete_existing and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    @pl.utilities.rank_zero_only
    def _cleanup(self, path: pathlib.Path) -> None:
        for f in path.iterdir():
            loguru.logger.info(f"Deleting {f} ({os.stat(f).st_size / 1e6:.2f} MB)..")
            if f.is_dir():
                shutil.rmtree(f)
            else:
                os.remove(f)
        shutil.rmtree(path)
