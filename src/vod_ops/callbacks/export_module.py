import os
import pathlib
import warnings

import fsspec
import gcsfs
import lightning as L
import torch
import transformers
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger

from .base import Callback


class ExportModule(Callback):
    """Pretty print the first training batch."""

    def __init__(
        self,
        *,
        output_dir: str | pathlib.Path = "exported-module",
        include_files: None | list[str] = None,  # Include additional files in the export
        upload_path: None | str = None,  # Upload the exported module to a remote location
        only_submodule: None | str = None,
        submodules: None | list[str],
        on_fit_end: bool = True,  # at the end the whole training routine
        on_train_end: bool = False,  # at the end of each training period
    ) -> None:
        super().__init__()
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.include_files = include_files or []
        self.upload_path = upload_path
        self.only_submodule = only_submodule
        if only_submodule is not None and submodules is not None:
            submodules = []
            warnings.warn("Both `only_submodule` and `submodules` are set. Ignoring `submodules`.")  # noqa: B028
        self.submodules = submodules

        # triggers
        self._on_fit_end = on_fit_end
        self._on_train_end = on_train_end

    def on_fit_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102, ARG002
        if self._on_fit_end:
            self._export_module(module=module)

    def on_train_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102, ARG002
        if self._on_train_end:
            self._export_module(module=module)

    def _export_module(self, *, module: torch.nn.Module) -> None:
        """Export the model.

        # TODO: handle sharded models (e.g., FSDP, DeepSpeed)
        """
        if fabric_wrappers.is_wrapped(module):
            module = module.module

        # Export the module
        if self.only_submodule is not None:
            submodule_ = getattr(module, self.only_submodule, None)
            if submodule_ is None:
                raise ValueError(f"Submodule `{self.only_submodule}` not found in `{type(module).__name__}`")
            _export_module(submodule_, self.output_dir, self.include_files)
        elif self.submodules is None:
            _export_module(module, self.output_dir, self.include_files)
        else:
            for submodule in self.submodules:
                submodule_ = getattr(module, submodule, None)
                if submodule_ is None:
                    logger.debug(f"Submodule `{submodule}` not found in `{type(module).__name__}`")
                    continue
                _export_module(submodule_, self.output_dir / submodule, self.include_files)

        # Upload the module
        if self.upload_path is not None:
            _upload_dir(self.output_dir, self.upload_path)


def _export_module(
    module: torch.nn.Module | transformers.PreTrainedModel,
    output_dir: pathlib.Path,
    include_files: list[str],
) -> None:
    """Export a module."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the extra files
    for f in include_files:
        logger.info(f"Copying `{f}` to `{output_dir.absolute()}`")
        if not pathlib.Path(f).exists():
            warnings.warn(f"File `{f}` does not exist", stacklevel=2)
        if pathlib.Path(f).is_dir():
            warnings.warn(f"Cannot copy directory `{f}`. Only files are supported", stacklevel=2)
        if pathlib.Path(f).is_symlink():
            warnings.warn(f"Cannot copy symlink `{f}`. Only files are supported", stacklevel=2)
        os.system(f"cp {f} {output_dir.absolute()}")

    # Handle `transformers.PreTrainedModel`
    if isinstance(module, transformers.PreTrainedModel):
        logger.info(f"Exporting module `{type(module).__name__}` to `{output_dir.absolute()}` (HF pretrained)")
        module.save_pretrained(output_dir)
    else:
        logger.info(f"Exporting module `{type(module).__name__}` to `{output_dir.absolute()}` (torch state dict)")
        torch.save(module.state_dict(), output_dir / "model.pt")


def init_gcloud_filesystem() -> fsspec.AbstractFileSystem:
    """Initialize a GCS filesystem."""
    try:
        token = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError as exc:
        raise RuntimeError("Missing `GOOGLE_APPLICATION_CREDENTIALS` environment variables. ") from exc
    try:
        project = os.environ["GCLOUD_PROJECT_ID"]
    except KeyError as exc:
        raise RuntimeError("Missing `GCLOUD_PROJECT_ID` environment variables. ") from exc
    return gcsfs.GCSFileSystem(token=token, project=project)


FILESYSTEMS = {
    "gs": init_gcloud_filesystem,
}


def _upload_dir(local_dir: pathlib.Path, remote_path: str) -> None:
    """Upload a directory to a remote path."""
    if "://" not in remote_path:
        raise ValueError(f"Invalid remote path `{remote_path}`. Must be of the form `protocol://path`")
    protocol, rpath = remote_path.split("://", 1)
    if protocol not in FILESYSTEMS:
        raise ValueError(f"Invalid protocol `{protocol}`. Must be one of {list(FILESYSTEMS.keys())}")

    # Initialize the filesystem
    fs = FILESYSTEMS[protocol]()

    # Upload the files
    logger.info(f"Uploading `{local_dir}` to `{remote_path}`")
    fs.put(
        str(local_dir),
        rpath,
        recursive=True,
    )
