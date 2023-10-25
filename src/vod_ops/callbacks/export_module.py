import os
import pathlib
import typing as typ
import warnings

import fsspec
import gcsfs
import lightning as L
import omegaconf as omg
import torch
import transformers
import vod_configs
from lightning.fabric import wrappers as fabric_wrappers
from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger

from .base import Callback

ConfLike = typ.Union[dict[str, typ.Any], omg.DictConfig]


class ExportModule(Callback):
    """Pretty print the first training batch."""

    def __init__(
        self,
        *,
        output_dir: str | pathlib.Path = "exported-module",
        include_files: None | list[str] = None,  # Include additional files in the export
        upload_path: None | str = None,  # Upload the exported module to a remote location
        submodules: None
        | str
        | list[str],  # When setting this, only export the given submodules, each to a separate dir
        on_fit_end: bool = True,  # at the end the whole training routine
        on_train_end: bool = False,  # at the end of each training period
        tokenizers: None
        | transformers.PreTrainedTokenizer
        | ConfLike
        | dict[str, transformers.PreTrainedTokenizer | dict[str, ConfLike]] = None,
    ) -> None:
        super().__init__()
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.include_files = include_files or []
        self.upload_path = upload_path
        if isinstance(submodules, str):
            submodules = [submodules]
        self.submodules = submodules
        if submodules is not None:
            if not isinstance(tokenizers, (dict, omg.DictConfig)):
                raise ValueError(
                    f"Must provide a `tokenizers` dict when using `submodules`. Found `{type(tokenizers)}`"
                )
            if set(self.submodules) >= set(tokenizers.keys()):  # type: ignore
                raise ValueError(
                    f"`submodules` list `{self.submodules}` must match `tokenizers` keys {tokenizers.keys()}"
                )
            self.tokenizers = tokenizers or {}
        else:
            self.tokenizers = tokenizers or None

        # triggers
        self._on_fit_end = on_fit_end
        self._on_train_end = on_train_end

    def on_fit_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102, ARG002
        if self._on_fit_end:
            self._export_module(module=module)

    def on_train_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102, ARG002
        if self._on_train_end:
            self._export_module(module=module)

    @rank_zero_only
    def _export_module(self, *, module: torch.nn.Module) -> None:
        """Export the model.

        # TODO: handle sharded models (e.g., FSDP, DeepSpeed)
        """
        if fabric_wrappers.is_wrapped(module):
            module = module.module

        # Export the module
        if self.submodules is None:
            _export_module(
                module,
                self.output_dir,
                self.include_files,
                tokenizer=self.tokenizers,  # Expect a single tokenizer
            )
        elif isinstance(self.submodules, (list, omg.ListConfig)):
            for submodule in self.submodules:
                submodule_ = getattr(module, submodule, None)
                if submodule_ is None:
                    logger.debug(f"Submodule `{submodule}` not found in `{type(module).__name__}`")
                    continue
                export_path = self.output_dir / submodule if len(self.submodules) > 1 else self.output_dir
                _export_module(
                    submodule_,
                    export_path,
                    self.include_files,
                    tokenizer=self.tokenizers[submodule],  # type: ignore # Expect a tokenizer per module
                )
        else:
            raise TypeError(f"Invalid type `{type(self.submodules)}` for `submodules`")

        # Upload the module
        if self.upload_path is not None:
            _upload_dir(self.output_dir, self.upload_path)


def _export_module(
    module: torch.nn.Module | transformers.PreTrainedModel,
    output_dir: pathlib.Path,
    include_files: list[str],
    tokenizer: None | transformers.PreTrainedTokenizerBase | ConfLike = None,
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

    if tokenizer is None:
        return

    # Handle the tokenizer
    if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
        tokenizer = vod_configs.TokenizerConfig(**tokenizer).instantiate()  # type: ignore
    logger.info(f"Exporting tokenizer `{type(tokenizer).__name__}` to `{output_dir.absolute()}` (HF pretrained)")
    tokenizer.save_pretrained(output_dir)


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
