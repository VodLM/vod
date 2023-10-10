import json
import pathlib
import warnings

import lightning as L
import torch
import transformers
from lightning.fabric import wrappers as fabric_wrappers
from loguru import logger

from .base import Callback


def _export_module(
    module: torch.nn.Module | transformers.PreTrainedModel,
    output_dir: pathlib.Path,
    templates: None | dict[str, str],
) -> None:
    """Export a module."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if templates is not None:
        (output_dir / "templates.json").write_text(json.dumps(templates, indent=2))

    # Handle `transformers.PreTrainedModel`
    if isinstance(module, transformers.PreTrainedModel):
        logger.info(f"Exporting module `{type(module).__name__}` to `{output_dir.absolute()}` (HF pretrained)")
        module.save_pretrained(output_dir)
    else:
        logger.info(f"Exporting module `{type(module).__name__}` to `{output_dir.absolute()}` (torch state dict)")
        torch.save(module.state_dict(), output_dir / "model.pt")


class ExportModule(Callback):
    """Pretty print the first training batch."""

    def __init__(
        self,
        *,
        output_dir: str | pathlib.Path = "exported-module",
        only_submodule: None | str = None,
        submodules: None | list[str],
        templates: None | dict[str, str],
        on_fit_end: bool = True,  # at the end the whole training routine
        on_train_end: bool = False,  # at the end of each training period
    ) -> None:
        super().__init__()
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.only_submodule = only_submodule
        if only_submodule is not None and submodules is not None:
            submodules = []
            warnings.warn("Both `only_submodule` and `submodules` are set. Ignoring `submodules`.")  # noqa: B028
        self.submodules = submodules
        if templates is not None:
            templates = {k: str(v) for k, v in templates.items()}
        self.templates = templates

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

        if self.only_submodule is not None:
            submodule_ = getattr(module, self.only_submodule, None)
            if submodule_ is None:
                raise ValueError(f"Submodule `{self.only_submodule}` not found in `{type(module).__name__}`")
            _export_module(submodule_, self.output_dir, self.templates)
        elif self.submodules is None:
            _export_module(module, self.output_dir, self.templates)
        else:
            for submodule in self.submodules:
                submodule_ = getattr(module, submodule, None)
                if submodule_ is None:
                    logger.debug(f"Submodule `{submodule}` not found in `{type(module).__name__}`")
                    continue
                _export_module(submodule_, self.output_dir / submodule, self.templates)
