import abc
import functools
import typing as typ

import omegaconf as omg
import torch
import vod_types as vt
from datasets.fingerprint import Hasher, hashregister
from transformers import pytorch_utils, trainer_pt_utils
from vod_models.support import maybe_instantiate
from vod_tools import fingerprint

VodSystemMode = typ.Literal["encode", "evaluate", "generate"]


class VodSystem(torch.nn.Module):
    """A Machine learning system for the VOD framework."""

    def __init__(
        self,
        optimizer: None | dict | omg.DictConfig | functools.partial = None,
        scheduler: None | dict | omg.DictConfig | functools.partial = None,
    ):
        super().__init__()
        if isinstance(optimizer, (dict, omg.DictConfig)):
            optimizer = maybe_instantiate(optimizer)  # type: ignore
            if not isinstance(optimizer, functools.partial):
                raise TypeError(f"Expected a partial function, got {type(optimizer)}")
        if isinstance(scheduler, (dict, omg.DictConfig)):
            scheduler = maybe_instantiate(scheduler)  # type: ignore
            if not isinstance(scheduler, functools.partial):
                raise TypeError(f"Expected a partial function, got {type(scheduler)}")

        self.optimizer_cls: functools.partial = optimizer  # type: ignore
        self.scheduler_cls: functools.partial = scheduler  # type: ignore

    def forward(
        self, batch: typ.Mapping[str, torch.Tensor], *, mode: VodSystemMode = "encode", **kws: typ.Any
    ) -> typ.Mapping[str, torch.Tensor]:
        """Handles multiple modes in the forward pass.

        NOTE: this is required so `torch.compile()` and other optimizations can work.
        """
        if mode == "encode":
            return self.encode(batch, **kws)
        if mode == "evaluate":
            return self.evaluate(batch, **kws)
        if mode == "generate":
            return self.generate(batch, **kws)

        raise ValueError(f"Unknown mode {mode}.")

    @abc.abstractmethod
    def evaluate(
        self,
        batch: typ.Mapping[str, torch.Tensor],
        **kws: typ.Any,
    ) -> vt.ModelOutput:  # noqa: ARG002
        """Run a forward pass and compute the gradients."""
        ...

    @abc.abstractmethod
    def generate(
        self,
        batch: typ.Mapping[str, torch.Tensor],
        **kws: typ.Any,
    ) -> typ.Mapping[str, torch.Tensor]:
        """Generate completions given the inputs."""
        ...

    @abc.abstractmethod
    def encode(
        self,
        batch: typ.Mapping[str, torch.Tensor],
        **kws: typ.Any,
    ) -> typ.Mapping[str, torch.Tensor]:
        """Computes the embeddings for the query and the document."""

    @abc.abstractmethod
    def get_encoding_shape(self) -> None | tuple[int, ...]:
        """Dimension of the model output."""
        ...

    def get_fingerprint(self) -> str:
        """Return a fingerprint of the model."""
        return fingerprint.fingerprint_torch_module(self)

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Configure the optimizer and the learning rate scheduler."""
        if isinstance(self.optimizer_cls, functools.partial):
            weight_decay = self.optimizer_cls.keywords.get("weight_decay", None)
        else:
            weight_decay = None

        # If no weight_decay is provided, instantiate the optimizer directly
        if weight_decay is None:
            return self.optimizer_cls(self.parameters())

        # Instantiate the optimizer à la HuggingFace
        # https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/trainer.py#L1075C15-L1075C15
        decay_parameters = trainer_pt_utils.get_parameter_names(self, pytorch_utils.ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]
        return self.optimizer_cls(optimizer_grouped_parameters)

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> None | torch.optim.lr_scheduler._LRScheduler:
        """Init the learning rate scheduler."""
        if self.scheduler_cls is None:
            return None

        return self.scheduler_cls(optimizer)


@hashregister(VodSystem)
def _hash_(hasher: Hasher, value: VodSystem) -> str:  # noqa: ARG001
    return fingerprint.fingerprint_torch_module(value)
