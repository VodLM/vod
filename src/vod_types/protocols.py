import typing as typ
from typing import Any

import torch


class SupportsGetFingerprint(typ.Protocol):
    """A protocol for objects that support getting a fingerprint."""

    def get_fingerprint(self) -> str:
        """Get a fingerprint for the object."""
        ...


class EncoderLike(typ.Protocol):
    """A protocol for objects that acts like a `transformers.PreTrainedModel` encoder."""

    def __call__(
        self,
        input_ids: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Encode the input."""
        ...
