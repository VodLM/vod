import abc

import torch
import torch.nn
import vod_types as vt


class Gradients(abc.ABC):
    """Base class for the gradients layer. The gradients layer is a pure function (no torch params)."""

    @abc.abstractmethod
    def __call__(
        self,
        *,
        batch: vt.RealmBatch,
        query_encoding: torch.Tensor,  # the encoding of the queries
        section_encoding: torch.Tensor,  # the encoding of the documents/sections
    ) -> vt.RealmOutput:
        """Compute the gradients/loss."""
        raise NotImplementedError
