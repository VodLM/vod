# pylint: disable=no-member

from typing import Any

import pydantic
import rich
import torch

from raffle_ds_research.tools.pydantic_torch import constrained_tensor, validate_shapes_consistency
from raffle_ds_research.tools.utils.pretty import repr_tensor


class SupervisedRetrievalBatch(pydantic.BaseModel):
    """Defines the input batch for the supervised retrieval model."""

    class Config:
        """pydantic config."""

        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    # questions
    question_input_ids: constrained_tensor(
        shape=["bs", "q_length"],
        dtype=torch.long,
    ) = pydantic.Field(
        ...,
        alias="question.input_ids",
    )
    question_attention_mask: constrained_tensor(
        shape=["bs", "q_length"],
        dtype=torch.long,
    ) = pydantic.Field(
        ...,
        alias="question.attention_mask",
    )

    # sections
    section_input_ids: constrained_tensor(
        shape=["bs", "n_secs", "s_length"],
        dtype=torch.long,
    ) = pydantic.Field(
        ...,
        alias="section.input_ids",
    )
    section_attention_mask: constrained_tensor(
        shape=["bs", "n_secs", "s_length"],
        dtype=torch.long,
    ) = pydantic.Field(
        ...,
        alias="section.attention_mask",
    )

    # sections info (label, score)
    section_label: constrained_tensor(
        shape=["bs", "n_secs"],
        dtype=torch.bool,
    ) = pydantic.Field(
        ...,
        alias="section.label",
    )
    section_score: constrained_tensor(
        shape=["bs", "n_secs"],
        dtype=torch.float,
    ) = pydantic.Field(
        ...,
        alias="section.score",
    )

    @pydantic.root_validator
    def _validate_shapes(cls, values: dict[str, Any]) -> dict[str, Any]:
        values = validate_shapes_consistency(cls, values)
        return values

    def __repr__(self) -> str:
        """String representation of the batch."""
        attrs = [
            f"{k}={repr_tensor(v)}" if isinstance(v, torch.Tensor) else f"{k}={v}" for k, v in self.__dict__.items()
        ]
        attrs = ", ".join(attrs)
        return f"{type(self).__name__}({attrs})"

    def __str__(self) -> str:
        """String representation of the batch."""
        return self.__repr__()


if __name__ == "__main__":
    batch = SupervisedRetrievalBatch(
        **{
            "question.input_ids": torch.randn(2, 512).long(),
            "question.attention_mask": torch.randn(2, 512).long(),
            "section.input_ids": torch.randn(2, 32, 512).long(),
            "section.attention_mask": torch.randn(2, 32, 512).long(),
            "section.label": torch.randn(2, 32).bool(),
            "section.score": torch.randn(2, 32).float(),
        }
    )
    rich.print(batch)
