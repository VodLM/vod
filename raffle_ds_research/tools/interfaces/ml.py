import enum
import typing

import pydantic
import torch


class FieldType(enum.Enum):
    """Enum for token fields."""

    QUESTION: str = "question"  # type: ignore
    SECTION: str = "section"  # type: ignore


class TokenizedField(pydantic.BaseModel):
    """Models a tokenized text field."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        use_enum_values = True

    field: typing.Optional[FieldType] = None
    """The type of the field (e.g., question, section)"""
    input_ids: torch.Tensor
    """The batch of token ids."""
    attention_mask: torch.Tensor
    """The batch of attention masks."""
    token_type_ids: typing.Optional[torch.Tensor] = None
    """The batch of token type ids."""


@typing.runtime_checkable
class ProtocolEncoder(typing.Protocol):
    """Protocol for ML models encoding tokenized text into vectors."""

    def __call__(self, data: TokenizedField) -> torch.Tensor:
        """Embed/encode a tokenized field into a vector."""
        ...

    def get_output_shape(self, field: typing.Optional[FieldType] = None) -> tuple[int, ...]:
        """Get the output shape of the encoder. Set `-1` for unknown dimensions."""
        ...
