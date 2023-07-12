from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Self, Type, Union

import numpy as np
import torch
from pydantic import BaseModel

from raffle_ds_research.tools.utils.pretty import repr_tensor


class ConstrainedTensor:
    """Define a `torch.Tensor` with constraints on its dtype, device, and shape.

    Shapes can be specified as a tuple of integers or strings. Strings represent
    a dimension variable, which can be used to check that two tensors have the
    same shape along a given dimension. For example, if you want to ensure that
    two tensors have the same number of channels, you can specify the shape as
    `('N', 'C', 'H', 'W')`. The shape will be checked to ensure that the tensors
    have the same number of channels and pixels.

    See `validate_shapes_consistency` for more details.
    """

    allow_casting: bool = False
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    shape: Optional[tuple[Union[str, int], ...]] = None

    @classmethod
    def __get_validators__(
        cls: Type[Self],
    ) -> Iterable[Callable[[Any], torch.Tensor]]:
        """Return a list of validators for the constrained tensor."""
        yield cls._validate_py_type
        yield cls._validate_dtype
        yield cls._validate_device
        yield cls._validate_shape

    @classmethod
    def __modify_schema__(cls: Type[Self], field_schema: Dict[str, Any]) -> None:
        """Modify the schema for the constrained tensor."""
        ...

    @classmethod
    def _validate_py_type(cls: Type[Self], v: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            if cls.allow_casting:
                v = torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
            else:
                raise TypeError(f"Expected `torch.Tensor` or `np.ndarray`, got {v.dtype}")
        return v

    @classmethod
    def _validate_dtype(cls: Type[Self], v: torch.Tensor) -> torch.Tensor:
        if cls.dtype is None:
            return v

        if v.dtype != cls.dtype:
            if cls.allow_casting:
                v = v.to(cls.dtype)
            else:
                raise ValueError(f"Expected dtype {cls.dtype}, got {v.dtype}")
        return v

    @classmethod
    def _validate_device(cls: Type[Self], v: torch.Tensor) -> torch.Tensor:
        if cls.device is None:
            return v

        if v.device != cls.device:
            if cls.allow_casting:
                v = v.to(cls.device)
            else:
                raise ValueError(f"Expected device {cls.device}, got {v.device}")
        return v

    @classmethod
    def _validate_shape(cls: Type[Self], v: torch.Tensor) -> torch.Tensor:
        if cls.shape is None:
            return v

        dim = len(cls.shape)
        if dim != len(v.shape):
            raise ValueError(
                f"Expected {dim} dimensions, got {len(v.shape)} " f"(tensor_shape={v.shape}, target_shape={cls.shape})"
            )
        #
        for i, target_shape in enumerate(cls.shape):
            if isinstance(target_shape, str) or target_shape == -1:
                continue
            if v.shape[i] != target_shape:
                raise ValueError(
                    f"Expected shape {cls.shape}, got {v.shape}. "
                    f"Mismatch at dim {i}. "
                    f"(tensor_shape={v.shape}, target_shape={cls.shape})"
                )

        return v

    def __repr__(self) -> str:
        """Return a string representation of the constrained tensor."""
        attrs = [
            f"{k}={repr_tensor(v)}" if isinstance(v, torch.Tensor) else f"{k}={v}" for k, v in self.__dict__.items()
        ]
        attrs = ", ".join(attrs)
        return f"ConstrainedTensor({attrs})"


def constrained_tensor(
    *,
    allow_casting: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    shape: Optional[Union[torch.Size, Iterable[Union[str, int]]]] = None,
) -> Type[ConstrainedTensor]:
    """Define a `ConstrainedTensor` class with the given constraints."""
    if shape is not None:
        shape = list(shape)
    namespace = {"allow_casting": allow_casting, "dtype": dtype, "device": device, "shape": shape}
    attrs = [
        f"{k}={repr_tensor(v)}" if isinstance(v, torch.Tensor) else f"{k}={v}"
        for k, v in namespace.items()
        if v is not None
    ]
    attrs = ",".join(attrs)
    cls_name = f"ConstrainedTensor[{attrs}]"
    return type(cls_name, (ConstrainedTensor,), namespace)


def validate_shapes_consistency(model: BaseModel, values: dict[str, Any]) -> dict[str, Any]:
    """Validate the shapes consistency of the tensors given a `pydantic.BaseModel`."""
    constrained_shapes = {
        k: v.type_.shape
        for k, v in model.__fields__.items()
        if issubclass(v.type_, ConstrainedTensor) and v.type_.shape is not None
    }

    # init the shape_variable structure
    _init = "<|INITIALIZED|>"
    shapes_variables = {d: _init for shp in constrained_shapes.values() for d in shp if isinstance(d, str)}

    # scan through the actual shape values
    for k, v in values.items():
        if k in constrained_shapes:
            constrained_shape = constrained_shapes[k]
            actual_shape = v.shape
            for i, d in enumerate(constrained_shape):
                if not isinstance(d, str):
                    continue
                if shapes_variables[d] == _init:
                    shapes_variables[d] = int(actual_shape[i])
                elif shapes_variables[d] != int(actual_shape[i]):
                    raise ValueError(
                        f"Inconsistent shape for variable {d}: " f"{shapes_variables[d]} != {actual_shape[i]}"
                    )

    return values
