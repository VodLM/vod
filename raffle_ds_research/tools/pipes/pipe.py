from __future__ import annotations

from abc import ABC
from copy import copy
from typing import TypeVar, Generic, Optional, Type
from typing import Union

import torch
from pydantic import BaseModel, PrivateAttr, Extra, Field

I = TypeVar("I", bound=Union[None, BaseModel])
"""Models an input bach."""
O = TypeVar("O", bound=Union[None, BaseModel])
"""Models an output bach."""
E = TypeVar("I", bound=Union[None, BaseModel])
"""Models an input example."""


class IoValidator:
    def __init__(
        self,
        input_model: Optional[Type[I]],
        output_model: Optional[Type[O]],
        example_model: Optional[Type[E]],
        cast: bool = False,
    ):
        self._input_model = input_model
        self._output_model = output_model
        self._example_model = example_model
        self._cast = cast

    @staticmethod
    def _validate_and_maybe_cast(batch, *, model: Optional[Type[BaseModel]], cast: bool, **kwargs):
        if model is not None:
            output = model(**batch)
            if cast:
                batch = output
            else:
                batch = output.dict(**kwargs)
        return batch

    def valid_in(self, batch: dict):
        batch = self._validate_and_maybe_cast(batch, model=self._input_model, cast=self._cast)
        return batch

    def valid_out(self, batch: dict):
        batch = self._validate_and_maybe_cast(batch, model=self._output_model, cast=self._cast, by_alias=True)
        return batch

    def valid_example(self, example: dict):
        example = self._validate_and_maybe_cast(example, model=self._example_model, cast=self._cast)
        return example


class Pipe(Generic[I, O, E], ABC, BaseModel):
    """A pipe object represents a function that can be applied to a batch of examples.
    Pipes also accept lists of examples and collate them into a batch before applying the function.
    Pipes enable validating the inputs and outputs of the function using `pydantic.BaseModel`,
    although this functionality is optional.

    ## Processing batches
    The `__call__` method takes a batch of examples and returns a batch of outputs.
    The main function of the pipe is defined by the `_process_batch` method.

    ## Collating examples
    The collate function is defined by the `_collate_egs` method.
    It is a function that takes a list of examples and returns a batch.
    By default, it uses the `default_collate` function from `torch.utils.data.dataloader`.

    ## Validating inputs and outputs
    The `_input_model` and `_output_model` attributes define the models used to validate
    the inputs and outputs of the function. When the attributes are set to `None`,
    validation is skipped.
    """

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid

    update_batch: bool = False
    fn_kwargs: dict = Field({}, description="kwargs to pass to the function")

    # input to the validation function
    _input_model: Optional[Type[I]] = PrivateAttr(None)
    _output_model: Optional[Type[O]] = PrivateAttr(None)
    _example_model: Optional[Type[E]] = PrivateAttr(None)
    _io_validator: IoValidator = PrivateAttr(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        io_validator = IoValidator(
            input_model=self._input_model,
            output_model=self._output_model,
            example_model=self._example_model,
        )
        object.__setattr__(self, "_io_validator", io_validator)

    def __call__(self, batch: list[dict] | dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        # update the kwargs with the defaults
        kwargs = self.update_kwargs(kwargs)

        # collate the batch
        if isinstance(batch, list):
            batch = self._collate_egs(batch, **kwargs)

        # process the batch
        output = self._validate_and_process_batch(batch, idx=idx, **kwargs)
        return output

    def _collate_egs(self, examples: list[dict], **kwargs) -> dict:
        """Convert a list of examples to a single batch."""
        examples = [self._io_validator.valid_example(example) for example in examples]
        return torch.utils.data.dataloader.default_collate(examples)

    def _process_batch(self, batch: dict, idx: Optional[list[int]] = None, **kwargs) -> dict:
        """Process a batch."""
        raise NotImplementedError

    def _validate_and_process_batch(self, batch: dict, **kwargs) -> dict:
        # process the batch
        batch = self._io_validator.valid_in(batch)
        output_batch = self._process_batch(copy(batch), **kwargs)
        output_batch = self._io_validator.valid_out(output_batch)

        # potentially add the original batch to the output
        if self.update_batch:
            output_batch = {**batch, **output_batch}

        return output_batch

    def update_kwargs(self, kwargs):
        kwargs = {**self.fn_kwargs, **kwargs}
        return kwargs
