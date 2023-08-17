# Pipes

Pipes are containerized operations for data processing.
They allow composing complex data processing pipelines in a simple and Functional way.

## `Pipe`

### Main functionality

A pipe is a unit of transformation:

```python
class Pipe(...):
    def __call__(self, batch: dict, **kwargs) -> dict: ...


pipe_a = Pipe()
pipe_b = Pipe()

# batch_in -> pipe_a -> pipe_b -> batch_out
batch_in = {...}
batch_out_ = pipe_a(batch_in)
batch_out = pipe_b(batch_out_)
```

### Validation

Pipes can perform input and output validation:
```python
from pydantic import BaseModel

class InputModel(BaseModel):
    ...

class OutputModel(BaseModel):
    ...

class PipeWithValidation(Pipe):
    _input_model = InputModel
    _output_model = OutputModel


pipe = PipeWithValidation()
batch_in = {...}
batch_out = pipe(batch_in) # <- raises ValidationError if batch_in or batch_out are not valid
```


### `datasets.Dataset.map`

Pipes can be used with `datasets.Dataset.map`, which allows for parallel processing of large datasets and caching transformations.

```python
pipe = MyPipe()

dataset = load_dataset(...)
processed_dataset = dataset.map(
    pipe,
    batched=True, # <-- required
    with_indices=True, # <-- required
    num_proc=8, # <-- optional
)
```

### `colalte_fn`

...



