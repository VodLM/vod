# raffle-ds-research

User-friendly and scalable experimentation framework for modern NLP


## Install

```shell
poetry env use 3.10 # <- requires python 3.10 to be installed
poetry install

# in case of `InitError` (on GCP): run the following
# --> see `https://github.com/python-poetry/poetry/issues/1917#issuecomment-1251667047`
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install

# faiss segmentation fault
# --> install faiss using conda first
# --> see `https://github.com/facebookresearch/faiss/issues/2317`
conda install -c pytorch faiss-cpu

# Install `elasticsearch`
# https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html
# start elasticsearch
sudo systemctl enable elasticsearch.service --now
```

## Usage

### Train models

To train a model, use:
```shell
poetry run train
```

The `train` endpoint uses `hydra` to parse arguments and configure the run.
See `configs/main.yaml` for the default configuration. You can override any of the default values by passing them as arguments to the `train` endpoint. For example, to train a model with a different encoder, use:
```shell
poetry run train model/encoder=t5-base batch_size.per_device=4
```

Recipes define a pre-configured set of arguments.
Recipes are defined in `configs/recipe`.
To use a recipe, for example `t5-base`, use:
```shell
poetry run train +recipe=t5-base
```

### Serve a model for testing

```shell
# api stuff, coming soon
```