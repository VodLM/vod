# raffle-ds-research

User-friendly and scalable experimentation framework for modern NLP

## Install

```shell
poetry env use 3.9 # <- requires python 3.9 to be installed
poetry install

# in case of `InitError` (on GCP): run the following
# --> see `https://github.com/python-poetry/poetry/issues/1917#issuecomment-1251667047`
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install

# faiss segmentation fault
# --> install faiss using conda first
# --> see `https://github.com/facebookresearch/faiss/issues/2317`
conda install -c pytorch faiss-cpu

# Install and start `elasticsearch`
# https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
sudo apt-get update
sudo apt-get install apt-transport-https
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
sudo apt-get update && sudo apt-get install elasticsearch
sudo /bin/systemctl daemon-reload
sudo /bin/systemctl enable elasticsearch.service
sudo systemctl start elasticsearch.service


# Install faiss-gpu on A100 - see `https://github.com/kyamagu/faiss-wheels/issues/54`
# dl wheels from  `https://github.com/kyamagu/faiss-wheels/releases/tag/v1.7.3`
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
poetry run train +recipe=frank-t5-base
```

### Serve a model for testing

```shell
# api stuff, coming soon
```

## Tips and Tricks

### Build faiss-gpu

```bash
# install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
# setup base env - try to run it, or follow the script step by step
bash setup-mamba.sh
# build faiss - try to run it, or follow the script step by step
bash build-faiss.sh
```

### Slow faiss initialization on GPU

Faiss can take up to 30min to compile CUDA kernels. See [this GitHub issue](https://github.com/facebookresearch/faiss/issues/1177).
