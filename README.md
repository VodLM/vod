# VOD - Variational Open-Domain

A scalable experimentation framework for latent-retrieval models.

## Basic installation

```shell
# first, install elasticsearch (see `scripts/setup-vm.sh`, run commands one by one)
poetry env use 3.10 # <- requires python 3.10 to be installed
poetry install
```

> **Note** See the tips and tricks section to install an environment using cuda

## Usage

To train a model, use:

```shell
poetry run train
```

The `train` endpoint uses `hydra` to parse arguments and configure the run.
See `configs/main.yaml` for the default configuration. You can override any of the default values by passing them as arguments to the `train` endpoint. For example, to train a model with a different encoder, use:

```shell
poetry run train model/encoder=t5-base batch_size.per_device=4
```

Configurations can be overriden using `patch` configurations (experiment, hardware, etc.). For instance, to train a retrieval model (base size) using torch DDP:

```shell
poetry run train +patch/task=retrieval +patch/arch=ddp-base
```

## Tips and Tricks

<details>
  <summary>Install in a Mamba env and build faiss-gpu</summary>

```bash
# install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
# setup base env - try to run it, or follow the script step by step
bash setup-mamba-env.sh
# build faiss - try to run it, or follow the script step by step
bash build-faiss.sh
```

</details>

<details>
  <summary>Poetry install troubleshooting guide</summary>

```shell
# in case of `InitError` (on GCP): run the following
# --> see `https://github.com/python-poetry/poetry/issues/1917#issuecomment-1251667047`
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
```

</details>

<details>
  <summary>Faiss segmentation fault on cpu</summary>

```shell
# faiss segmentation fault
# --> install faiss using conda first
# --> see `https://github.com/facebookresearch/faiss/issues/2317`
conda install -c pytorch faiss-cpu
```

</details>

<details>
  <summary>Slow faiss initialization on GPU</summary>

Faiss can take up to 30min to compile CUDA kernels. See [this GitHub issue](https://github.com/facebookresearch/faiss/issues/1177).

</details>
