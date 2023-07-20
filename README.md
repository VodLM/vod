<div align="center">

<img alt="Lightning" src="assets/vod-banner.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

<p align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

 </div>

<br/>

<div align="center">
Research paper: <a href="https://arxiv.org/abs/2210.06345">Variational Open-Domain Question Answering</a>, ICML 2023
</div>

<br/>

## Latest News 🔥

- July 19' 2023: We integrated [Qdrant](https://qdrant.tech/) as a search backend 📦
- June 15' 2023: We adopted Ligthning's [Fabric](https://lightning.ai/docs/fabric/stable/) ⚡️
- April 24' 2023: We will be presenting [Variational Open-Domain Question Answering](https://arxiv.org/abs/2210.06345) at [ICML2023 @ Hawaii](https://icml.cc/Conferences/2023/Dates) 🌊

## What is VOD? 🎯

VOD aims at building, training and evaluating next-generation retrieval-augmented language models (REALMs). The project started with our research paper [Variational Open-Domain Question Answering](https://arxiv.org/abs/2210.06345), in which we introduce the VOD objective: a new variational objective for end-to-end training of REALMs.

The original paper only explored the application of the VOD objective to multiple-choice ODQA, this repo aims at exploring generative tasks (generative QA, language modelling and chat). We are building tools to make training of large generative search models possible and developper-friendly. The main modules are:

- `vod_gradients`: computing the gradients for REALM and retrieval models
- `vod_dataloaders`: efficient `torch.utils.DataLoader` with dynamic retrieval
from multiple search engines
- `vod_search`: a common interface to handle `sparse` and `dense` search engines ([elasticsearch](https://www.elastic.co/), [faiss](https://www.google.com/search?q=faiss&sourceid=chrome&ie=UTF-8), [Qdrant](https://qdrant.tech/))
- `vod_models`: a collection of REALMs using large retrievers (T5s) and OS generative models (RWKV, LLAMA 2, etc.).

## Roadmap Summer 2023 ☀️
Progress tracked in https://github.com/VodLM/vod/issues/1

The repo is currently in **research preview**. This means we already have a few components in place, but we still have work to do before a wider adoption of VOD, and before training next-gen REALMS. Our objectives for this summer are:

- [x] Search API: add Filtering Capabilities
- [ ] Datasets: support more datasets for common IR & Gen AI
- [ ] Modelling: implement REALM for Generative Tasks
- [ ] Gradients: VOD for Generative Tasks
- [ ] UX: plug-and-play, extendable

## Join us 🤝

If you also see great potential in combining LLMs with search components, join the team! We welcome developers interested in building scalable and easy-to-use tools as well as NLP researchers.

## Project Structure 🏗️

| Module          | Usage                                                 | Status |
|-----------------|-------------------------------------------------------|--------|
| vod_cli         | CLI to train REALMs                                   | ⚠️      |
| vod_configs     | Hydra and Pydantic configs                            | ✅      |
| vod_dataloaders | Dataloaders for retrieval-augmented tasks             | ✅      |
| vod_datasets    | Dataset loaders (MSMarco, etc.)                       | ⚠️      |
| vod_gradients   | Computing gradients for end-to-end REALM training     |    ❌    |
| vod_models      | A collection of REALMs                                |   ❌     |
| vod_search      | Hybrid search using elasticsearch, faiss and Qdrant   |     ✅   |
| vod_tools       | A collection of easy-to-use tools                     |    ✅    |
| vod_workflows   | Main recipes (training, benchmarking, indexing, etc.) |   ⚠️     |

> **Note** The code for VOD gradient and sampling methods currently lives at [VodLM/vod-gradients](https://github.com/VodLM/vod-gradients). The project is still under development and will be integrated into this repo in the next month.

## Installation 📦

We only support development mode for now. You need to install and run `elasticsearch` on your system. Then install poetry and the project using:

```shell
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

> **Note** See the tips and tricks section to build faiss latest with CUDA support

## Examples

```shell
# How load MSMarco
poetry run python -m examples.load_msmarco

# How to start and use a `faiss` search engine
poetry run python -m examples.faiss_search

# How to start and use a `qdrant` search engine
poetry run python -m examples.qdrant_search

# How to compute embeddings for a large dataset using `lighning.Fabric`
poetry run python -m examples.predict

# How to build dataloaders with a Hybrid search engine
poetry run python -m examples.dataloader

```

## Using the trainer CLI 🚀

VOD allows training large retrieval models while dynamically retrieving sections from a large knowledge base. The CLI is accessible with:

```shell
poetry run train
```

<details>
<summary>Arguments & config files</summary>

The `train` endpoint uses `hydra` to parse arguments and configure the run.
See `configs/main.yaml` for the default configuration. You can override any of the default values by passing them as arguments to the `train` endpoint. For example, to train a model with a different encoder, use:

```shell
poetry run train model/encoder=t5-base batch_size.per_device=4
```

Configurations can be overriden using `patch` configurations (experiment, hardware, etc.). For instance, to train a retrieval model (base size) using torch DDP:

```shell
poetry run train +patch/task=retrieval +patch/arch=ddp-base
```

</details>

## Tips and Tricks 🦊

<details>
  <summary>Setup a Mamba environment and build faiss-gpu</summary>

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
  <summary>Handle faiss segmentation fault on cpu</summary>

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

## Citation 📚

### Variational Open-Domain Question Answering

This repo is a clean re-write of the original code [FindZebra/fz-openqa](https://github.com/FindZebra/fz-openqa) aiming at handling larger datasets, larger models and generative tasks.

```
@InProceedings{pmlr-v202-lievin23a,
  title = 	 {Variational Open-Domain Question Answering},
  author =       {Li\'{e}vin, Valentin and Motzfeldt, Andreas Geert and Jensen, Ida Riis and Winther, Ole},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {20950--20977},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/lievin23a/lievin23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/lievin23a.html},
  abstract = 	 {Retrieval-augmented models have proven to be effective in natural language processing tasks, yet there remains a lack of research on their optimization using variational inference. We introduce the Variational Open-Domain (VOD) framework for end-to-end training and evaluation of retrieval-augmented models, focusing on open-domain question answering and language modelling. The VOD objective, a self-normalized estimate of the Rényi variational bound, approximates the task marginal likelihood and is evaluated under samples drawn from an auxiliary sampling distribution (cached retriever and/or approximate posterior). It remains tractable, even for retriever distributions defined on large corpora. We demonstrate VOD’s versatility by training reader-retriever BERT-sized models on multiple-choice medical exam questions. On the MedMCQA dataset, we outperform the domain-tuned Med-PaLM by +5.3% despite using 2.500$\times$ fewer parameters. Our retrieval-augmented BioLinkBERT model scored 62.9% on the MedMCQA and 55.0% on the MedQA-USMLE. Last, we show the effectiveness of our learned retriever component in the context of medical semantic search.}
}
```

## Sponsors 🏫

The project is currently supported by the [Technical University of Denmark (DTU)](https://www.dtu.dk/) and [Raffle.ai](https://raffle.ai).
