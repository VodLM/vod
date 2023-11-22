<div align="center">

<img alt="VOD" src="assets/vod-banner.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

<p align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.11-blue?style=for-the-badge&logo=python&logoColor=white"></a>
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

- October 16' 2023: We released the version 0.2.0 of `vod` - simpler & better code ✨
- September 21' 2023: We integrated the [BeIR](https://github.com/beir-cellar/beir) datasets 🍻
- July 19' 2023: We integrated [Qdrant](https://qdrant.tech/) as a search backend 📦
- June 15' 2023: We adopted Ligthning's [Fabric](https://lightning.ai/docs/fabric/stable/) ⚡️
- April 24' 2023: We will be presenting [Variational Open-Domain Question Answering](https://arxiv.org/abs/2210.06345) at [ICML2023 @ Hawaii](https://icml.cc/Conferences/2023/Dates) 🌊

## What is VOD? 🎯

VOD aims at building, training and evaluating next-generation retrieval-augmented language models (REALMs). The project started with our research paper [Variational Open-Domain Question Answering](https://arxiv.org/abs/2210.06345), in which we introduce the VOD objective: a new variational objective for end-to-end training of REALMs.

The original paper only explored the application of the VOD objective to multiple-choice ODQA, this repo aims at exploring generative tasks (generative QA, language modelling and chat). We are building tools to make training of large generative search models possible and developper-friendly. The main modules are:

- `vod_dataloaders`: efficient `torch.utils.DataLoader` with dynamic retrieval
from multiple search engines
- `vod_search`: a common interface to handle `sparse` and `dense` search engines ([elasticsearch](https://www.elastic.co/), [faiss](https://www.google.com/search?q=faiss&sourceid=chrome&ie=UTF-8), [Qdrant](https://qdrant.tech/))
- `vod_models`: a collection of REALMs using large retrievers (T5s, e5, etc.) and OS generative models (RWKV, LLAMA 2, etc.). We also include `vod_gradients`, a module to compute gradients of RAGs end-to-end.

## Roadmap Summer 2023 ☀️

Progress tracked in <https://github.com/VodLM/vod/issues/1>

The repo is currently in **research preview**. This means we already have a few components in place, but we still have work to do before a wider adoption of VOD, and before training next-gen REALMS. Our objectives for this summer are:

- [x] Search API: add Filtering Capabilities
- [x] Datasets: support more datasets for common IR & Gen AI
- [x] UX: plug-and-play, extendable
- [x] Modelling: implement REALM for Generative Tasks
- [ ] Gradients: VOD for Generative Tasks

## Join us 🤝

If you also see great potential in combining LLMs with search components, join the team! We welcome developers interested in building scalable and easy-to-use tools as well as NLP researchers.

## Project Structure 🏗️

| Module          | Usage                                                                           | Status |
|-----------------|---------------------------------------------------------------------------------|--------|
| vod_configs     | Sturctured `pydantic` configurations                                            | ✅      |
| vod_dataloaders | Dataloaders for retrieval-augmented tasks                                       | ✅      |
| vod_datasets    | Universal dataset interface (`rosetta`) and custom dataloaders (e.g., BeIR)     | ✅      |
| vod_exps        | Research experiments, configurable with `hydra`                                 | ✅      |
| vod_models      | A collection of REALMs + gradients (retrieval, VOD, etc.)                       | ⚠️     |
| vod_ops         | ML operations using `lightning.Fabric` (training, benchmarking, indexing, etc.) | ✅      |
| vod_search      | Hybrid and sharded search clients (`elasticsearch`, `faiss` and `qdrant`)       | ✅      |
| vod_tools       | A collection of utilities  (pretty printing, argument parser, etc.)             | ✅      |
| vod_types       | A collection data structures and python `typing` modules                        | ✅      |

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
# How load datasets wtih the universal `rosetta` intergace
poetry run python -m examples.datasets.rosetta

# How to start and use a `faiss` search engine
poetry run python -m examples.search.faiss

# How to start and use a `qdrant` search engine
poetry run python -m examples.search.qdrant

# How to compute embeddings for a large dataset using `lighning.Fabric`
poetry run python -m examples.features.predict

# How to build a Realm dataloader backed by a Hybrid search engine
poetry run python -m examples.features.dataloader

```

## Using the trainer CLI 🚀

VOD allows training large retrieval models while dynamically retrieving sections from a large knowledge base. The CLI is accessible with:

```shell
poetry run train

# Debugging -- Run the training script with a small model and small dataset
poetry run train model/encoder=debug datasets=scifact
```

<details>
<summary>🔧 Arguments & config files</summary>

The `train` endpoint uses `hydra` to parse arguments and configure the run.
See `vod_exps/hydra/main.yaml` for the default configuration. You can override any of the default values by passing them as arguments to the `train` endpoint. For example, to train a T5-base encoder on MSMarco using FSDP:

```shell
poetry run train model/encoder=t5-base batch_size.per_device=4 datasets=msmarco fabric/strategy=fsdp
```

</details>

## Contribution (development environment)

```shell
# Install pinned pip first
pip install -r $(git rev-parse --show-toplevel)/pip-requirements.txt

# Install shared development dependencies and project/library-specific dependencies
pip install -r $(git rev-parse --show-toplevel)/dev-requirements.txt -r requirements.txt

# Typecheck the code
pyright .
```

## Technical details

<details>
<summary>🐙 Multiple datasets & Sharded search</summary>

VOD is built for multi-dataset training. Youn can multiple training/validation/test datasets, each pointing to a different corpus. Each corpus can be augmented with a specific search backend. For instance this config allows using `Qdrant` as a backend for the `squad` sections and `QuALITY` contexts while using `faiss` to index Wikipedia.

VOD implement a hybrid sharded search engine. This means that for each indexed corpus, VOD fits multiple search engines (e.g., Elasticsearch + Qdrant). At query time, data points are dispatched to each shard (corpus) based on the `dataset.link` attribute.

<div align="center">
<img alt="Sharded search" src="assets/sharded-search.png" width="800px" style="max-width: 100%;">
</div>

</details>

## Tips and Tricks 🦊

<details>
<summary>🐍 Setup a Mamba environment and build faiss-gpu from source</summary>

```bash
# install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
# setup base env - try to run it, or follow the script step by step
bash setup-scripts/setup-mamba-env.sh
# build faiss - try to run it, or follow the script step by step
bash setup-scripts/build-faiss-gpu.sh
# **Optional**: Install faiss-gpu in your poetry env:
export PYPATH=`poetry run which python`
(cd libs/faiss/build/faiss/python && $PYPATH setup.py install)
```

</details>

## Citation 📚

### Variational Open-Domain Question Answering

This repo is a clean re-write of the original code [FindZebra/fz-openqa](https://github.com/FindZebra/fz-openqa) aiming at handling larger datasets, larger models and generative tasks.

```
@InProceedings{pmlr-v202-lievin23a,
  title =   {Variational Open-Domain Question Answering},
  author =       {Li\'{e}vin, Valentin and Motzfeldt, Andreas Geert and Jensen, Ida Riis and Winther, Ole},
  booktitle =   {Proceedings of the 40th International Conference on Machine Learning},
  pages =   {20950--20977},
  year =   {2023},
  editor =   {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume =   {202},
  series =   {Proceedings of Machine Learning Research},
  month =   {23--29 Jul},
  publisher =    {PMLR},
  pdf =   {https://proceedings.mlr.press/v202/lievin23a/lievin23a.pdf},
  url =   {https://proceedings.mlr.press/v202/lievin23a.html},
  abstract =   {Retrieval-augmented models have proven to be effective in natural language processing tasks, yet there remains a lack of research on their optimization using variational inference. We introduce the Variational Open-Domain (VOD) framework for end-to-end training and evaluation of retrieval-augmented models, focusing on open-domain question answering and language modelling. The VOD objective, a self-normalized estimate of the Rényi variational bound, approximates the task marginal likelihood and is evaluated under samples drawn from an auxiliary sampling distribution (cached retriever and/or approximate posterior). It remains tractable, even for retriever distributions defined on large corpora. We demonstrate VOD’s versatility by training reader-retriever BERT-sized models on multiple-choice medical exam questions. On the MedMCQA dataset, we outperform the domain-tuned Med-PaLM by +5.3% despite using 2.500$\times$ fewer parameters. Our retrieval-augmented BioLinkBERT model scored 62.9% on the MedMCQA and 55.0% on the MedQA-USMLE. Last, we show the effectiveness of our learned retriever component in the context of medical semantic search.}
}
```

## Partners 🏫

The project is currently supported by the [Technical University of Denmark (DTU)](https://www.dtu.dk/) and [Raffle.ai](https://raffle.ai).

<div align="center">

<img alt="DTU logo" src="assets/dtu-logo.png" width="40px" style="max-width: 100%;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img alt="Raffle.ai logo" src="assets/raffle-logo.png" width="120px" style="max-width: 100%;">


</div>
