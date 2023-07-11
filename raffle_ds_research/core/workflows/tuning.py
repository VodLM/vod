from __future__ import annotations

import dataclasses
import pathlib
import typing
from typing import Any

import lightning as L
import torch
import transformers
from loguru import logger
from torch import nn

from raffle_ds_research.core import config as core_config
from raffle_ds_research.core import mechanics
from raffle_ds_research.core.mechanics import search_engine
from raffle_ds_research.core.ml import gradients
from raffle_ds_research.core.workflows.precompute import PrecomputedDsetVectors
from raffle_ds_research.core.workflows.utils import support
from raffle_ds_research.tools import dstruct
from raffle_ds_research.tools.utils.progress import IterProgressBar

_DEFAULT_TUNE_LIST = ["bm25"]


def _min_score_no_nan(score: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the minimum score along a dimension, ignoring NaNs."""
    min_scores_along_dim, _ = torch.min(
        torch.where(torch.isnan(score), torch.tensor(float("inf")), score),
        dim=dim,
        keepdim=True,
    )
    return min_scores_along_dim


class HybridRanker(nn.Module):
    """A ranker that combines multiple scores."""

    def __init__(
        self,
        parameters: dict[str, Any],
        require_grads: list[str],
    ) -> None:
        super().__init__()
        self.grad_fn = gradients.KlDivGradients()
        self.params = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(v), requires_grad=k in require_grads) for k, v in parameters.items()}
        )

    def pydict(self) -> dict[str, Any]:
        """Return a python dictionary of the parameters."""
        return {k: v.item() for k, v in self.params.items()}

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute the hybrid score."""
        hybrid_score = None
        for key, weight in self.params.items():
            batch_key = f"section.{key}"
            if batch_key not in batch:
                continue

            # Fetch the score
            score = batch[batch_key]
            min_scores_along_dim = _min_score_no_nan(score, dim=1)
            score = torch.where(torch.isnan(score), min_scores_along_dim, score)

            # Add the score to the hybrid score
            hybrid_score = weight * score if hybrid_score is None else hybrid_score + weight * score

        if hybrid_score is None:
            raise ValueError("No hybrid score was computed")

        # Compute the gradients
        hybrid_logprobs = torch.log_softmax(hybrid_score, dim=-1)
        return self.grad_fn(batch, retriever_logprobs=hybrid_logprobs)


K = typing.TypeVar("K")


def tune_parameters(
    parameters: dict[str, float],
    tune: None | list[str] = None,
    *,
    fabric: L.Fabric,
    factories: dict[K, mechanics.DatasetFactory],
    vectors: dict[K, PrecomputedDsetVectors],
    search_config: core_config.SearchConfig,
    collate_config: core_config.RetrievalCollateConfig,
    dataloader_config: core_config.DataLoaderConfig,
    cache_dir: pathlib.Path,
    serve_on_gpu: bool = True,
    n_tuning_steps: int = 1_000,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
) -> dict[str, float]:
    """Run benchmarks on a retrieval task."""
    tune = tune or _DEFAULT_TUNE_LIST
    if not all(t in parameters for t in tune):
        raise ValueError(f"tune list `{tune}` contains unknown parameters")

    if fabric.is_global_zero:
        # Define the model
        model = HybridRanker(parameters=parameters, require_grads=tune)

        # make task
        task = _make_tuning_task(
            factories=factories,
            vectors=vectors,
        )

        with search_engine.build_search_engine(
            sections=task.sections.data,
            vectors=task.sections.vectors,
            config=search_config,
            cache_dir=cache_dir,
            faiss_enabled=True,
            bm25_enabled=True,
            serve_on_gpu=serve_on_gpu,
        ) as master:
            search_client = master.get_client()

            # instantiate the dataloader
            dataloader = support.instantiate_retrieval_dataloader(
                questions=task.questions,
                sections=task.sections,
                tokenizer=tokenizer,
                search_client=search_client,
                collate_config=collate_config,
                dataloader_config=dataloader_config,
                parameters=parameters,
            )

            try:
                # Optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

                # run the evaluation
                output = {}
                should_stop = False
                step = 0
                with IterProgressBar() as pbar:
                    ptask = pbar.add_task(
                        "Tuning parameters", total=n_tuning_steps, info=_info_bar(output=output, model=model)
                    )
                    while not should_stop:
                        for batch in dataloader:
                            # compute the gradients
                            output = model(batch)
                            loss = output["loss"]
                            loss.backward()

                            # update the parameters
                            optimizer.step()
                            optimizer.zero_grad()

                            # do optimization here
                            step += 1
                            pbar.update(ptask, advance=1, info=_info_bar(output=output, model=model))
                            if step >= n_tuning_steps:
                                should_stop = True
                                break
            except KeyboardInterrupt:
                logger.warning("Parameter tuning interrupted (KeyboardInterrupt).")

        # update the parameters
        parameters = model.pydict()

    # broadcast the metrics to all the workers
    return fabric.broadcast(parameters)


def _info_bar(output: dict[str, Any], model: HybridRanker) -> str:
    """Return the info bar."""
    base = ""
    if "loss" in output:
        base += f" loss={output['loss'].item():.3f}"

    for param, value in model.pydict().items():
        base += f" {param}={value:.3f}"

    return base


@dataclasses.dataclass(frozen=True)
class TuningTask:
    """Holds the train and validation datasets."""

    questions: support.DsetWithVectors
    sections: support.DsetWithVectors


def _make_tuning_task(
    factories: dict[K, mechanics.DatasetFactory],
    vectors: dict[K, PrecomputedDsetVectors],
) -> TuningTask:
    """Create the `RetrievalTask` from the training and validation factories."""

    def _vec(key: K, field: typing.Literal["question", "section"]) -> dstruct.TensorStoreFactory:
        """Safely fetch the relevant `vector` from the `PrecomputedDsetVectors` structure."""
        x = vectors[key]
        if field == "question":
            return x.questions
        if field == "section":
            return x.sections
        raise ValueError(f"Unknown field: {field}")

    return TuningTask(
        questions=support.concatenate_datasets(
            [
                support.DsetWithVectors.cast(data=factory.get_qa_split(), vectors=_vec(key, "question"))
                for key, factory in factories.items()
            ]
        ),
        sections=support.concatenate_datasets(
            [
                support.DsetWithVectors.cast(data=factory.get_sections(), vectors=_vec(key, "section"))
                for key, factory in factories.items()
            ]
        ),
    )
