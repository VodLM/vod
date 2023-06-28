# pylint: disable=no-member
from __future__ import annotations

import abc
import math
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn
from pydantic.fields import Field
from pydantic.main import BaseModel
from torch.distributions import Categorical


class Gradients(torch.nn.Module):
    """Base class for the gradients layer.s."""

    @abc.abstractmethod
    def forward(self, intermediate_results: dict) -> dict:
        """Compute the gradients/loss."""
        raise NotImplementedError

    def forward_backward(
        self,
        batch: dict[str, torch.Tensor],
        fwd_fn: Callable[[dict], dict],
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
        loss_scaler: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass with a backward pass."""
        fwd_output = fwd_fn(batch, **kwargs)
        grad_output = self({**batch, **fwd_output})
        loss = grad_output["loss"]
        if loss_scaler is not None:
            loss *= loss_scaler
        if backward_fn is None:
            loss.backward()
        else:
            backward_fn(loss)

        return grad_output


class GradientInputs(BaseModel):
    """collection of inputs for the supervised gradients model."""

    class Config:
        """pydantic config."""

        arbitrary_types_allowed = True

    hq: torch.Tensor
    hd: torch.Tensor
    targets: torch.Tensor = Field(
        ...,
        description="Retrieval labels.",
        alias="section.label",
    )
    scores: torch.Tensor = Field(
        ...,
        description="Retrieval scores.",
        alias="section.score",
    )
    bm25: Optional[torch.Tensor] = Field(
        None,
        description="bm25 Retrieval scores.",
        alias="section.bm25",
    )
    faiss: Optional[torch.Tensor] = Field(
        None,
        description="faiss Retrieval scores.",
        alias="section.faiss",
    )


class SelfSupervisedGradients(Gradients):
    """Compute the gradients for the `self-supervised` method."""

    def forward(self, inputs: dict) -> dict:
        """Parse the inputs and compute the loss."""
        data = GradientInputs(**inputs)

        # compute the scores for each pair of (question, section)
        # Note: we can add negative samples across batch here.
        scores = _compute_retriever_logprobs(data)

        # Compute the targets, they are defined as labelled sections
        #  which are assigned with the max. model score.
        masked_scores = scores.clone()
        masked_scores[~data.targets] = -torch.inf
        targets = masked_scores.argmax(dim=-1).detach()

        # Compute the loss
        # Note: we can also use the signal from all
        #       the positive documents.
        pz = Categorical(logits=masked_scores)
        log_pz = pz.log_prob(targets)
        loss = -log_pz.mean()

        return {"loss": loss, "_targets": data.targets, "_logits": scores}


class KlDivGradients(Gradients):
    """Compute the KL divergence between the model and the data."""

    def __init__(
        self,
        eps: Optional[float] = None,
        bm25_guidance_weight: float = 0.0,
        self_supervision_weight: float = 1.0,
    ):
        super().__init__()
        if eps:
            self.log_eps = math.log(eps)
        else:
            self.log_eps = -math.inf

        self.bm25_guidance_weight = bm25_guidance_weight
        self.self_supervision_weight = self_supervision_weight

    def forward(
        self,
        inputs: dict,
        _model_logprobs: Optional[torch.Tensor] = None,
        _n_positive: Optional[torch.Tensor] = None,
    ) -> dict:
        """Parse the inputs and compute the loss."""
        data = GradientInputs(**inputs)

        # 1. Compute the KL divergence between the model and the data
        # Determine the masked sections
        is_padding = data.scores.isinf() & (data.scores < 0)

        # 2. compute the probabilities for each pair of (question, section) assigned by the model
        retriever_logprobs = _compute_retriever_logprobs(data, is_padding)

        # 3. compute the reference probabilities for each pair of (question, section)
        data_targets = _compute_data_targets(data, is_padding)

        # 4.1 compute the number of positives
        if _n_positive is None:
            _n_positive = data_targets.sum(dim=1)
            if (data_targets.sum(dim=1) == 0).any():
                raise ValueError("The batch contains a question without positive section.")

        # 4.2 compute the model probabilities
        model_probs = retriever_logprobs.exp().detach() if _model_logprobs is None else _model_logprobs.exp().detach()

        # 5. Compute the loss: KL divergences between the model and the sampling distributions
        w = 1 / _n_positive[:, None] * (model_probs - data_targets)
        loss = torch.sum(w.detach() * retriever_logprobs, dim=-1).mean()

        # 6. Compute the KL divergences between the model and the sampling distributions
        # KL ( p_ref(z) | p_model(z)) for `p_ref` = score, bm25, faiss
        kls = {
            key: _compute_kld(retriever_logprobs, ref_scores)
            for key, ref_scores in {
                "score": data.scores,
                "bm25": data.bm25,
                "faiss": data.faiss,
                "data": torch.where(data_targets > 0, 0.0, -math.inf),
            }.items()
            if ref_scores is not None
        }

        return {
            "loss": loss,
            "_targets": data_targets,
            "_logits": retriever_logprobs,
            "_n_positive": _n_positive,
            **{f"kl_{k}": kl.mean().detach() for k, kl in kls.items()},
        }

    # def forward_backward(
    #     self,
    #     batch: dict[str, torch.Tensor],
    #     fwd_fn: Callable[[dict], dict],
    #     backward_fn: Callable[[torch.Tensor], None] | None = None,
    #     loss_scaler: float | None = None,
    #     **kwargs: Any,
    # ) -> dict[str, torch.Tensor]:
    #     with torch.no_grad():
    #         fwd_out = fwd_fn(batch)


def _compute_retriever_logprobs(data: GradientInputs, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the log-probabilities for each pair of (question, section) assigned by the model."""
    if data.hd.dim() == 2:  # noqa: PLR2004
        retriever_logprobs = torch.einsum("bh, dh -> bd", data.hq, data.hd)
    elif data.hd.dim() == 3:  # noqa: PLR2004
        retriever_logprobs = torch.einsum("bh, bdh -> bd", data.hq, data.hd)
    else:
        raise ValueError(f"Invalid dimension for `hd`: {data.hd.shape}")
    if mask is not None:
        retriever_logprobs.masked_fill_(mask, -torch.inf)

    return retriever_logprobs.log_softmax(dim=-1)


@torch.no_grad()
def _compute_data_targets(data: GradientInputs, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the reference probabilities for each pair of (question, section)."""
    data_targets = data.targets.float()
    if mask is not None:
        data_targets.masked_fill_(mask, 0.0)

    return data_targets


@torch.jit.script  # type: ignore
def _masked_logprobs(logits: torch.Tensor, is_defined: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~is_defined, -math.inf)
    logits = logits.log_softmax(dim=-1)
    return logits


@torch.jit.script  # type: ignore
def _compute_kld(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL divergence between the model and the data."""
    p_is_defined = p_logits.isfinite()
    q_is_defined = q_logits.isfinite()
    p_logpropbs = _masked_logprobs(p_logits, p_is_defined)
    q_logprobs = _masked_logprobs(q_logits, p_is_defined)

    # compute the KL
    kl_div_terms = q_logprobs.exp() * (q_logprobs - p_logpropbs)
    kl_div_terms.masked_fill_(~p_is_defined | ~q_is_defined, 0)
    return kl_div_terms.sum(dim=-1)


# def _forward_backward_step(
#     *,
#     fabric: L.Fabric,
#     ranker: Ranker,
#     batch: dict[str, torch.Tensor],
#     loss_scaler: typing.Optional[float] = None,
#     **kwargs,  # noqa: ANN003
# ) -> dict[str, float]:
#     step_metrics = ranker.evaluate(batch, **kwargs)
#     loss = step_metrics["loss"]
#     if loss_scaler is not None:
#         loss *= loss_scaler
#     fabric.backward(loss)
#     return step_metrics


# def _chunked_forward_backward_step(
#     *,
#     fabric: L.Fabric,
#     ranker: Ranker,
#     batch: dict[str, torch.Tensor],
#     loss_scaler: typing.Optional[float] = None,
#     chunk_size: int = 8,
# ) -> dict[str, float]:
#     pipes.pprint_batch(batch, header="Chunked Training Step")

#     # Forward pass without greadients
#     with torch.no_grad():
#         full_outputs = ranker.training_step(batch, filter_output=False)
#         rich.print(full_outputs)
#         batch["section.precomputed_logprobs"] = full_outputs["_logits"].log_softmax(dim=-1)

#     section_input_ids = batch["section.input_ids"]
#     {2: True, 3: False}[section_input_ids.ndim]
#     # TODO: implement flattened version

#     # compute the gradients by chunks
#     for batch_chunk in _iter_chunks(batch, chunk_size=chunk_size):
#         pipes.pprint_batch(batch_chunk, header="Chunk")
#         _forward_backward_step(
#             fabric=fabric,
#             ranker=ranker,
#             batch=batch_chunk,
#             loss_scaler=loss_scaler,
#         )

#     return full_outputs


def _iter_chunks(
    batch: dict[str, torch.Tensor],
    chunk_size: int = 8,
    prefix: str = "section.",
) -> Iterable[dict[str, torch.Tensor]]:
    section_input_ids = batch[f"{prefix}input_ids"]
    n_doc = section_input_ids.shape[1]
    doc_keys = [k for k in batch if k.startswith(prefix)]
    other_keys = [k for k in batch if not k.startswith(prefix)]
    for i in range(0, n_doc, chunk_size):
        yield {
            **{k: batch[k][:, i : i + chunk_size].contiguous() for k in doc_keys},
            **{k: batch[k] for k in other_keys},
        }
