# pylint: disable=no-member
from __future__ import annotations

import abc
import copy
import math
from typing import Any, Callable, Iterable, Optional

import pydantic
import torch
import torch.nn
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
        fwd_fn: None | Callable[[dict], dict],
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
        loss_scaler: Optional[float] = None,
        backward_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass with a backward pass."""
        fwd_output = fwd_fn(batch, **kwargs) if fwd_fn is not None else {}
        grad_output = self({**batch, **fwd_output}, **kwargs)

        # compute the loss
        loss = grad_output["loss"]
        if loss_scaler is not None:
            loss *= loss_scaler

        # backward pass
        backward_kwargs = backward_kwargs or {}
        if backward_fn is None:
            loss.backward(**backward_kwargs)
        else:
            backward_fn(loss, **backward_kwargs)

        return grad_output


class GradientInputs(pydantic.BaseModel):
    """collection of inputs for the supervised gradients model."""

    class Config:
        """pydantic config."""

        arbitrary_types_allowed = True

    hq: torch.Tensor
    hd: torch.Tensor
    targets: torch.Tensor = pydantic.Field(
        ...,
        description="Retrieval labels.",
        alias="section.label",
    )
    scores: torch.Tensor = pydantic.Field(
        ...,
        description="Retrieval scores.",
        alias="section.score",
    )
    bm25: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="bm25 Retrieval scores.",
        alias="section.bm25",
    )
    faiss: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="faiss Retrieval scores.",
        alias="section.faiss",
    )

    # Precomputed logprobs from the model.
    pre_logits: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="Precomputed logprobs from the model.",
        alias="section.pre_logits",
    )

    pre_n_positive: Optional[torch.Tensor] = pydantic.Field(
        None,
        description="Precomputed total number of positive documents.",
        alias="section.pre_n_positive",
    )


class SelfSupervisedGradients(Gradients):
    """Compute the gradients for the `self-supervised` method."""

    def forward(self, inputs: dict, **kwargs: Any) -> dict:
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
        section_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        if eps:
            self.log_eps = math.log(eps)
        else:
            self.log_eps = -math.inf

        self.bm25_guidance_weight = bm25_guidance_weight
        self.self_supervision_weight = self_supervision_weight
        self.section_chunk_size = section_chunk_size

    def forward(
        self, inputs: dict[str, torch.Tensor], skip_diagnostics: bool = False, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
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
        if data.pre_n_positive is None:
            _n_positive = data_targets.sum(dim=1)
            if (data_targets.sum(dim=1) == 0).any():
                raise ValueError("The batch contains a question without positive section.")
        else:
            _n_positive = data.pre_n_positive

        # 4.2 compute the model probabilities
        model_probs = retriever_logprobs.exp().detach() if data.pre_logits is None else data.pre_logits.exp().detach()

        # 5. Compute the loss: KL divergences between the model and the sampling distributions
        w = 1 / _n_positive[:, None] * (model_probs - data_targets)
        loss = torch.sum(w.detach() * retriever_logprobs, dim=-1).mean()

        # 6. Compute the KL divergences between the model and the sampling distributions
        # KL ( p_ref(z) | p_model(z)) for `p_ref` = score, bm25, faiss
        if skip_diagnostics:
            kls = {}
        else:
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

    def forward_backward(
        self,
        batch: dict[str, torch.Tensor],
        fwd_fn: Callable[[dict], dict],
        backward_fn: Callable[[torch.Tensor], None] | None = None,
        loss_scaler: float | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward and backward pass with gradient accumulation."""
        if self.section_chunk_size is None:
            return super().forward_backward(
                batch,
                fwd_fn=fwd_fn,
                backward_fn=backward_fn,
                loss_scaler=loss_scaler,
                **kwargs,
            )

        # Gather the section/question attributes
        section_input_ids = batch["section.input_ids"]
        question_input_ids = batch["question.input_ids"]

        # Evaluate the number of sections per question
        if section_input_ids.ndim == 3:  # noqa: PLR2004
            n_sections = section_input_ids.shape[1]
            eff_chunk_size = self.section_chunk_size
        elif section_input_ids.ndim == 2:  # noqa: PLR2004
            n_sections = section_input_ids.shape[0]
            eff_chunk_size = self.section_chunk_size * question_input_ids.shape[0]
        else:
            raise ValueError(f"Invalid section_input_ids shape: `{section_input_ids.shape}`")

        # Skip chunk-processing if the number of sections is smaller than the chunk size
        if n_sections <= self.section_chunk_size:
            return super().forward_backward(
                batch,
                fwd_fn=fwd_fn,
                backward_fn=backward_fn,
                loss_scaler=loss_scaler,
                **kwargs,
            )
        # Encode the questions and sections - without gradients
        with torch.no_grad():
            encodings = fwd_fn(batch, **kwargs)

        # Run a forward pass using all sections, pre-compute the retriever log_probs
        with torch.no_grad():
            full_output = self({**batch, **encodings})
            precomputed = {f"section.pre{k}": full_output[k] for k in full_output if k.startswith("_")}

        # Compute the forward/backward pass for each chunk of sections
        batch_sections = {**{k: v for k, v in batch.items() if k.startswith("section.")}, **precomputed}
        question_batch = {k: v for k, v in batch.items() if k.startswith("question.")}
        for section_chunk in _iter_chunks(
            batch_sections,
            chunk_size=eff_chunk_size,
            ref_key="section.input_ids",
            pass_through=[  # don't chunk these
                "section.pre_n_positive",
            ],
            always_batched=[  # always consider these as batched
                "section.label",
                "section.score",
                "section.bm25",
                "section.faiss",
                "section.pre_targets",
                "section.pre_logits",
                "hd",
            ],
        ):
            # Encode the chunk of sections with the questions
            # NB: the algoerithm could be made more efficient by pre-computing the question encodings
            #     and only computing the section encodings for each chunk. Nevertheless, this requires retaining
            #     the gradient graph, and this one would grow with the number of chunks. This is why we recompute
            #     the question encodings for each chunk.
            batch_chunk = {**question_batch, **section_chunk}
            chunk_encodings = fwd_fn(batch_chunk, **kwargs)

            # Evaluate the gradients and backard for the chunk
            section_chunk_batch = {
                **batch_chunk,
                **chunk_encodings,
            }

            super().forward_backward(
                section_chunk_batch,
                fwd_fn=None,
                backward_fn=backward_fn,
                loss_scaler=loss_scaler,
                skip_diagnostics=True,
                **kwargs,
            )

        return full_output


def _iter_chunks(
    batch: dict[str, torch.Tensor],
    chunk_size: int = 8,
    ref_key: str = "section.input_ids",
    pass_through: list[str] | None = None,
    always_batched: list[str] | None = None,
) -> Iterable[dict[str, torch.Tensor]]:
    always_batched = always_batched or []
    pass_through = pass_through or []
    constants = {k: batch[k] for k in pass_through}
    batch = {k: v for k, v in batch.items() if k not in pass_through}

    # infer the number of documents
    ref_input_ids = batch[ref_key]
    if ref_input_ids.ndim == 3:  # noqa: PLR2004
        n_doc = ref_input_ids.shape[1]
        batched = True
    elif ref_input_ids.ndim == 2:  # noqa: PLR2004
        n_doc = ref_input_ids.shape[0]
        batched = False
    else:
        raise ValueError(f"Invalid `{ref_input_ids}` shape: `{ref_input_ids.shape}`")

    # iterate over the chunks
    for i in range(0, n_doc, chunk_size):
        chunk = copy.copy(constants)
        for key, value in batch.items():
            if batched or key in always_batched:
                chunk[key] = value[:, i : i + chunk_size].contiguous()
            else:
                chunk[key] = value[i : i + chunk_size].contiguous()

        yield chunk


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
def _masked_logprobs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -math.inf)
    logits = logits.log_softmax(dim=-1)
    return logits


@torch.jit.script  # type: ignore
def _compute_kld(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL divergence between the model and the data."""
    p_is_defined = p_logits.isfinite()
    p_logpropbs = _masked_logprobs(p_logits, p_is_defined)

    q_is_defined = q_logits.isfinite()
    q_logprobs = _masked_logprobs(q_logits, q_is_defined)

    # compute the KL
    kl_div_terms = torch.where(
        p_is_defined & q_is_defined,
        q_logprobs.exp() * (q_logprobs - p_logpropbs),
        0.0,
    )
    return kl_div_terms.sum(dim=-1)
