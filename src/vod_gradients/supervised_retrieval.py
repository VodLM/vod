from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, Iterable, Optional

import lightning as L
import torch
import torch.nn

from src.vod_gradients import base


class KlDivGradients(base.Gradients):
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

    def __call__(
        self,
        inputs: dict[str, torch.Tensor],
        skip_diagnostics: bool = False,
        retriever_logprobs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Parse the inputs and compute the loss."""
        data = base.GradientInputs(**inputs)

        # 1. Compute the KL divergence between the model and the data
        # Determine the masked sections
        is_padding = data.scores.isinf() & (data.scores < 0)

        # 2. compute the probabilities for each pair of (question, section) assigned by the model
        if retriever_logprobs is None:
            retriever_logprobs = _compute_retriever_logprobs(data, is_padding)
        else:
            retriever_logprobs = retriever_logprobs.masked_fill(is_padding, -torch.inf)

        # 3. compute the reference probabilities for each pair of (question, section)
        data_targets = _compute_data_targets(data, is_padding)

        # 4.1 compute the number of positives
        if data.pre_n_positive is None:
            _n_positive = data_targets.sum(dim=1)
            if (_n_positive == 0).any():
                warnings.warn("This batch contains a question without positive section.", stacklevel=2)

            _n_positive = torch.where(_n_positive == 0, (~is_padding).float().sum(dim=1), _n_positive)
            ...
        else:
            _n_positive = data.pre_n_positive

        # 4.2 compute the model probabilities
        model_probs = retriever_logprobs.exp().detach() if data.pre_logits is None else data.pre_logits.exp().detach()

        # 5. Compute the loss: KL divergences between the model and the sampling distributions
        w = 1 / _n_positive[:, None] * (model_probs - data_targets)
        loss = torch.sum(w.detach() * retriever_logprobs, dim=-1)
        per_sample_mask = _n_positive > 0
        if per_sample_mask.any():
            loss = torch.where(per_sample_mask, loss, torch.zeros_like(loss))
            loss = loss.sum() / per_sample_mask.sum()
        else:
            loss = torch.full_like(loss, fill_value=math.nan).sum()

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
        fabric: None | L.Fabric = None,
        loss_scaler: float | None = None,
        no_backward_sync: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward and backward pass with gradient accumulation."""
        if self.section_chunk_size is None:
            return super().forward_backward(
                batch,
                fwd_fn=fwd_fn,
                fabric=fabric,
                loss_scaler=loss_scaler,
                no_backward_sync=no_backward_sync,
                fwd_kwargs={**kwargs, "compute_metrics": True, "mode": "evaluate"},
            )

        # Gather the section/question attributes
        section_input_ids = batch["section.input_ids"]
        question_input_ids = batch["question.input_ids"]

        # Evaluate the number of sections per question
        if section_input_ids.ndim == 3:  # noqa: PLR2004
            n_sections_per_q = section_input_ids.shape[1]
            eff_chunk_size = self.section_chunk_size
            n_chunks = n_sections_per_q // eff_chunk_size
        elif section_input_ids.ndim == 2:  # noqa: PLR2004
            n_sections_per_q = section_input_ids.shape[0] // question_input_ids.shape[0]
            eff_chunk_size = self.section_chunk_size * question_input_ids.shape[0]
            n_chunks = section_input_ids.shape[0] // eff_chunk_size
        else:
            raise ValueError(f"Invalid section_input_ids shape: `{section_input_ids.shape}`")

        # Skip chunk-processing if the number of sections is smaller than the chunk size
        if n_sections_per_q <= self.section_chunk_size:
            return super().forward_backward(
                batch,
                fwd_fn=fwd_fn,
                fabric=fabric,
                loss_scaler=loss_scaler,
                no_backward_sync=no_backward_sync,
                fwd_kwargs={**kwargs, "compute_metrics": True, "mode": "evaluate"},
            )

        # Run a forward pass using all sections, pre-compute the retriever log_probs and
        with torch.no_grad():
            full_output = fwd_fn(
                batch,
                **{**kwargs, "compute_metrics": True, "mode": "evaluate", "filter_output": False},
            )
            precomputed = {f"section.pre{k}": full_output[k] for k in full_output if k.startswith("_")}

        # Compute the forward/backward pass for each chunk of sections
        batch_sections = {**{k: v for k, v in batch.items() if k.startswith("section.")}, **precomputed}
        question_batch = {k: v for k, v in batch.items() if k.startswith("question.")}
        for j, section_chunk in enumerate(
            _iter_chunks(
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
                ],
            )
        ):
            # Encode the chunk of sections along with the questions
            # NB: the algoerithm could be made more efficient by pre-computing the question encodings
            #     and only computing the section encodings for each chunk. Nevertheless, this requires retaining
            #     the gradient graph, and this one would grow with the number of chunks. This is why we recompute
            #     the question encodings for each chunk.
            batch_chunk = {**question_batch, **section_chunk}
            is_last_chunk = j == n_chunks - 1
            super().forward_backward(
                batch_chunk,
                fwd_fn=fwd_fn,
                fabric=fabric,
                loss_scaler=loss_scaler,
                no_backward_sync=no_backward_sync or not is_last_chunk,
                fwd_kwargs={**kwargs, "compute_metrics": False, "mode": "evaluate", "filter_output": False},
                skip_diagnostics=True,
            )

        # Filter the output & return
        return {k: v for k, v in full_output.items() if not k.startswith("_")}


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

    # Infer the number of documents
    ref_input_ids = batch[ref_key]
    if ref_input_ids.ndim == 3:  # noqa: PLR2004
        n_doc = ref_input_ids.shape[1]
        batched = True
    elif ref_input_ids.ndim == 2:  # noqa: PLR2004
        n_doc = ref_input_ids.shape[0]
        batched = False
    else:
        raise ValueError(f"Invalid `{ref_input_ids}` shape: `{ref_input_ids.shape}`")

    # Iterate over and yield the chunks
    for i in range(0, n_doc, chunk_size):
        chunk = copy.copy(constants)
        for key, value in batch.items():
            if batched or key in always_batched:
                chunk[key] = value[:, i : i + chunk_size].contiguous()
            else:
                chunk[key] = value[i : i + chunk_size].contiguous()

        yield chunk


def _compute_retriever_logprobs(data: base.GradientInputs, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
def _compute_data_targets(data: base.GradientInputs, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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