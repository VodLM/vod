import math
import typing as typ
import warnings

import lightning as L
import torch
import torch.nn
from vod_gradients import base


class SupervisedRetrievalGradients(base.Gradients):
    """Compute the KL divergence between the model and the data."""

    def __init__(
        self,
        guidance_weight: float = 0.0,
        guidance: typ.Literal["sparse", "zero"] = "sparse",
        self_supervision_weight: float = 0.0,
        anchor_weight: float = 0.0,
    ):
        super().__init__()
        self.guidance = guidance
        self.guidance_weight = guidance_weight
        self.self_supervision_weight = self_supervision_weight
        self.anchor_weight = anchor_weight

    def __call__(
        self,
        inputs: dict[str, torch.Tensor],
        skip_diagnostics: bool = False,
        retriever_scores: None | torch.Tensor = None,
        **kws: typ.Any,
    ) -> dict[str, torch.Tensor]:
        """Parse the inputs and compute the loss."""
        data = base.GradientInputs(**inputs)

        # 1. Compute the KL divergence between the model and the data
        # Determine the masked sections
        is_padding = data.scores.isinf() & (data.scores < 0)

        # 2. compute the probabilities for each pair of (question, section) assigned by the model
        if data.hq is None or data.hd is None:
            if retriever_scores is None:
                raise ValueError("`hq` and `hd` were not provided, `retriever_scores` must be specified.")
        else:
            retriever_scores = _compute_retriever_scores(hq=data.hq, hd=data.hd, mask=is_padding)
        retriever_logprobs = retriever_scores.log_softmax(dim=-1)

        # 3. compute the reference probabilities for each pair of (question, section)
        data_targets = _cast_data_targets(data.targets, is_padding)

        # 4.1 compute the number of positives
        n_positives = data_targets.sum(dim=1)
        if (n_positives == 0).any():
            warnings.warn("This batch contains a question without positive section.", stacklevel=2)

        n_positives = torch.where(n_positives == 0, (~is_padding).float().sum(dim=1), n_positives)

        # 5. Compute the loss: KL divergences between the model and the sampling distributions
        loss = _compute_loss(
            ref_retriever_probs=retriever_logprobs.exp().detach(),
            retriever_logprobs=retriever_logprobs,
            data_targets=data_targets,
            n_positive=n_positives,
            section_mask=is_padding,
        )

        # compute auxiliary losses
        aux_losses = {}
        if self.guidance_weight > 0:
            # Guide the model towards the reference scores
            ref_score = {
                "sparse": data.sparse,
                "zero": torch.zeros_like(data.scores),
            }[self.guidance]
            guidance_loss = _compute_hubert_loss(retriever_logprobs, ref_score)
            loss += self.guidance_weight * guidance_loss
            aux_losses[f"{self.guidance}_guidance"] = guidance_loss
        if self.self_supervision_weight > 0:
            # Force the model to assign more mass on the highest scoring section
            retriever_logprobs_pos = torch.where(data_targets > 0, retriever_logprobs, -math.inf)
            self_target_indices = torch.argmax(retriever_logprobs_pos, dim=-1)
            loss_self_supervision = torch.nn.functional.cross_entropy(
                retriever_logprobs_pos[n_positives > 0],
                self_target_indices[n_positives > 0],
            )
            loss += self.self_supervision_weight * loss_self_supervision
            aux_losses["self_supervision"] = loss_self_supervision
        if self.anchor_weight > 0:
            # Center the scores around 0
            anchor_loss = retriever_scores[retriever_scores.isfinite()].pow(2).mean()
            loss += self.anchor_weight * anchor_loss
            aux_losses["anchor"] = anchor_loss

        # 7. Compute the KL divergences between the model and the sampling distributions
        # KL ( p_ref(z) | p_model(z)) for `p_ref` = score, sparse, dense
        if skip_diagnostics:
            kls = {}
        else:
            kls = {
                key: _compute_kld(retriever_logprobs, ref_scores)
                for key, ref_scores in {
                    "score": data.scores,
                    "sparse": data.sparse,
                    "dense": data.dense,
                    "data": torch.where(data_targets > 0, 0.0, -math.inf),
                }.items()
                if ref_scores is not None
            }

        return {
            "loss": loss,
            "_targets": data_targets,
            "_logits": retriever_logprobs,
            "_n_positives": n_positives,
            **{f"kl_{k}": kl.mean().detach() for k, kl in kls.items()},
            **aux_losses,
        }

    def forward_backward(
        self,
        batch: dict[str, torch.Tensor],
        fwd_fn: typ.Callable[[dict], dict],
        fabric: None | L.Fabric = None,
        loss_scaler: float | None = None,
        no_backward_sync: bool = False,
        **kws: typ.Any,
    ) -> dict[str, torch.Tensor]:
        """Forward and backward pass with gradient accumulation."""
        return super().forward_backward(
            batch,
            fwd_fn=fwd_fn,
            fabric=fabric,
            loss_scaler=loss_scaler,
            no_backward_sync=no_backward_sync,
            fwd_kws={**kws, "compute_metrics": True, "mode": "evaluate"},
        )


@torch.jit.script  # type: ignore
def _compute_loss(
    *,
    ref_retriever_probs: torch.Tensor,
    retriever_logprobs: torch.Tensor,
    data_targets: torch.Tensor,
    section_mask: torch.Tensor,
    n_positive: torch.Tensor,
) -> torch.Tensor:
    """KL divervgence between the model and the data.

    `nabla kld = 1 / Np * sum_{i=1}^Np (p_i - 1[i in P]) nabla log p_i`
    """
    w = 1 / n_positive[:, None] * (ref_retriever_probs - data_targets)
    loss = torch.sum(
        torch.where(section_mask, 0, w.detach() * retriever_logprobs),
        dim=-1,
    )
    has_positive_section = n_positive > 0
    if has_positive_section.any():
        loss = torch.where(has_positive_section, loss, torch.zeros_like(loss))
        loss = loss.sum() / has_positive_section.float().sum()
    else:
        loss = torch.full_like(loss, fill_value=math.nan).sum()
    return loss


@torch.jit.script  # type: ignore
def _compute_hubert_loss(a_scores: torch.Tensor, b_scores: torch.Tensor) -> torch.Tensor:
    mask = a_scores.isfinite() & b_scores.isfinite()
    return torch.nn.functional.huber_loss(a_scores[mask], b_scores[mask])


@torch.jit.script  # type: ignore
def _compute_retriever_scores(
    hq: torch.Tensor,
    hd: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the score for each pair of (question, section) assigned by the model."""
    if hd.dim() == 2:  # noqa: PLR2004
        retriever_scores = torch.einsum("bh, dh -> bd", hq, hd)
    elif hd.dim() == 3:  # noqa: PLR2004
        retriever_scores = torch.einsum("bh, bdh -> bd", hq, hd)
    else:
        raise ValueError(f"Invalid dimension for `hd`: {hd.shape}")

    # Mask the scores
    retriever_scores.masked_fill_(mask, -torch.inf)

    return retriever_scores


@torch.jit.script  # type: ignore
@torch.no_grad()
def _cast_data_targets(
    data_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the reference probabilities for each pair of (question, section)."""
    data_targets = data_targets.float()
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
