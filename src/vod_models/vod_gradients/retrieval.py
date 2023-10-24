import math
import typing as typ
import warnings

import torch
import torch.nn
import vod_types as vt

from .base import Gradients

GuidanceType = typ.Literal["sparse", "zero"]


class RetrievalGradients(Gradients):
    """Compute the KL divergence between the model and the data."""

    def __init__(
        self,
        guidance: GuidanceType = "zero",
        guidance_weight: float = 0.0,
        self_supervision_weight: float = 0.0,
        score_decay: float = 0.0,
    ):
        super().__init__()
        self.guidance = guidance
        self.guidance_weight = guidance_weight
        self.self_supervision_weight = self_supervision_weight
        self.score_decay = score_decay

    def __call__(
        self,
        *,
        batch: vt.RealmBatch,
        query_encoding: torch.Tensor,  # the encoding of the queries
        section_encoding: torch.Tensor,  # the encoding of the documents/sections
    ) -> vt.RealmOutput:
        """Compute the KL divergence between the model and the data."""
        # 1. Determine the masked sections
        is_padding = batch.section__score.isinf() & (batch.section__score < 0)

        # 2. compute the probabilities for each pair of (question, section) assigned by the model
        retriever_scores = _compute_retriever_scores(
            query_encoding=query_encoding,
            section_encoding=section_encoding,
            mask=is_padding,
        )
        retriever_logprobs = retriever_scores.log_softmax(dim=-1)

        # 3. compute the reference probabilities for each pair of (question, section)
        data_binary_targets = _cast_data_targets(batch.section__relevance, is_padding)

        # 4 compute the number of positives
        n_positives = data_binary_targets.sum(dim=1)
        if (n_positives == 0).any():
            warnings.warn("This batch contains a question without positive section.", stacklevel=2)

        n_positives = torch.where(n_positives == 0, (~is_padding).float().sum(dim=1), n_positives)

        # 5. Compute the loss: KL divergences between the model and the sampling distributions
        loss = _compute_loss(
            ref_retriever_probs=retriever_logprobs.exp().detach(),
            retriever_logprobs=retriever_logprobs,
            data_targets=data_binary_targets,
            n_positive=n_positives,
            section_mask=is_padding,
        )

        # compute auxiliary losses
        loss, diagnostics = self._auxiliary_losses(
            batch=batch,
            loss=loss,
            retriever_scores=retriever_scores,
            data_targets=data_binary_targets,
            n_positives=n_positives,
        )

        # 7. Compute the KL divergences between the model and the sampling distributions
        # KL ( p_ref(z) | p_model(z)) for `p_ref` = score, sparse, dense, data
        for key, ref_scores in {
            "kl_score": batch.section__score,
            "kl_sparse": batch.section__sparse,
            "kl_dense": batch.section__dense,
        }.items():
            if ref_scores is None:
                continue
            diagnostics[key] = _compute_kld(retriever_logprobs, ref_scores).mean().detach()

        return vt.RealmOutput(
            loss=loss,
            retriever_scores=retriever_scores,
            diagnostics=diagnostics,
        )

    def _auxiliary_losses(
        self,
        *,
        batch: vt.RealmBatch,
        loss: torch.Tensor,
        retriever_scores: torch.Tensor,
        data_targets: torch.Tensor,
        n_positives: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        retriever_logprobs = retriever_scores.log_softmax(dim=-1)
        aux_losses = {}
        if self.guidance_weight > 0:
            guidance_loss = _guidance_loss(batch, retriever_logprobs, guidance_type=self.guidance)  # type: ignore
            loss += self.guidance_weight * guidance_loss
            aux_losses[f"{self.guidance}_guidance"] = guidance_loss
        if self.self_supervision_weight > 0:
            loss_self_supervision = _self_supervision_loss(data_targets, retriever_logprobs, n_positives)
            loss += self.self_supervision_weight * loss_self_supervision
            aux_losses["self_supervision"] = loss_self_supervision
        if self.score_decay > 0:
            score_decay_loss = _score_decay_loss(retriever_scores)
            loss += self.score_decay * score_decay_loss
            aux_losses["score_decay"] = score_decay_loss

        return loss, aux_losses


def _guidance_loss(
    batch: vt.RealmBatch,
    retriever_logprobs: torch.Tensor,
    guidance_type: GuidanceType = "zero",
) -> torch.Tensor:
    """Guide the model towards the reference scores."""
    ref_score = {
        "sparse": batch.section__sparse,
        "zero": torch.zeros_like(batch.section__score),
    }[guidance_type]
    return _compute_hubert_loss(retriever_logprobs, ref_score)


def _self_supervision_loss(
    data_targets: torch.Tensor,
    retriever_logprobs: torch.Tensor,
    n_positives: torch.Tensor,
) -> torch.Tensor:
    """Force the model to assign more mass on the highest scoring section."""
    retriever_logprobs_pos = torch.where(data_targets > 0, retriever_logprobs, -math.inf)
    self_target_indices = torch.argmax(retriever_logprobs_pos, dim=-1)
    return torch.nn.functional.cross_entropy(
        retriever_logprobs_pos[n_positives > 0],
        self_target_indices[n_positives > 0],
    )


def _score_decay_loss(retriever_scores: torch.Tensor) -> torch.Tensor:
    """Center the scores around 0, ensure better numerical stability for downstream vector search."""
    return retriever_scores[retriever_scores.isfinite()].pow(2).mean()


@torch.jit.script
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


@torch.jit.script
def _compute_hubert_loss(a_scores: torch.Tensor, b_scores: torch.Tensor) -> torch.Tensor:
    mask = a_scores.isfinite() & b_scores.isfinite()
    return torch.nn.functional.huber_loss(a_scores[mask], b_scores[mask])


@torch.jit.script
def _compute_retriever_scores(
    query_encoding: torch.Tensor,
    section_encoding: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the score for each pair of (question, section) assigned by the model."""
    if section_encoding.dim() == 2:  # noqa: PLR2004
        retriever_scores = torch.einsum("bh, dh -> bd", query_encoding, section_encoding)
    elif section_encoding.dim() == 3:  # noqa: PLR2004
        retriever_scores = torch.einsum("bh, bdh -> bd", query_encoding, section_encoding)
    else:
        raise ValueError(f"Invalid dimension for `hd`: {section_encoding.shape}")

    # Mask the scores
    retriever_scores.masked_fill_(mask, -torch.inf)

    return retriever_scores


@torch.jit.script
def _cast_data_targets(
    data_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the reference probabilities for each pair of (question, section)."""
    data_targets = (data_targets > 0).float()
    data_targets.masked_fill_(mask, 0.0)

    return data_targets


@torch.jit.script
def _masked_logprobs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -math.inf)
    logits = logits.log_softmax(dim=-1)
    return logits


@torch.jit.script
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
