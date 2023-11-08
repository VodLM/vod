import torch
import torch.nn
import vod_types as vt
from vod_models.vod_gradients.retrieval import _compute_retriever_scores

from .base import Gradients


class MarginalLikelihoodGradients(Gradients):
    """Compute the marginal likelihood or REALM systems."""

    def __call__(
        self,
        *,
        batch: vt.RealmBatch,
        query_encoding: torch.Tensor,
        section_encoding: torch.Tensor,
        lm_logits: torch.Tensor,
    ) -> vt.RealmOutput:
        """Compute the gradients/loss of Realm system using the marginal likelihood with the in-batch approximation."""
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
        input_ids: torch.Tensor = batch.lm__input_ids  # type: ignore
        attention_mask: torch.Tensor = batch.lm__attention_mask  # type: ignore
        logp_x__z = _compute_lm_logprobs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lm_logits=lm_logits,
        )

        # 4. compute the loss/marginal likelihood
        logp_x = torch.logsumexp(retriever_logprobs + logp_x__z, dim=-1)
        loss = -logp_x.mean()

        return vt.RealmOutput(
            loss=loss,
            retriever_scores=retriever_scores,
        )


@torch.jit.script
def _compute_lm_logprobs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, lm_logits: torch.Tensor
) -> torch.Tensor:
    """Compute the log-probabilities of the language model."""
    shifted_input_ids = input_ids[..., 1:]
    shifted_attention_mask = attention_mask[..., 1:]
    shifted_logits = lm_logits[..., :-1, :]
    shifted_logits = shifted_logits.masked_fill((shifted_attention_mask == 0).unsqueeze(-1), -torch.inf)
    shifted_logp = (
        torch.nn.functional.log_softmax(shifted_logits, dim=-1)[..., :-1]
        .gather(dim=-1, index=shifted_input_ids.unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(shifted_attention_mask == 0, 0.0)
    )
    return shifted_logp.sum(dim=-1) / shifted_attention_mask.sum(dim=-1)
