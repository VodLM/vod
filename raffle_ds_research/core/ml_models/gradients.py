# pylint: disable=no-member
import abc
import math
from typing import Optional

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


class SupervisedGradientsInputs(BaseModel):
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

    def forward(self, intermediate_results: dict) -> dict:
        """Parse the inputs and compute the loss."""
        data = SupervisedGradientsInputs(**intermediate_results)

        # compute the scores for each pair of (question, section)
        # Note: we can add negative samples across batch here.
        scores = torch.einsum("bh,bdh->bd", data.hq, data.hd)

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

    def forward(self, intermediate_results: dict) -> dict:
        """Parse the inputs and compute the loss."""
        data = SupervisedGradientsInputs(**intermediate_results)

        # 1. Compute the KL divergence between the model and the data
        # Determine the masked sections
        is_padding = data.scores.isinf() & (data.scores < 0)

        # Define the logits of the model `q_model(z)`
        # set -inf wherever:
        #   - the section is masked (padding)
        model_scores = torch.einsum("bh, bdh -> bd", data.hq, data.hd)
        model_scores.masked_fill_(is_padding, -torch.inf)
        model_logits = model_scores.log_softmax(dim=-1)

        # Define the logits target distribution `p_data(z)`
        # set -log(eps) wherever:
        #   - the section is labelled as negative
        #   - except when all the sections for a given are negative
        is_negative_section = data.targets < 1
        q_with_positives = ~is_negative_section.all(dim=-1)
        if not q_with_positives.all():
            raise ValueError(
                f"Some questions do not have any positive section. prop={q_with_positives.float().mean():.2%}"
            )
        where_zero_prob = is_negative_section | ~q_with_positives[:, None]
        data_logits = torch.where(
            where_zero_prob,
            self.log_eps,  # set -log(eps) ~ -inf
            torch.zeros_like(model_logits),  # log(1) = 0
        )
        # set -inf wherever:
        #   - the section is masked (padding)
        data_logits.masked_fill_(is_padding, -torch.inf)
        data_logits = data_logits.log_softmax(dim=-1)

        # Compute the loss `KL( p_data(z) || p_model(z) )`
        # Exclude:
        #   - the sections which are masked (padding)
        #   - the questions where there are no positive
        mask = (data_logits.isinf() & (data_logits < 0)) | is_padding | ~q_with_positives[:, None]
        kl_div_terms = torch.where(mask, 0, -data_logits.exp() * (model_logits - data_logits))
        kl_div = kl_div_terms.sum(dim=-1)
        kl_div_loss = (q_with_positives * kl_div).sum() / q_with_positives.sum()

        # 2. Compute the self-supervision loss
        is_multi_targets = ((data.targets > 0).sum(dim=-1) > 1).float()
        ref_scores_where_positives = torch.where(data.targets, data.scores, -torch.inf)
        model_scores_where_positives = torch.where(data.targets, model_scores, -torch.inf)
        self_supervised_target = ref_scores_where_positives.argmax(dim=-1)
        self_supervised_loss = torch.nn.functional.cross_entropy(
            model_scores_where_positives,
            self_supervised_target,
            reduction="none",
        )
        self_supervised_loss = (is_multi_targets * self_supervised_loss).sum() / is_multi_targets.sum()

        # 3. Compute the KL divergences between the model and the sampling distributions
        # KL ( p_model(z) || p_ref(z)) for `p_ref` = score, bm25, faiss
        kls = {
            key: _compute_kld(model_logits, ref_scores)
            for key, ref_scores in {"score": data.scores, "bm25": data.bm25, "faiss": data.faiss}.items()
            if ref_scores is not None
        }

        # 4. Add the bm25 guidance loss
        kl_bm25 = kls.get("bm25", None)
        if self.bm25_guidance_weight > 0 and kl_bm25 is None:
            raise ValueError(f"bm25_guidance_weight={self.bm25_guidance_weight} but no bm25 scores are provided.")
        if kl_bm25 is not None:
            kl_bm25 = kl_bm25.mean()
        else:
            kl_bm25 = 0.0

        # 5. compute the final loss
        loss = kl_div_loss
        if self.bm25_guidance_weight > 0:
            loss += self.bm25_guidance_weight * kl_bm25
        if self.self_supervision_weight > 0:
            loss += self.self_supervision_weight * self_supervised_loss

        output = {
            "loss": loss,
            "kl_div_loss": kl_div_loss,
            "self_supervised_loss": self_supervised_loss,
            **_make_evaluation_data(
                targets=data.targets,
                model_logits=model_logits,
            ),
            **{f"kl_{k}": kl.mean().detach() for k, kl in kls.items()},
        }
        return output


def _compute_kld(
    model_logits: torch.Tensor,
    ref_scores: torch.Tensor,
) -> torch.Tensor:
    # compute the KL divergence between the model and the data
    is_defined = ref_scores.isfinite()
    ref_logits_ = ref_scores.masked_fill(~is_defined, -math.inf)
    ref_logits_ = ref_logits_.log_softmax(dim=-1)

    kl_div_terms = -ref_logits_.exp() * (model_logits - ref_logits_)
    kl_div_terms.masked_fill_(~is_defined, 0)
    kl_div = kl_div_terms.sum(dim=-1)
    return kl_div


@torch.no_grad()
def _make_evaluation_data(
    targets: torch.Tensor,
    model_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "_targets": targets > 0,
        "_logits": model_logits,
    }
