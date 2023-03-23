import abc
import math
from typing import Optional

import torch.nn
from pydantic.fields import Field
from pydantic.main import BaseModel
from torch.distributions import Categorical


class Gradients(torch.nn.Module):
    @abc.abstractmethod
    def forward(self, intermediate_results: dict) -> dict:
        raise NotImplementedError


class SupervisedGradientsInputs(BaseModel):
    class Config:
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


class SelfSupervisedGradients(Gradients):
    def forward(self, intermediate_results: dict) -> dict:
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

    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        if eps:
            self.log_eps = math.log(eps)
        else:
            self.log_eps = -math.inf

    def forward(self, intermediate_results: dict) -> dict:
        data = SupervisedGradientsInputs(**intermediate_results)

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
        # todo: remove this, should not be necessary on Frank.
        #  raise an error if this is the case.
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
        loss = (q_with_positives * kl_div).sum() / q_with_positives.sum()

        return {
            "loss": loss,
            **_make_evaluation_data(
                targets=data.targets,
                model_logits=model_logits,
                retrieval_scores=data.scores,
            ),
        }


@torch.no_grad()
def _make_evaluation_data(
    targets: torch.Tensor,
    model_logits: torch.Tensor,
    retrieval_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sorted_ids = model_logits.argsort(dim=-1, descending=True)
    targets_ = targets.gather(dim=-1, index=sorted_ids)
    model_logits_ = model_logits.gather(dim=-1, index=sorted_ids)

    # compute the KL divergence between the model and the data
    is_defined = retrieval_scores.isfinite()
    retrieval_logits_ = retrieval_scores.masked_fill(~is_defined, -math.inf)
    retrieval_logits_ = retrieval_logits_.log_softmax(dim=-1)

    kl_div_terms = -retrieval_logits_.exp() * (model_logits_ - retrieval_logits_)
    kl_div_terms.masked_fill_(~is_defined, 0)
    kl_div = kl_div_terms.sum(dim=-1)

    return {
        "_targets": targets_ > 0,
        "_logits": model_logits_,
        "kl_sampler": kl_div.mean(),
    }
