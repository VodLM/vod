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
        is_padding = data.scores.isinf()

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
        is_negative = ~data.targets
        q_with_positives = ~is_negative.all(dim=-1)
        data_zero_prob = is_negative | ~q_with_positives[:, None]
        data_logits = torch.where(
            data_zero_prob,
            self.log_eps,
            torch.zeros_like(model_logits),
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

        # import rich
        # rich.print({
        #     "loss": loss,
        #     "kl_div": kl_div,
        #     # "q_with_positives": q_with_positives,
        #     # "model_logits": model_logits,
        #     "data_logits": data_logits,
        # })

        return {"loss": loss, "_targets": data.targets, "_logits": model_logits}
