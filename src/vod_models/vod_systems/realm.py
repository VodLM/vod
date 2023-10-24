import functools
import typing as typ

import omegaconf as omg
import torch
import transformers
import vod_configs
import vod_types as vt
from transformers import modeling_outputs
from vod_models import vod_encoder, vod_gradients
from vod_models.support import apply_tweaks

from .ranker import Ranker

AutoregressiveLanguageModel = (
    transformers.BlenderbotForConditionalGeneration | transformers.LlamaForCausalLM | transformers.OPTForCausalLM
)


class Realm(Ranker):
    """A Retrieval-augmented Language Model."""

    def __init__(
        self,
        encoder: vod_encoder.VodEncoder,
        lm: AutoregressiveLanguageModel,
        gradients: vod_gradients.Gradients,
        optimizer: None | dict | omg.DictConfig | functools.partial = None,
        scheduler: None | dict | omg.DictConfig | functools.partial = None,
        tweaks: None | dict | omg.DictConfig | vod_configs.TweaksConfig = None,
    ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            encoder=encoder,
            gradients=gradients,
            tweaks=tweaks,
        )

        # Prepare the language model with optional optimizations
        self.lm = apply_tweaks(lm, tweaks)  # type: ignore

    def evaluate(
        self,
        batch: vt.RealmBatch,
        **kws: typ.Any,
    ) -> vt.RealmOutput:  # noqa: ARG002
        """Run a forward pass and compute the gradients."""
        encoded = self.encode(batch)
        lm_logits = self._forward_lm(
            input_ids=batch.lm__input_ids,  # type: ignore
            attention_mask=batch.lm__attention_mask,  # type: ignore
        )
        return self.gradients(batch=batch, **encoded, lm_logits=lm_logits)

    def _forward_lm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the language model."""
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])
        attention_mask = attention_mask.view(-1, input_shape[-1])
        output: modeling_outputs.CausalLMOutputWithPast = self.lm(input_ids, attention_mask)
        lm_logits = output.logits
        return lm_logits.view(*input_shape, -1)
