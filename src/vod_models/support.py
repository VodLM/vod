import typing as typ

import omegaconf as omg
import torch
import transformers
import vod_configs
from hydra.utils import instantiate
from loguru import logger
from peft import mapping as peft_mapping
from peft import utils as peft_utils
from vod_models import vod_encoder

FIELD_MAPPING: dict[vod_encoder.VodEncoderInputType, str] = {"query": "query_encoding", "section": "section_encoding"}


def maybe_instantiate(conf_or_obj: typ.Any | omg.DictConfig, **kws: typ.Any) -> object:
    """Instantiate a config if needed."""
    if isinstance(conf_or_obj, (omg.DictConfig, dict)):
        return instantiate(conf_or_obj, **kws)
    return None


def apply_tweaks(  # noqa: C901, PLR0912
    module: torch.nn.Module | transformers.PreTrainedModel,
    tweaks: None | vod_configs.support.TweaksConfig,
) -> torch.nn.Module:
    """Apply training tweaks to the model."""
    if tweaks is None:
        return module
    if tweaks.prepare_for_kbit_training:
        # Cast parameters and register hooks to enable checkpointing
        module = peft_utils.other.prepare_model_for_kbit_training(
            module,
            use_gradient_checkpointing=tweaks.gradient_checkpointing,
        )
        logger.debug("Prepared for kbit training.")

    if tweaks.gradient_checkpointing:
        # Enable gradient checkpointing
        try:
            module.gradient_checkpointing_enable()  # type: ignore
            logger.debug("Gradient checkpointing Enabled.")
        except Exception as exc:
            logger.warning(f"Failed to enable gradient checkpointing: {exc}")

    if tweaks.peft_config is not None:
        # Apply PEFT optimizations
        module = peft_mapping.get_peft_model(module, tweaks.peft_config)  # type: ignore
        logger.debug(f"PEFT enabled `{type(tweaks.peft_config).__name__}`.")

    if tweaks.force_dtype is not None:
        # Cast the parameters to the specified dtype
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[tweaks.force_dtype]
        for _, param in module.named_parameters():
            if param.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                param.data = param.data.to(dtype)
        logger.debug(f"PEFT dtype `{dtype}` applied to float parameters.")

    if tweaks.compile:
        # Compile the model
        try:
            module = torch.compile(module, **tweaks.compile_kwargs)  # type: ignore
        except Exception as exc:
            if "ldconfig" in str(exc):
                raise RuntimeError(
                    ""
                    "Failed to compile the model. "
                    "This might be useful: `https://discuss.pytorch.org/t/dynamo-exceptions-with-distributeddataprallel-compile/186768/5?u=vlievin`"
                ) from exc
            raise exc
        logger.debug("`torch.compile` enabled (encoder)")

    return module
