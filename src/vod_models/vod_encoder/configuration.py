from __future__ import annotations

import copy
from typing import Any, Optional

import transformers
from typing_extensions import Literal, TypeAlias

AggMethod: TypeAlias = Literal["mean", "max", "cls", "none"]

VodEncoderInputType: TypeAlias = Literal["query", "section"]


class VodPoolerConfig:
    """Configuration for a VOD head."""

    projection_size: Optional[int] = None
    output_activation: Optional[str] = None
    output_norm: Optional[str] = None
    agg_method: AggMethod = "mean"

    def __init__(
        self,
        projection_size: Optional[int] = None,
        output_activation: Optional[str] = None,
        output_norm: Optional[str] = None,
        agg_method: AggMethod = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.projection_size = projection_size
        self.output_activation = output_activation
        self.output_norm = output_norm
        self.agg_method = agg_method

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict."""
        return {
            "projection_size": self.projection_size,
            "output_activation": self.output_activation,
            "output_norm": self.output_norm,
            "agg_method": self.agg_method,
        }


class VodEncoderConfig(transformers.PretrainedConfig):
    """Configuration for a VOD encoder."""

    backbone: transformers.BertConfig | transformers.T5Config | transformers.RobertaConfig
    pooler: VodPoolerConfig

    def __init__(
        self,
        backbone: None | dict | transformers.BertConfig | transformers.T5Config | transformers.RobertaConfig = None,
        pooler: None | VodPoolerConfig = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if backbone is None:
            backbone = transformers.BertConfig()
        elif isinstance(backbone, dict):
            backbone = copy.deepcopy(backbone)
            name_or_path = backbone.pop("_name_or_path")
            backbone = transformers.AutoConfig.from_pretrained(name_or_path, **backbone)
        self.backbone = backbone  # type: ignore

        if pooler is None:
            pooler = VodPoolerConfig()
        elif isinstance(pooler, dict):
            pooler = VodPoolerConfig(**pooler)
        self.pooler = pooler

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict."""
        return {
            "backbone": self.backbone.to_dict(),
            "pooler": self.pooler.to_dict(),
        }
