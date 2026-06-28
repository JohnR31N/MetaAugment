from __future__ import annotations

import flax.linen as nn

from meta_augment.config import ModelConfig
from meta_augment.networks.task_networks.backbones.preact_resnet import PreActResNet, PreActResNet18
from meta_augment.networks.task_networks.backbones.wide_resnet_28_10 import WideResNet


def create_task_model(
    config: ModelConfig,
    *,
    num_classes: int,
    axis_name: str | None,
) -> nn.Module:
    architecture = config.architecture.lower()
    if architecture in {"wide_resnet", "wrn", "wrn_28_10"}:
        return WideResNet(
            depth=config.depth,
            width=config.width,
            num_classes=num_classes,
            dropout_rate=config.dropout_rate,
            axis_name=axis_name,
        )
    if architecture in {"preact_resnet18", "preact-resnet18", "preact18", "preact_resnet_18"}:
        if config.depth not in {0, 18}:
            raise ValueError("preact_resnet18 expects model.depth to be 18 or omitted/defaulted to 18")
        return PreActResNet18(
            num_classes=num_classes,
            width=config.width,
            dropout_rate=config.dropout_rate,
            axis_name=axis_name,
        )
    if architecture in {"preact_resnet", "preact-resnet", "preact"}:
        return PreActResNet(
            depth=config.depth,
            width=config.width,
            num_classes=num_classes,
            dropout_rate=config.dropout_rate,
            axis_name=axis_name,
        )
    raise ValueError(f"Unsupported model architecture: {config.architecture}")
