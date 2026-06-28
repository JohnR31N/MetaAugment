from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "cifar10"
    data_dir: str = "data"
    val_size: int = 1000
    batch_size: int = 128
    eval_batch_size: int = 256
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "wide_resnet"
    depth: int = 28
    width: int = 10
    dropout_rate: float = 0.0


@dataclass(frozen=True)
class CompetitorConfig:
    name: str = "metaaugment"


@dataclass(frozen=True)
class AugmentConfig:
    cutout_size: int = 16
    epsilon: float = 0.1
    num_transforms_per_sample: int = 1
    sampler_update_epochs: int = 1
    sampler_history_epochs: int = 50
    translate_const: float = 10.0


@dataclass(frozen=True)
class OptimConfig:
    epochs: int = 600
    task_learning_rate: float = 0.1
    policy_learning_rate: float = 1.0e-3
    inner_learning_rate: float = 1.0e-2
    learn_inner_learning_rate: bool = False
    momentum: float = 0.9
    weight_decay: float = 5.0e-4
    label_smoothing: float = 0.0
    warmup_epochs: int = 0
    max_grad_norm: float = 0.0


@dataclass(frozen=True)
class SystemConfig:
    workdir: str = "runs/cifar10_wrn28_10_metaaugment"
    seed: int = 0
    use_pmap: bool = True
    init_distributed: bool = False
    log_every: int = 50
    eval_every_epochs: int = 1
    save_every_epochs: int = 10
    keep_checkpoints: int = 3


@dataclass(frozen=True)
class Config:
    competitor: CompetitorConfig = CompetitorConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    augment: AugmentConfig = AugmentConfig()
    optim: OptimConfig = OptimConfig()
    system: SystemConfig = SystemConfig()


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    updates = {}
    field_map = {field.name: field for field in fields(instance)}
    for key, value in values.items():
        if key not in field_map:
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config section {key!r} expects a mapping")
            updates[key] = _merge_dataclass(current, value)
        else:
            updates[key] = value
    return replace(instance, **updates)


def load_config(path: str | Path | None) -> Config:
    config = Config()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return _merge_dataclass(config, values)


def override_config(config: Config, dotted_key: str, value: Any) -> Config:
    parts = dotted_key.split(".")
    if len(parts) != 2:
        raise ValueError("Overrides must use section.field syntax")
    section, key = parts
    return _merge_dataclass(config, {section: {key: value}})


def to_dict(config: Config) -> dict[str, Any]:
    return asdict(config)
