from __future__ import annotations

from typing import Any

import optax
from flax.traverse_util import path_aware_map


def cosine_schedule(
    *,
    base_learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int = 0,
) -> optax.Schedule:
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(0, steps_per_epoch * warmup_epochs)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0 if warmup_steps > 0 else base_learning_rate,
        peak_value=base_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )


def _decay_mask(params: Any) -> Any:
    def should_decay(path: tuple[str, ...], _: Any) -> bool:
        name = path[-1]
        if name in {"bias", "scale", "inner_log_lr"}:
            return False
        if "batch_stats" in path:
            return False
        return True

    return path_aware_map(should_decay, params)


def sgd_tx(
    *,
    learning_rate: float | optax.Schedule,
    momentum: float,
    weight_decay: float,
    max_grad_norm: float = 0.0,
) -> optax.GradientTransformation:
    transforms = []
    if max_grad_norm and max_grad_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    if weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(weight_decay, mask=_decay_mask))
    transforms.append(optax.sgd(learning_rate, momentum=momentum, nesterov=True))
    return optax.chain(*transforms)
