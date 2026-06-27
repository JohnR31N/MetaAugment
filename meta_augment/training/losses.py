from __future__ import annotations

import jax
import jax.numpy as jnp


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, num_classes)
    if label_smoothing > 0.0:
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / num_classes
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot * log_probs, axis=-1)


def normalized_policy_weights(raw_weights: jnp.ndarray) -> jnp.ndarray:
    return raw_weights / (jnp.sum(raw_weights) + 1.0e-8)


def weighted_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    raw_weights: jnp.ndarray,
    *,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    per_example = cross_entropy_loss(
        logits,
        labels,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
    )
    weights = normalized_policy_weights(raw_weights)
    return jnp.sum(weights * per_example), per_example
