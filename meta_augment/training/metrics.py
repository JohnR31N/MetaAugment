from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def accuracy_sum(logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    return jnp.sum(correct * mask)


def topk_accuracy(logits: jnp.ndarray, labels: jnp.ndarray, k: int = 5) -> jnp.ndarray:
    k = min(k, logits.shape[-1])
    _, topk = lax.top_k(logits, k)
    correct = jnp.any(topk == labels[:, None], axis=-1)
    return jnp.mean(correct.astype(jnp.float32))


def topk_accuracy_sum(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
    k: int = 5,
) -> jnp.ndarray:
    k = min(k, logits.shape[-1])
    _, topk = lax.top_k(logits, k)
    correct = jnp.any(topk == labels[:, None], axis=-1).astype(jnp.float32)
    return jnp.sum(correct * mask)
