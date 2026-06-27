from __future__ import annotations

import jax.numpy as jnp


CIFAR_STATS = {
    "cifar10": (
        jnp.asarray([0.4914, 0.4822, 0.4465], dtype=jnp.float32),
        jnp.asarray([0.2470, 0.2435, 0.2616], dtype=jnp.float32),
    ),
    "cifar100": (
        jnp.asarray([0.5071, 0.4867, 0.4408], dtype=jnp.float32),
        jnp.asarray([0.2675, 0.2565, 0.2761], dtype=jnp.float32),
    ),
    "synthetic": (
        jnp.asarray([0.5, 0.5, 0.5], dtype=jnp.float32),
        jnp.asarray([0.25, 0.25, 0.25], dtype=jnp.float32),
    ),
}


def normalize_images(images: jnp.ndarray, dataset: str) -> jnp.ndarray:
    mean, std = CIFAR_STATS[dataset.lower()]
    return (images - mean.reshape((1, 1, 1, 3))) / std.reshape((1, 1, 1, 3))
