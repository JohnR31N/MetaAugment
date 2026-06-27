from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class LinearHead(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.num_classes, name="classifier")(features)
