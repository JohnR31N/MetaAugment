from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class PolicyNetwork(nn.Module):
    """MetaAugment policy MLP.

    The paper uses one hidden layer for image features and one for the
    transformation embedding, concatenates them, then emits a sigmoid weight.
    """

    hidden_size: int = 100

    @nn.compact
    def __call__(self, image_features: jnp.ndarray, transform_embedding: jnp.ndarray) -> jnp.ndarray:
        feature_branch = nn.relu(nn.Dense(self.hidden_size, name="feature_fc")(image_features))
        transform_branch = nn.relu(nn.Dense(self.hidden_size, name="transform_fc")(transform_embedding))
        x = jnp.concatenate([feature_branch, transform_branch], axis=-1)
        weights = nn.sigmoid(nn.Dense(1, name="weight_fc")(x))
        return jnp.squeeze(weights, axis=-1)
