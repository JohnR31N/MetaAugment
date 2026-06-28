from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class PreActBasicBlock(nn.Module):
    channels: int
    stride: int
    dropout_rate: float
    axis_name: str | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        y = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1.0e-5,
            axis_name=self.axis_name,
            name="bn1",
        )(x)
        y = nn.relu(y)
        if x.shape[-1] != self.channels or self.stride != 1:
            shortcut = nn.Conv(
                self.channels,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                name="shortcut",
            )(y)
        else:
            shortcut = x
        y = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(y)
        y = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1.0e-5,
            axis_name=self.axis_name,
            name="bn2",
        )(y)
        y = nn.relu(y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(y)
        return shortcut + y


class PreActResNet(nn.Module):
    """CIFAR-style pre-activation ResNet v2.

    Uses the 6n+2 CIFAR depth convention: 20, 32, 44, 56, 110, ...
    """

    depth: int
    width: int
    num_classes: int
    dropout_rate: float = 0.0
    axis_name: str | None = None

    def setup(self) -> None:
        if (self.depth - 2) % 6 != 0:
            raise ValueError("PreActResNet depth must satisfy depth = 6n + 2")
        self.blocks_per_group = (self.depth - 2) // 6
        self.channels: Sequence[int] = (
            16 * self.width,
            32 * self.width,
            64 * self.width,
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool,
        return_features: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Conv(
            16 * self.width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="stem",
        )(x)

        for group_id, (channels, stride) in enumerate(
            zip(self.channels, (1, 2, 2), strict=True)
        ):
            for block_id in range(self.blocks_per_group):
                x = PreActBasicBlock(
                    channels=channels,
                    stride=stride if block_id == 0 else 1,
                    dropout_rate=self.dropout_rate,
                    axis_name=self.axis_name,
                    name=f"group{group_id + 1}_block{block_id + 1}",
                )(x, train=train)

        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1.0e-5,
            axis_name=self.axis_name,
            name="final_bn",
        )(x)
        x = nn.relu(x)
        features = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes, name="classifier")(features)
        if return_features:
            return logits, features
        return logits
