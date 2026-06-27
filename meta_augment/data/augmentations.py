from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random


OP_NAMES: tuple[str, ...] = (
    "AutoContrast",
    "Equalize",
    "Rotate",
    "Posterize",
    "Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Identity",
)
NUM_OPS = len(OP_NAMES)
NO_MAGNITUDE_OPS = jnp.asarray([0, 1, 13], dtype=jnp.int32)


def initial_sampler_probs() -> jnp.ndarray:
    return jnp.full((NUM_OPS, NUM_OPS), 1.0 / float(NUM_OPS * NUM_OPS), dtype=jnp.float32)


def _uses_magnitude(op_id: jnp.ndarray) -> jnp.ndarray:
    return ~jnp.any(op_id[..., None] == NO_MAGNITUDE_OPS, axis=-1)


def transformation_embedding(
    op1: jnp.ndarray,
    op2: jnp.ndarray,
    magnitude1: jnp.ndarray,
    magnitude2: jnp.ndarray,
) -> jnp.ndarray:
    batch_size = op1.shape[0]
    rows = jnp.arange(batch_size)
    value1 = jnp.where(_uses_magnitude(op1), magnitude1 + 1.0, 11.0)
    value2 = jnp.where(_uses_magnitude(op2), magnitude2 + 1.0, 11.0)
    embedding = jnp.zeros((batch_size, NUM_OPS * 2), dtype=jnp.float32)
    embedding = embedding.at[rows, op1 * 2].set(value1)
    embedding = embedding.at[rows, op2 * 2 + 1].set(value2)
    return embedding


def sample_transformations(
    key: jnp.ndarray,
    sampler_probs: jnp.ndarray,
    batch_size: int,
    num_transforms_per_sample: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    total = batch_size * num_transforms_per_sample
    key_ids, key_mag1, key_mag2 = random.split(key, 3)
    logits = jnp.log(jnp.reshape(sampler_probs, (-1,)) + 1.0e-8)
    pair_ids = random.categorical(key_ids, logits, shape=(total,))
    op1 = pair_ids // NUM_OPS
    op2 = pair_ids % NUM_OPS
    magnitude1 = random.uniform(key_mag1, (total,), minval=0.0, maxval=10.0)
    magnitude2 = random.uniform(key_mag2, (total,), minval=0.0, maxval=10.0)
    return op1, op2, magnitude1, magnitude2, pair_ids


def _random_signed_magnitude(magnitude: jnp.ndarray, key: jnp.ndarray, max_value: float) -> jnp.ndarray:
    sign = jnp.where(random.bernoulli(key), 1.0, -1.0)
    return sign * magnitude / 10.0 * max_value


def _clip(image: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(image, 0.0, 1.0)


def _rgb_to_luma(image: jnp.ndarray) -> jnp.ndarray:
    coeffs = jnp.asarray([0.299, 0.587, 0.114], dtype=image.dtype)
    return jnp.sum(image * coeffs, axis=-1, keepdims=True)


def _blend(image1: jnp.ndarray, image2: jnp.ndarray, factor: jnp.ndarray) -> jnp.ndarray:
    return _clip(image1 + factor * (image2 - image1))


def _auto_contrast(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    del magnitude, key
    low = jnp.min(image, axis=(0, 1), keepdims=True)
    high = jnp.max(image, axis=(0, 1), keepdims=True)
    scale = jnp.where(high > low, 1.0 / (high - low), 1.0)
    return _clip((image - low) * scale)


def _equalize_channel(channel: jnp.ndarray) -> jnp.ndarray:
    values = jnp.clip(jnp.rint(channel * 255.0), 0, 255).astype(jnp.int32)
    hist = jnp.bincount(values.reshape((-1,)), length=256)
    cdf = jnp.cumsum(hist)
    nonzero = hist > 0
    cdf_min = jnp.min(jnp.where(nonzero, cdf, cdf[-1]))
    denom = jnp.maximum(cdf[-1] - cdf_min, 1)
    lut = jnp.clip(jnp.rint((cdf - cdf_min) * 255.0 / denom), 0, 255)
    return lut[values].astype(jnp.float32) / 255.0


def _equalize(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    del magnitude, key
    channels = jax.vmap(_equalize_channel, in_axes=2, out_axes=2)(image)
    return _clip(channels)


def _posterize(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    del key
    bits = jnp.rint(8.0 - magnitude / 10.0 * 4.0)
    shift = 8.0 - jnp.clip(bits, 4.0, 8.0)
    scale = jnp.power(2.0, shift)
    return jnp.floor(image * 255.0 / scale) * scale / 255.0


def _solarize(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    del key
    threshold = 1.0 - magnitude / 10.0
    return jnp.where(image < threshold, image, 1.0 - image)


def _color(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    factor = 1.0 + _random_signed_magnitude(magnitude, key, 0.9)
    gray = jnp.broadcast_to(_rgb_to_luma(image), image.shape)
    return _blend(gray, image, factor)


def _contrast(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    factor = 1.0 + _random_signed_magnitude(magnitude, key, 0.9)
    mean = jnp.mean(_rgb_to_luma(image), axis=(0, 1), keepdims=True)
    return _clip((image - mean) * factor + mean)


def _brightness(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    factor = 1.0 + _random_signed_magnitude(magnitude, key, 0.9)
    return _clip(image * factor)


def _blur(image: jnp.ndarray) -> jnp.ndarray:
    padded = jnp.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    total = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1] * 5.0
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return total / 13.0


def _sharpness(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    factor = 1.0 + _random_signed_magnitude(magnitude, key, 0.9)
    return _blend(_blur(image), image, factor)


def _affine(image: jnp.ndarray, matrix: jnp.ndarray, translate: jnp.ndarray) -> jnp.ndarray:
    height, width = image.shape[:2]
    ys, xs = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    center = jnp.asarray([(width - 1) / 2.0, (height - 1) / 2.0], dtype=jnp.float32)
    coords = jnp.stack([xs.astype(jnp.float32), ys.astype(jnp.float32)], axis=-1) - center
    source = coords @ matrix.T + center + translate
    src_x = source[..., 0]
    src_y = source[..., 1]

    x0 = jnp.floor(src_x).astype(jnp.int32)
    y0 = jnp.floor(src_y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    def gather(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = jnp.clip(x, 0, width - 1)
        y = jnp.clip(y, 0, height - 1)
        pixel = image[y, x]
        return jnp.where(in_bounds[..., None], pixel, 0.5)

    wa = (x1.astype(jnp.float32) - src_x) * (y1.astype(jnp.float32) - src_y)
    wb = (x1.astype(jnp.float32) - src_x) * (src_y - y0.astype(jnp.float32))
    wc = (src_x - x0.astype(jnp.float32)) * (y1.astype(jnp.float32) - src_y)
    wd = (src_x - x0.astype(jnp.float32)) * (src_y - y0.astype(jnp.float32))

    return _clip(
        gather(y0, x0) * wa[..., None]
        + gather(y1, x0) * wb[..., None]
        + gather(y0, x1) * wc[..., None]
        + gather(y1, x1) * wd[..., None]
    )


def _rotate(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    angle = -jnp.deg2rad(_random_signed_magnitude(magnitude, key, 30.0))
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    matrix = jnp.stack(
        [
            jnp.stack([cos_a, -sin_a]),
            jnp.stack([sin_a, cos_a]),
        ]
    ).astype(jnp.float32)
    return _affine(image, matrix, jnp.zeros((2,), dtype=jnp.float32))


def _shear_x(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    shear = _random_signed_magnitude(magnitude, key, 0.3)
    matrix = jnp.stack(
        [
            jnp.stack([jnp.asarray(1.0, dtype=jnp.float32), -shear]),
            jnp.stack([jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(1.0, dtype=jnp.float32)]),
        ]
    )
    return _affine(image, matrix, jnp.zeros((2,), dtype=jnp.float32))


def _shear_y(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    shear = _random_signed_magnitude(magnitude, key, 0.3)
    matrix = jnp.stack(
        [
            jnp.stack([jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)]),
            jnp.stack([-shear, jnp.asarray(1.0, dtype=jnp.float32)]),
        ]
    )
    return _affine(image, matrix, jnp.zeros((2,), dtype=jnp.float32))


def _translate_x(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jnp.ndarray,
    translate_const: float,
) -> jnp.ndarray:
    shift = -_random_signed_magnitude(magnitude, key, translate_const)
    return _affine(
        image,
        jnp.eye(2, dtype=jnp.float32),
        jnp.stack([shift, jnp.asarray(0.0, dtype=jnp.float32)]),
    )


def _translate_y(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jnp.ndarray,
    translate_const: float,
) -> jnp.ndarray:
    shift = -_random_signed_magnitude(magnitude, key, translate_const)
    return _affine(
        image,
        jnp.eye(2, dtype=jnp.float32),
        jnp.stack([jnp.asarray(0.0, dtype=jnp.float32), shift]),
    )


def _identity(image: jnp.ndarray, magnitude: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    del magnitude, key
    return image


def apply_op(
    image: jnp.ndarray,
    op_id: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jnp.ndarray,
    translate_const: float,
) -> jnp.ndarray:
    branches = (
        _auto_contrast,
        _equalize,
        _rotate,
        _posterize,
        _solarize,
        _color,
        _contrast,
        _brightness,
        _sharpness,
        _shear_x,
        _shear_y,
        partial(_translate_x, translate_const=translate_const),
        partial(_translate_y, translate_const=translate_const),
        _identity,
    )
    return lax.switch(op_id, branches, image, magnitude, key)


def random_flip_crop(images: jnp.ndarray, key: jnp.ndarray, padding: int = 4) -> jnp.ndarray:
    keys = random.split(key, images.shape[0])

    def transform(image: jnp.ndarray, image_key: jnp.ndarray) -> jnp.ndarray:
        y_key, x_key, flip_key = random.split(image_key, 3)
        padded = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)), constant_values=0.5)
        y = random.randint(y_key, (), 0, padding * 2 + 1)
        x = random.randint(x_key, (), 0, padding * 2 + 1)
        cropped = lax.dynamic_slice(padded, (y, x, 0), image.shape)
        flipped = jnp.flip(cropped, axis=1)
        return jnp.where(random.bernoulli(flip_key), flipped, cropped)

    return jax.vmap(transform)(images, keys)


def cutout(images: jnp.ndarray, key: jnp.ndarray, size: int) -> jnp.ndarray:
    if size <= 0:
        return images
    keys = random.split(key, images.shape[0])
    height, width = images.shape[1:3]
    half = size // 2
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")

    def apply(image: jnp.ndarray, image_key: jnp.ndarray) -> jnp.ndarray:
        y_key, x_key = random.split(image_key)
        y = random.randint(y_key, (), 0, height)
        x = random.randint(x_key, (), 0, width)
        keep = (jnp.abs(yy - y) > half) | (jnp.abs(xx - x) > half)
        return jnp.where(keep[..., None], image, 0.5)

    return jax.vmap(apply)(images, keys)


def apply_metaaugment(
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jnp.ndarray,
    sampler_probs: jnp.ndarray,
    *,
    num_transforms_per_sample: int,
    cutout_size: int,
    translate_const: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key_base, key_sample, key_op1, key_op2, key_cutout = random.split(key, 5)
    base_images = random_flip_crop(images, key_base)
    batch_size = images.shape[0]
    op1, op2, mag1, mag2, pair_ids = sample_transformations(
        key_sample, sampler_probs, batch_size, num_transforms_per_sample
    )
    repeated_images = jnp.repeat(base_images, num_transforms_per_sample, axis=0)
    repeated_labels = jnp.repeat(labels, num_transforms_per_sample, axis=0)
    keys1 = random.split(key_op1, repeated_images.shape[0])
    keys2 = random.split(key_op2, repeated_images.shape[0])
    augmented = jax.vmap(apply_op, in_axes=(0, 0, 0, 0, None))(
        repeated_images, op1, mag1, keys1, translate_const
    )
    augmented = jax.vmap(apply_op, in_axes=(0, 0, 0, 0, None))(
        augmented, op2, mag2, keys2, translate_const
    )
    augmented = cutout(augmented, key_cutout, cutout_size)
    embedding = transformation_embedding(op1, op2, mag1, mag2)
    return augmented, repeated_labels, embedding, pair_ids
