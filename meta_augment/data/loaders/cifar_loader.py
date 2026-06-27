from __future__ import annotations

import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


@dataclass(frozen=True)
class CifarArrays:
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    num_classes: int

    @property
    def train_size(self) -> int:
        return int(self.train_labels.shape[0])

    @property
    def val_size(self) -> int:
        return int(self.val_labels.shape[0])

    @property
    def test_size(self) -> int:
        return int(self.test_labels.shape[0])


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    tmp = target.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)


def _extract(archive: Path, root: Path, extracted_dir: str) -> Path:
    output = root / extracted_dir
    if output.exists():
        return output
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return output


def _read_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _reshape_images(flat: np.ndarray) -> np.ndarray:
    images = flat.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return images.astype(np.float32) / 255.0


def _load_cifar10(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = root / "cifar-10-python.tar.gz"
    _download(CIFAR10_URL, archive)
    directory = _extract(archive, root, "cifar-10-batches-py")

    train_images = []
    train_labels = []
    for batch_id in range(1, 6):
        batch = _read_pickle(directory / f"data_batch_{batch_id}")
        train_images.append(batch["data"])
        train_labels.extend(batch["labels"])

    test = _read_pickle(directory / "test_batch")
    return (
        _reshape_images(np.concatenate(train_images, axis=0)),
        np.asarray(train_labels, dtype=np.int32),
        _reshape_images(test["data"]),
        np.asarray(test["labels"], dtype=np.int32),
    )


def _load_cifar100(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = root / "cifar-100-python.tar.gz"
    _download(CIFAR100_URL, archive)
    directory = _extract(archive, root, "cifar-100-python")

    train = _read_pickle(directory / "train")
    test = _read_pickle(directory / "test")
    return (
        _reshape_images(train["data"]),
        np.asarray(train["fine_labels"], dtype=np.int32),
        _reshape_images(test["data"]),
        np.asarray(test["fine_labels"], dtype=np.int32),
    )


def _load_synthetic(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    num_classes = 10
    train_images = rng.random((256, 32, 32, 3), dtype=np.float32)
    train_labels = rng.integers(0, num_classes, size=(256,), dtype=np.int32)
    test_images = rng.random((128, 32, 32, 3), dtype=np.float32)
    test_labels = rng.integers(0, num_classes, size=(128,), dtype=np.int32)
    return train_images, train_labels, test_images, test_labels, num_classes


def load_cifar_dataset(
    dataset: str,
    data_dir: str | Path,
    val_size: int = 1000,
    seed: int = 0,
) -> CifarArrays:
    """Load CIFAR arrays and hold out validation images from the train split."""
    name = dataset.lower()
    root = Path(data_dir)
    if name == "cifar10":
        train_images, train_labels, test_images, test_labels = _load_cifar10(root)
        num_classes = 10
    elif name == "cifar100":
        train_images, train_labels, test_images, test_labels = _load_cifar100(root)
        num_classes = 100
    elif name == "synthetic":
        train_images, train_labels, test_images, test_labels, num_classes = _load_synthetic(seed)
        val_size = min(val_size, 64)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not 0 < val_size < train_labels.shape[0]:
        raise ValueError("val_size must be between 1 and the number of train examples - 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(train_labels.shape[0])
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return CifarArrays(
        train_images=train_images[train_indices],
        train_labels=train_labels[train_indices],
        val_images=train_images[val_indices],
        val_labels=train_labels[val_indices],
        test_images=test_images,
        test_labels=test_labels,
        num_classes=num_classes,
    )


def iter_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    *,
    shuffle: bool,
    drop_remainder: bool,
) -> Iterator[dict[str, np.ndarray]]:
    indices = np.arange(labels.shape[0])
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, labels.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.shape[0] < batch_size and drop_remainder:
            continue
        mask = np.ones((batch_indices.shape[0],), dtype=np.float32)
        if batch_indices.shape[0] < batch_size:
            pad = batch_size - batch_indices.shape[0]
            batch_indices = np.pad(batch_indices, (0, pad), mode="edge")
            mask = np.pad(mask, (0, pad), mode="constant")
        yield {
            "image": images[batch_indices],
            "label": labels[batch_indices],
            "mask": mask,
        }


class CyclingBatcher:
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        *,
        shuffle: bool = True,
    ) -> None:
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.rng = rng
        self.shuffle = shuffle
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None

    def next(self) -> dict[str, np.ndarray]:
        if self._iterator is None:
            self._iterator = iter_batches(
                self.images,
                self.labels,
                self.batch_size,
                self.rng,
                shuffle=self.shuffle,
                drop_remainder=True,
            )
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            return self.next()


def shard_batch(batch: dict[str, np.ndarray], device_count: int) -> dict[str, np.ndarray]:
    if device_count == 1:
        return {key: value.reshape((1,) + value.shape) for key, value in batch.items()}
    size = batch["label"].shape[0]
    if size % device_count != 0:
        raise ValueError(f"Batch size {size} must be divisible by {device_count} devices")
    per_device = size // device_count
    return {
        key: value.reshape((device_count, per_device) + value.shape[1:])
        for key, value in batch.items()
    }
