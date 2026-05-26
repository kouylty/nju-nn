#!/usr/bin/env python3
import argparse
import gzip
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"


IRIS_LABELS = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


def parse_idx_images(path: Path, flatten: bool) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file: magic={magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    expected = num_images * rows * cols
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} pixels, expected {expected}")

    shape = (num_images, rows * cols) if flatten else (num_images, rows, cols)
    images = data.reshape(shape)
    return images.astype(np.float32) / 255.0


def parse_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file: magic={magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    if labels.size != num_labels:
        raise ValueError(f"{path} has {labels.size} labels, expected {num_labels}")
    return labels.astype(np.int8)


def save_npz(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    print(f"saved {path.relative_to(ROOT)} X={X.shape} {X.dtype} y={y.shape} {y.dtype}")


def prepare_idx_dataset(name: str, flatten: bool) -> None:
    raw = RAW_DIR / name
    out = PROCESSED_DIR / name

    train_X = parse_idx_images(raw / "train-images-idx3-ubyte.gz", flatten=flatten)
    train_y = parse_idx_labels(raw / "train-labels-idx1-ubyte.gz")
    test_X = parse_idx_images(raw / "t10k-images-idx3-ubyte.gz", flatten=flatten)
    test_y = parse_idx_labels(raw / "t10k-labels-idx1-ubyte.gz")

    save_npz(out / "train.npz", train_X, train_y)
    save_npz(out / "test.npz", test_X, test_y)


def load_iris_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            rows.append([float(v) for v in parts[:4]])
            labels.append(IRIS_LABELS[parts[4]])

    if not rows:
        raise ValueError(f"no Iris rows found in {path}")
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int8)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        test_size = int(round(len(indices) * test_ratio))
        test_indices.extend(indices[:test_size])
        train_indices.extend(indices[test_size:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def prepare_iris(test_ratio: float, seed: int) -> None:
    raw = RAW_DIR / "iris"
    out = PROCESSED_DIR / "iris"
    data_file = raw / "iris.data"
    if not data_file.exists():
        data_file = raw / "bezdekIris.data"

    X, y = load_iris_raw(data_file)
    train_X, train_y, test_X, test_y = stratified_split(X, y, test_ratio, seed)

    save_npz(out / "train.npz", train_X, train_y)
    save_npz(out / "test.npz", test_X, test_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local datasets for experiments.")
    parser.add_argument(
        "--dataset",
        choices=("all", "iris", "mnist", "fashion-mnist"),
        default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--image-shape",
        choices=("flat", "image"),
        default="flat",
        help="Store MNIST/Fashion-MNIST as (N, 784) or (N, 28, 28).",
    )
    parser.add_argument("--iris-test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    flatten = args.image_shape == "flat"
    if args.dataset in ("all", "iris"):
        prepare_iris(args.iris_test_ratio, args.seed)
    if args.dataset in ("all", "mnist"):
        prepare_idx_dataset("mnist", flatten=flatten)
    if args.dataset in ("all", "fashion-mnist"):
        prepare_idx_dataset("fashion-mnist", flatten=flatten)


if __name__ == "__main__":
    main()
