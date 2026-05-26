#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append("./python")
sys.path.append(".")

import needle as ndl
import needle.nn as nn
from needle.data import DataLoader, FashionMNISTDataset, IrisDataset, MNISTDataset

from apps.residual_mlp import ResidualMLP


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


DATASET_INFO = {
    "iris": {"input_dim": 4, "num_classes": 3},
    "mnist": {"input_dim": 784, "num_classes": 10},
    "fashion-mnist": {"input_dim": 784, "num_classes": 10},
}


def build_datasets(name):
    if name == "iris":
        base = PROCESSED_DIR / "iris"
        return IrisDataset(str(base / "train.npz")), IrisDataset(str(base / "test.npz"))

    if name == "mnist":
        base = RAW_DIR / "mnist"
        return (
            MNISTDataset(
                str(base / "train-images-idx3-ubyte.gz"),
                str(base / "train-labels-idx1-ubyte.gz"),
            ),
            MNISTDataset(
                str(base / "t10k-images-idx3-ubyte.gz"),
                str(base / "t10k-labels-idx1-ubyte.gz"),
            ),
        )

    if name == "fashion-mnist":
        base = RAW_DIR / "fashion-mnist"
        return (
            FashionMNISTDataset(
                str(base / "train-images-idx3-ubyte.gz"),
                str(base / "train-labels-idx1-ubyte.gz"),
            ),
            FashionMNISTDataset(
                str(base / "t10k-images-idx3-ubyte.gz"),
                str(base / "t10k-labels-idx1-ubyte.gz"),
            ),
        )

    raise ValueError(f"unknown dataset: {name}")


def build_dataloaders(name, batch_size):
    train_dataset, test_dataset = build_datasets(name)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def accuracy(logits, y):
    pred = logits.numpy().argmax(axis=1)
    return float((pred == y).mean())


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"param_{i}": p.numpy() for i, p in enumerate(model.parameters())}
    np.savez_compressed(path, **arrays)


def load_model(model, path):
    data = np.load(path)
    params = model.parameters()
    for i, param in enumerate(params):
        key = f"param_{i}"
        if key not in data:
            raise ValueError(f"missing {key} in checkpoint {path}")
        arr = data[key].astype(param.numpy().dtype, copy=False)
        if arr.shape != param.shape:
            raise ValueError(f"{key} shape mismatch: checkpoint {arr.shape}, model {param.shape}")
        param.data = ndl.Tensor(arr, device=param.device, dtype=param.dtype)


def run_epoch(model, loss_fn, optimizer, dataloader, train, clip_grad=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x_tensor, y_tensor in dataloader:
        logits = model(x_tensor)
        loss = loss_fn(logits, y_tensor)
        labels = y_tensor.numpy().astype("int32")
        batch_size_actual = labels.shape[0]

        total_loss += float(loss.numpy()) * batch_size_actual
        total_correct += int((logits.numpy().argmax(axis=1) == labels).sum())
        total_count += batch_size_actual

        if train:
            loss.backward()
            if clip_grad is not None and hasattr(optimizer, "clip_grad_norm"):
                optimizer.clip_grad_norm(clip_grad)
            optimizer.step()
            optimizer.reset_grad()

    return total_loss / total_count, total_correct / total_count


def build_optimizer(name, params, lr, weight_decay):
    if name == "sgd":
        return ndl.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return ndl.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResidualMLP on local processed datasets.")
    parser.add_argument("--dataset", choices=tuple(DATASET_INFO.keys()), default="iris")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="checkpoints/residual_mlp.npz")
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    train_loader, test_loader = build_dataloaders(args.dataset, args.batch_size)
    info = DATASET_INFO[args.dataset]
    model = ResidualMLP(
        input_dim=info["input_dim"],
        hidden_dim=args.hidden_dim,
        num_classes=info["num_classes"],
        num_blocks=args.num_blocks,
    )
    loss_fn = nn.SoftmaxLoss()

    if args.load_path is not None:
        load_model(model, args.load_path)
        print(f"loaded checkpoint: {args.load_path}")

    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    if args.eval_only:
        test_loss, test_acc = run_epoch(
            model, loss_fn, optimizer, test_loader, train=False
        )
        print(f"eval test_loss={test_loss:.6f} test_acc={test_acc:.4f}")
        return

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            train=True,
            clip_grad=args.clip_grad,
        )
        test_loss, test_acc = run_epoch(
            model, loss_fn, optimizer, test_loader, train=False
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.4f}"
        )

    save_model(model, args.save_path)
    print(f"saved checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
