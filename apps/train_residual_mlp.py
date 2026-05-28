#!/usr/bin/env python3
import argparse
import csv
import json
import os
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


def progress_bar(iterable, total, desc, enabled):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


def slug_float(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")


def scheduler_slug(args):
    if args.scheduler == "none":
        return "none"
    if args.scheduler == "step":
        return f"step_s{args.step_size}_g{slug_float(args.gamma)}"
    if args.scheduler == "warmup":
        return (
            f"warmup_w{args.warmup_steps}_start{slug_float(args.warmup_start_lr)}"
        )
    if args.scheduler == "cosine":
        return (
            f"cosine_c{args.cosine_first_cycle_steps}_min"
            f"{slug_float(args.cosine_min_lr)}_mult{slug_float(args.cosine_cycle_mult)}"
        )
    raise ValueError(f"unknown scheduler: {args.scheduler}")


def default_experiment_name(args):
    return (
        f"ResidualMLP__{args.dataset}__{args.optimizer}__{scheduler_slug(args)}__"
        f"hd{args.hidden_dim}_blk{args.num_blocks}__"
        f"lr{slug_float(args.lr)}"
    )


def prepare_experiment_dir(args):
    name = args.experiment_name or default_experiment_name(args)
    output_dir = Path(args.output_root) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def serializable_config(args, output_dir, save_path):
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["resolved_save_path"] = str(save_path)
    config["experiment_name"] = output_dir.name
    return config


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_metrics_csv(path, metrics):
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def write_summary(output_dir, config, metrics):
    summary = {"config": config, "epochs_completed": len(metrics)}
    if metrics:
        best = max(metrics, key=lambda row: row["test_acc"])
        summary.update(
            {
                "final": metrics[-1],
                "best_test_acc": best["test_acc"],
                "best_test_acc_epoch": best["epoch"],
                "best_test_loss_at_best_acc": best["test_loss"],
            }
        )
    write_json(output_dir / "summary.json", summary)


def plot_metrics(output_dir, metrics):
    if not metrics:
        return

    cache_dir = output_dir / "matplotlib_cache"
    cache_dir.mkdir(exist_ok=True)
    old_mpl_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in metrics]
        train_loss = [row["train_loss"] for row in metrics]
        test_loss = [row["test_loss"] for row in metrics]
        train_acc = [row["train_acc"] for row in metrics]
        test_acc = [row["test_acc"] for row in metrics]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, label="train loss")
        axes[0].plot(epochs, test_loss, label="test loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, train_acc, label="train acc")
        axes[1].plot(epochs, test_acc, label="test acc")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "curves.png", dpi=160)
        plt.close(fig)
    except Exception as err:
        with open(output_dir / "plot_error.txt", "w", encoding="utf-8") as f:
            f.write(f"failed to create curves.png: {err}\n")
    finally:
        if old_mpl_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl_config


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


def run_epoch(
    model,
    loss_fn,
    optimizer,
    dataloader,
    train,
    clip_grad=None,
    epoch=None,
    progress=True,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    iterator = iter(dataloader)
    total_batches = len(dataloader.ordering)
    stage = "train" if train else "eval"
    desc = stage if epoch is None else f"{stage} epoch {epoch:03d}"
    iterator = progress_bar(iterator, total_batches, desc, progress)

    for x_tensor, y_tensor in iterator:
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

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{total_loss / total_count:.4f}",
                acc=f"{total_correct / total_count:.4f}",
            )

    return total_loss / total_count, total_correct / total_count


def build_optimizer(name, params, lr, weight_decay):
    if name == "sgd":
        return ndl.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return ndl.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return ndl.optim.StepDecay(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    if args.scheduler == "warmup":
        return ndl.optim.LinearWarmUp(
            optimizer,
            warmup_steps=args.warmup_steps,
            start_lr=args.warmup_start_lr,
        )
    if args.scheduler == "cosine":
        return ndl.optim.CosineDecayWithWarmRestarts(
            optimizer,
            first_cycle_steps=args.cosine_first_cycle_steps,
            min_lr=args.cosine_min_lr,
            cycle_mult=args.cosine_cycle_mult,
        )
    raise ValueError(f"unknown scheduler: {args.scheduler}")


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
    parser.add_argument("--scheduler", choices=("none", "step", "warmup", "cosine"), default="none")
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--warmup-start-lr", type=float, default=0.0)
    parser.add_argument("--cosine-first-cycle-steps", type=int, default=10)
    parser.add_argument("--cosine-min-lr", type=float, default=0.0)
    parser.add_argument("--cosine-cycle-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = prepare_experiment_dir(args)
    save_path = Path(args.save_path) if args.save_path is not None else output_dir / "model.npz"
    config = serializable_config(args, output_dir, save_path)
    write_json(output_dir / "config.json", config)

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
    scheduler = build_scheduler(args, optimizer)

    if args.eval_only:
        test_loss, test_acc = run_epoch(
            model,
            loss_fn,
            optimizer,
            test_loader,
            train=False,
            progress=not args.no_progress,
        )
        print(f"eval test_loss={test_loss:.6f} test_acc={test_acc:.4f}")
        write_json(
            output_dir / "eval_summary.json",
            {"config": config, "test_loss": test_loss, "test_acc": test_acc},
        )
        return

    metrics = []
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.step() if scheduler is not None else optimizer.lr
        train_loss, train_acc = run_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            train=True,
            clip_grad=args.clip_grad,
            epoch=epoch,
            progress=not args.no_progress,
        )
        test_loss, test_acc = run_epoch(
            model,
            loss_fn,
            optimizer,
            test_loader,
            train=False,
            epoch=epoch,
            progress=not args.no_progress,
        )
        metrics.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        write_metrics_csv(output_dir / "metrics.csv", metrics)
        write_summary(output_dir, config, metrics)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.4f} "
            f"lr={lr:.6g}"
        )

    plot_metrics(output_dir, metrics)
    if not args.no_save:
        save_model(model, save_path)
        print(f"saved checkpoint: {save_path}")
    print(f"saved logs: {output_dir}")


if __name__ == "__main__":
    main()
