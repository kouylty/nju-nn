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
from needle.data import (
    CIFAR10Dataset,
    DataLoader,
    Dataset,
    RandomCrop,
    RandomFlipHorizontal,
)

from apps.models import ResNet9


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIFAR_DIR = ROOT / "data" / "raw" / "cifar-10-batches-py"


class SubsetDataset(Dataset):
    def __init__(self, dataset, max_samples):
        super().__init__(transforms=None)
        self.dataset = dataset
        self.indices = np.arange(min(max_samples, len(dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


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
        return f"warmup_w{args.warmup_steps}_start{slug_float(args.warmup_start_lr)}"
    if args.scheduler == "cosine":
        return (
            f"cosine_c{args.cosine_first_cycle_steps}_min"
            f"{slug_float(args.cosine_min_lr)}_mult{slug_float(args.cosine_cycle_mult)}"
        )
    raise ValueError(f"unknown scheduler: {args.scheduler}")


def default_experiment_name(args):
    aug = "aug" if args.augment else "noaug"
    return (
        f"ResNet9__cifar10__{args.optimizer}__{scheduler_slug(args)}__"
        f"{aug}__lr{slug_float(args.lr)}"
    )


def prepare_experiment_dir(args):
    name = args.experiment_name or default_experiment_name(args)
    output_dir = Path(args.output_root) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_device(device_name):
    if device_name == "cpu_numpy":
        return ndl.cpu_numpy()
    if device_name == "cpu":
        return ndl.cpu()
    if device_name == "cuda":
        device = ndl.cuda()
        if not device.enabled():
            raise RuntimeError("Needle CUDA backend is not available")
        _probe_device(device)
        return device
    if device_name == "auto":
        cuda_device = ndl.cuda()
        if cuda_device.enabled():
            try:
                _probe_device(cuda_device)
                print("using device: cuda")
                return cuda_device
            except Exception as err:
                print(f"cuda probe failed, falling back to cpu_numpy: {err}")
        print("using device: cpu_numpy")
        return ndl.cpu_numpy()
    raise ValueError(f"unknown device: {device_name}")


def _probe_device(device):
    x = ndl.Tensor(np.zeros((1,), dtype=np.float32), device=device, requires_grad=False)
    _ = (x + 1).numpy()


def move_batch_to_device(batch, device):
    x_tensor, y_tensor = batch
    return (
        ndl.Tensor(x_tensor, device=device, requires_grad=False),
        ndl.Tensor(y_tensor, device=device, requires_grad=False),
    )


def serializable_config(args, output_dir, save_path, best_save_path, device):
    config = vars(args).copy()
    config["cifar_dir"] = str(Path(args.cifar_dir).resolve())
    config["output_dir"] = str(output_dir)
    config["resolved_save_path"] = str(save_path)
    config["resolved_best_save_path"] = str(best_save_path)
    config["resolved_device"] = str(device)
    config["experiment_name"] = output_dir.name
    return config


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_metrics_csv(path, metrics):
    fieldnames = [
        "epoch",
        "train_loss",
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
        test_acc = [row["test_acc"] for row in metrics]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, label="train loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

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


def build_datasets(args):
    train_transforms = None
    if args.augment:
        train_transforms = [
            RandomCrop(padding=args.crop_padding),
            RandomFlipHorizontal(p=args.flip_prob),
        ]

    train_dataset = CIFAR10Dataset(
        args.cifar_dir,
        train=True,
        transforms=train_transforms,
    )
    test_dataset = CIFAR10Dataset(args.cifar_dir, train=False)

    if args.max_train_samples is not None:
        train_dataset = SubsetDataset(train_dataset, args.max_train_samples)
    if args.max_test_samples is not None:
        test_dataset = SubsetDataset(test_dataset, args.max_test_samples)

    return train_dataset, test_dataset


def build_dataloaders(args):
    train_dataset, test_dataset = build_datasets(args)
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
    )


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


def run_epoch(
    model,
    loss_fn,
    optimizer,
    dataloader,
    train,
    clip_grad=None,
    epoch=None,
    progress=True,
    device=None,
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

    for batch in iterator:
        if device is not None:
            x_tensor, y_tensor = move_batch_to_device(batch, device)
        else:
            x_tensor, y_tensor = batch

        logits = model(x_tensor)

        if train:
            loss = loss_fn(logits, y_tensor)
            batch_size_actual = y_tensor.shape[0]
            loss_value = float(loss.numpy())
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite train loss at epoch={epoch}: {loss_value}"
                )
            total_loss += loss_value * batch_size_actual
            total_count += batch_size_actual

            loss.backward()
            if clip_grad is not None and hasattr(optimizer, "clip_grad_norm"):
                optimizer.clip_grad_norm(clip_grad)
            optimizer.step()
            optimizer.reset_grad()
        else:
            labels = y_tensor.numpy().astype("int32")
            batch_size_actual = labels.shape[0]
            total_correct += int((logits.numpy().argmax(axis=1) == labels).sum())
            total_count += batch_size_actual

        if hasattr(iterator, "set_postfix"):
            if train:
                iterator.set_postfix(loss=f"{total_loss / total_count:.4f}")
            else:
                iterator.set_postfix(acc=f"{total_correct / total_count:.4f}")

    avg_loss = total_loss / total_count if train else None
    avg_acc = total_correct / total_count if not train else None
    return avg_loss, avg_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet9 on local CIFAR-10.")
    parser.add_argument("--cifar-dir", default=str(DEFAULT_CIFAR_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--scheduler", choices=("none", "step", "warmup", "cosine"), default="none")
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--warmup-start-lr", type=float, default=0.0)
    parser.add_argument("--cosine-first-cycle-steps", type=int, default=10)
    parser.add_argument("--cosine-min-lr", type=float, default=0.0)
    parser.add_argument("--cosine-cycle-mult", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "cpu_numpy"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--best-save-path", default=None)
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
    best_save_path = (
        Path(args.best_save_path)
        if args.best_save_path is not None
        else output_dir / "best_model.npz"
    )
    device = resolve_device(args.device)
    config = serializable_config(args, output_dir, save_path, best_save_path, device)
    write_json(output_dir / "config.json", config)

    train_loader, test_loader = build_dataloaders(args)
    model = ResNet9(device=device)
    loss_fn = nn.SoftmaxLoss()

    if args.load_path is not None:
        load_model(model, args.load_path)
        print(f"loaded checkpoint: {args.load_path}")

    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = build_scheduler(args, optimizer)

    if args.eval_only:
        _, test_acc = run_epoch(
            model,
            loss_fn,
            optimizer,
            test_loader,
            train=False,
            progress=not args.no_progress,
            device=device,
        )
        print(f"eval test_acc={test_acc:.4f}")
        write_json(
            output_dir / "eval_summary.json",
            {"config": config, "test_acc": test_acc},
        )
        return

    metrics = []
    best_test_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.step() if scheduler is not None else optimizer.lr
        train_loss, _ = run_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            train=True,
            clip_grad=args.clip_grad,
            epoch=epoch,
            progress=not args.no_progress,
            device=device,
        )
        _, test_acc = run_epoch(
            model,
            loss_fn,
            optimizer,
            test_loader,
            train=False,
            epoch=epoch,
            progress=not args.no_progress,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_acc": test_acc,
        }
        metrics.append(row)
        write_metrics_csv(output_dir / "metrics.csv", metrics)
        write_summary(output_dir, config, metrics)

        if not args.no_save and test_acc > best_test_acc:
            best_test_acc = test_acc
            save_model(model, best_save_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"test_acc={test_acc:.4f}"
        )

    plot_metrics(output_dir, metrics)
    if not args.no_save:
        save_model(model, save_path)
        print(f"saved final checkpoint: {save_path}")
        print(f"saved best checkpoint: {best_save_path}")
    print(f"saved logs: {output_dir}")


if __name__ == "__main__":
    main()
