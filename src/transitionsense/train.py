"""Train a supervised baseline on precomputed HAPT windows.

Example:
    uv run python -m transitionsense.train --data-dir data/windows --model resnet \
        --output-dir results/resnet --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transitionsense.data import N_CHANNELS, WINDOW
from transitionsense.labels import N_CLASSES
from transitionsense.metrics import evaluate_predictions, subject_overlap_report, write_metrics
from transitionsense.models import build_model, parameter_count


def _load_split(data_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load one split's windows as tensors plus subject ids."""
    path = data_dir / f"windows_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run scripts/prepare_windows.py first")
    data = np.load(path)
    x = torch.from_numpy(data["X"].astype(np.float32))
    y = torch.from_numpy(data["y"].astype(np.int64))
    return x, y, data["subject_id"]


def _class_weights(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor | None:
    """Return balanced cross-entropy weights, clamped to keep rare classes from dominating."""
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights.clamp(max=20.0).to(device)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return argmax predictions for every batch in ``loader``."""
    model.eval()
    preds: list[np.ndarray] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def _evaluate_split(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    subjects: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, np.ndarray]:
    """Score a split and return ``(metrics_payload, predictions)``."""
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    y_pred = _predict(model, loader, device)
    payload = evaluate_predictions(y.numpy(), y_pred)
    return payload, y_pred


def train(args: argparse.Namespace) -> dict:
    """Run the training loop and write checkpoints, history, and test metrics.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The test metrics payload.
    """
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, subj_train = _load_split(data_dir, "train")
    x_val, y_val, subj_val = _load_split(data_dir, "validation")
    x_test, y_test, subj_test = _load_split(data_dir, "test")
    if args.limit:
        x_train, y_train, subj_train = x_train[: args.limit], y_train[: args.limit], subj_train[: args.limit]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.model).to(device)
    n_params = parameter_count(model)

    weights = None
    if args.class_weight == "balanced":
        weights = _class_weights(y_train, N_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    subject_sets = {
        "train": {int(s) for s in np.unique(subj_train)},
        "validation": {int(s) for s in np.unique(subj_val)},
        "test": {int(s) for s in np.unique(subj_test)},
    }
    overlap = subject_overlap_report(subject_sets)
    if not overlap["disjoint"]:
        raise RuntimeError(f"subject leakage across splits: {overlap['overlaps']}")

    history: list[dict] = []
    best_macro_f1 = -1.0
    best_state: dict | None = None
    best_epoch = -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * x.size(0)
        scheduler.step()

        val_pred = _predict(model, val_loader, device)
        val_metrics = evaluate_predictions(y_val.numpy(), val_pred)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader.dataset)),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_transition_macro_f1": val_metrics["transition_macro_f1"],
        }
        history.append(record)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} loss {record['train_loss']:.4f} "
            f"val macro-F1 {val_metrics['macro_f1']:.4f} "
            f"val transition-F1 {val_metrics['transition_macro_f1']:.4f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint = {
        "model": args.model,
        "config": {"in_channels": N_CHANNELS, "num_classes": N_CLASSES, "window": WINDOW},
        "state_dict": best_state,
        "epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "seed": args.seed,
    }
    torch.save(checkpoint, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics, test_pred = _evaluate_split(model, x_test, y_test, subj_test, device, args.batch_size)
    test_metrics.update(
        {
            "model": args.model,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "split": "test",
            "split_subjects": subject_overlap_report(subject_sets),
            "parameters": n_params,
            "train_seconds": time.time() - t0,
        }
    )
    write_metrics(out_dir / "test_metrics.json", test_metrics)
    np.savez(
        out_dir / "test_predictions.npz",
        y_true=y_test.numpy(),
        y_pred=test_pred,
        subject_id=subj_test,
        experiment_id=np.load(data_dir / "windows_test.npz")["experiment_id"],
        window_start=np.load(data_dir / "windows_test.npz")["window_start"],
    )
    print(
        f"done: test macro-F1 {test_metrics['macro_f1']:.4f} "
        f"transition-F1 {test_metrics['transition_macro_f1']:.4f} "
        f"({test_metrics['n_windows']} windows, {n_params} params)",
        flush=True,
    )
    return test_metrics


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/windows")
    parser.add_argument("--model", choices=["resnet", "transformer"], default="resnet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Cap training windows (smoke tests)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
