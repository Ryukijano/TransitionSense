"""Evaluate a saved checkpoint on the test split and write metrics JSON.

Example:
    uv run python -m transitionsense.evaluate --checkpoint results/resnet/best.pt \
        --data-dir data/windows --output results/resnet/test_metrics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from transitionsense.metrics import evaluate_predictions, subject_overlap_report, write_metrics
from transitionsense.models import build_model


def main() -> None:
    """Parse arguments, score the test split, and write the metrics payload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/windows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["model"], **checkpoint.get("config", {})).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = np.load(Path(args.data_dir) / "windows_test.npz")
    x = torch.from_numpy(data["X"].astype(np.float32))
    y = data["y"].astype(np.int64)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            batch = x[start : start + args.batch_size].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)

    subject_sets = {
        split: {int(s) for s in np.unique(np.load(Path(args.data_dir) / f"windows_{split}.npz")["subject_id"])}
        for split in ("train", "validation", "test")
    }
    payload = evaluate_predictions(y, y_pred)
    payload.update(
        {
            "model": checkpoint["model"],
            "checkpoint": str(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "split": "test",
            "split_subjects": subject_overlap_report(subject_sets),
        }
    )
    write_metrics(Path(args.output), payload)
    np.savez(
        Path(args.output).with_suffix(".predictions.npz"),
        y_true=y,
        y_pred=y_pred,
        subject_id=data["subject_id"],
        experiment_id=data["experiment_id"],
        window_start=data["window_start"],
    )
    print(f"macro-F1 {payload['macro_f1']:.4f}  transition-F1 {payload['transition_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
