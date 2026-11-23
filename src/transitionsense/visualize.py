"""Generate result figures: confusion matrices, per-class F1, model comparison, examples.

Example:
    uv run python -m transitionsense.visualize --results-dir results --output-dir results/figures \
        --hapt-root ../data/hapt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transitionsense.data import load_trials
from transitionsense.labels import ACTIVITY_NAMES, TRANSITION_INDICES


METRICS_NAMES = ("test_metrics.json", "opentslm_test_metrics.json")


def _metrics_path(model_dir: Path) -> Path | None:
    """Return the metrics JSON inside ``model_dir`` (supervised or generative name)."""
    for name in METRICS_NAMES:
        path = model_dir / name
        if path.is_file():
            return path
    return None


def _model_dirs(results_dir: Path, models: list[str] | None = None) -> list[Path]:
    """Return result subdirectories that hold a metrics JSON.

    Args:
        results_dir: Directory containing per-model result subdirectories.
        models: Explicit subdirectory names to include; when ``None``, every subdirectory
            with metrics is used except names ending in ``-smoke``.

    Returns:
        Sorted list of model result directories.
    """
    if models is not None:
        dirs = [results_dir / name for name in models]
    else:
        dirs = sorted(
            path
            for path in results_dir.iterdir()
            if path.is_dir()
            and _metrics_path(path) is not None
            and not path.name.endswith("-smoke")
        )
    missing = [path.name for path in dirs if _metrics_path(path) is None]
    if missing:
        raise FileNotFoundError(f"no metrics JSON under: {', '.join(missing)}")
    return dirs


def _load_metrics(model_dir: Path) -> dict:
    """Read a model's test metrics payload."""
    path = _metrics_path(model_dir)
    if path is None:
        raise FileNotFoundError(f"no metrics JSON under {model_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion_matrices(
    results_dir: Path, output_dir: Path, models: list[str] | None = None
) -> list[Path]:
    """Write one normalized confusion-matrix heatmap per model.

    Args:
        results_dir: Directory containing per-model result subdirectories.
        output_dir: Figure output directory.
        models: Explicit model subdirectory names (see :func:`_model_dirs`).

    Returns:
        The written PNG paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model_dir in _model_dirs(results_dir, models):
        metrics = _load_metrics(model_dir)
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        col_labels = list(ACTIVITY_NAMES)
        if matrix.shape[1] == len(ACTIVITY_NAMES) + 1:
            col_labels.append("unparseable")
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(matrix.shape[1]), col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ACTIVITY_NAMES)), ACTIVITY_NAMES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{metrics.get('model', model_dir.name)} — test confusion matrix (row-normalized)")
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        path = output_dir / f"confusion_matrix_{model_dir.name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_per_class_f1(
    results_dir: Path, output_dir: Path, models: list[str] | None = None
) -> list[Path]:
    """Write a per-class F1 bar chart per model, highlighting transition classes.

    Args:
        results_dir: Directory containing per-model result subdirectories.
        output_dir: Figure output directory.
        models: Explicit model subdirectory names (see :func:`_model_dirs`).

    Returns:
        The written PNG paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model_dir in _model_dirs(results_dir, models):
        metrics = _load_metrics(model_dir)
        scores = metrics["per_class"]
        f1 = [float(scores[name]["f1"]) for name in ACTIVITY_NAMES]
        colors = ["#d95f02" if i in TRANSITION_INDICES else "#1b9e77" for i in range(len(ACTIVITY_NAMES))]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(range(len(ACTIVITY_NAMES)), f1, color=colors)
        ax.set_xticks(range(len(ACTIVITY_NAMES)), ACTIVITY_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.set_title(f"{metrics.get('model', model_dir.name)} — per-class test F1 (orange = transitions)")
        fig.tight_layout()
        path = output_dir / f"per_class_f1_{model_dir.name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_model_comparison(
    results_dir: Path, output_dir: Path, models: list[str] | None = None
) -> Path | None:
    """Write a grouped bar chart of macro-F1 and transition macro-F1 across models.

    Args:
        results_dir: Directory containing per-model result subdirectories.
        output_dir: Figure output directory.
        models: Explicit model subdirectory names (see :func:`_model_dirs`).

    Returns:
        The written PNG path, or ``None`` when fewer than one model exists.
    """
    model_dirs = _model_dirs(results_dir, models)
    if not model_dirs:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [model_dir.name for model_dir in model_dirs]
    macro = [_load_metrics(model_dir)["macro_f1"] for model_dir in model_dirs]
    trans = [_load_metrics(model_dir)["transition_macro_f1"] for model_dir in model_dirs]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.5))
    ax.bar(x - width / 2, macro, width, label="macro-F1 (all classes)", color="#4575b4")
    ax.bar(x + width / 2, trans, width, label="macro-F1 (transitions only)", color="#d95f02")
    ax.set_xticks(x, names)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("TransitionSense — subject-disjoint test comparison")
    fig.tight_layout()
    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_loss_curves(
    results_dir: Path, output_dir: Path, models: list[str] | None = None
) -> Path | None:
    """Write a training/validation metric curve figure from ``history.json`` files.

    Args:
        results_dir: Directory containing per-model result subdirectories.
        output_dir: Figure output directory.
        models: Explicit model subdirectory names (see :func:`_model_dirs`).

    Returns:
        The written PNG path, or ``None`` when no history files exist.
    """
    histories = [
        (model_dir.name, json.loads((model_dir / "history.json").read_text(encoding="utf-8")))
        for model_dir in _model_dirs(results_dir, models)
        if (model_dir / "history.json").is_file()
    ]
    if not histories:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    has_f1 = any("val_macro_f1" in history[0] for _, history in histories)
    for name, history in histories:
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=name)
        if "val_macro_f1" in history[0]:
            axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label=f"{name} macro-F1")
            axes[1].plot(
                epochs,
                [row["val_transition_macro_f1"] for row in history],
                label=f"{name} transition-F1",
                ls="--",
            )
        elif has_f1:
            continue
        else:
            axes[1].plot(epochs, [row["val_loss"] for row in history], label=f"{name} val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("F1" if has_f1 else "val loss")
    if has_f1:
        axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    fig.suptitle("Training curves")
    fig.tight_layout()
    path = output_dir / "loss_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_example_transitions(
    model_dir: Path,
    output_dir: Path,
    *,
    hapt_root: Path | None,
    n_examples: int = 8,
) -> Path | None:
    """Plot raw IMU windows around transition predictions for qualitative inspection.

    Args:
        model_dir: One model's result directory (must contain ``test_predictions.npz``).
        output_dir: Figure output directory.
        hapt_root: Extracted HAPT root used to fetch the raw signals; the figure is skipped
            when ``None``.
        n_examples: Number of transition windows to show.

    Returns:
        The written PNG path, or ``None`` when predictions or raw data are unavailable.
    """
    if hapt_root is None:
        return None
    pred_path = model_dir / "test_predictions.npz"
    if not pred_path.is_file():
        pred_path = model_dir / "test_metrics.predictions.npz"
    if not pred_path.is_file():
        return None
    preds = np.load(pred_path)
    trials = {(trial.experiment_id, trial.subject_id): trial for trial in load_trials(hapt_root)}

    transition_idx = np.array(sorted(TRANSITION_INDICES))
    mask = np.isin(preds["y_true"], transition_idx) & (preds["y_pred"] < len(ACTIVITY_NAMES))
    order = np.argsort(preds["y_true"][mask])
    chosen = np.flatnonzero(mask)[order[:n_examples]]
    if chosen.size == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(chosen.size, 1, figsize=(12, 2.2 * chosen.size), squeeze=False)
    channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for row, index in enumerate(chosen):
        ax = axes[row, 0]
        trial = trials[(int(preds["experiment_id"][index]), int(preds["subject_id"][index]))]
        start = int(preds["window_start"][index])
        window_data = trial.signals[start : start + 128].T
        for channel in range(6):
            ax.plot(window_data[channel], lw=0.8, label=channel_names[channel] if row == 0 else None)
        true_name = ACTIVITY_NAMES[int(preds["y_true"][index])]
        pred_name = ACTIVITY_NAMES[int(preds["y_pred"][index])]
        mark = "correct" if preds["y_true"][index] == preds["y_pred"][index] else "wrong"
        ax.set_ylabel(f"{true_name}\n→ {pred_name} ({mark})", fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(fontsize=6, ncol=6, loc="upper right")
    fig.suptitle(f"Example transition windows — {model_dir.name} test predictions")
    fig.tight_layout()
    path = output_dir / "example_transitions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    """Generate every figure under ``--output-dir``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated result subdirectory names; default: all except *-smoke.",
    )
    parser.add_argument("--example-model", type=str, default=None,
                        help="Model dir for example_transitions.png (default: first selected).")
    parser.add_argument("--hapt-root", type=Path, default=None)
    parser.add_argument("--n-examples", type=int, default=8)
    args = parser.parse_args()

    models = [name.strip() for name in args.models.split(",")] if args.models else None
    selected = _model_dirs(args.results_dir, models)

    written = plot_confusion_matrices(args.results_dir, args.output_dir, models)
    written += plot_per_class_f1(args.results_dir, args.output_dir, models)
    comparison = plot_model_comparison(args.results_dir, args.output_dir, models)
    if comparison is not None:
        written.append(comparison)
    curves = plot_loss_curves(args.results_dir, args.output_dir, models)
    if curves is not None:
        written.append(curves)
    example_dir = (
        args.results_dir / args.example_model if args.example_model else selected[0]
        if selected
        else None
    )
    if example_dir is not None:
        examples = plot_example_transitions(
            example_dir, args.output_dir, hapt_root=args.hapt_root, n_examples=args.n_examples
        )
        if examples is not None:
            written.append(examples)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
