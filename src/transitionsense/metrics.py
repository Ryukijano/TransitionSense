"""Metrics for transition-aware evaluation: macro-F1, transition-only F1, confusion matrices.

All metrics are computed from integer label arrays with ``N_CLASSES`` classes; per-class
precision/recall/F1 come from the confusion matrix so zero-division is handled explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from transitionsense.labels import ACTIVITY_NAMES, N_CLASSES, TRANSITION_INDICES, TRANSITION_NAMES


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Compute the ``n_classes`` x ``n_classes`` confusion matrix.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        n_classes: Number of classes.

    Returns:
        Row ``i`` holds the predictions for true class ``i``.
    """
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_scores(matrix: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Derive per-class precision, recall, F1, and support from a confusion matrix.

    Args:
        matrix: The confusion matrix from :func:`confusion_matrix`.

    Returns:
        Class name to ``{"precision", "recall", "f1", "support"}``.
    """
    scores: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(ACTIVITY_NAMES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return scores


def _macro_f1(scores: dict[str, dict[str, float | int]], names: tuple[str, ...]) -> float:
    """Average F1 over the named classes."""
    values = [float(scores[name]["f1"]) for name in names]
    return float(np.mean(values)) if values else 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full metrics payload for a prediction set.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        extra: Additional top-level fields merged into the payload (model name, split, etc.).

    Returns:
        A JSON-serializable dict with accuracy, macro-F1, transition macro-F1, per-class
        scores, and the confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    matrix = confusion_matrix(y_true, y_pred)
    scores = per_class_scores(matrix)
    payload: dict[str, Any] = {
        "n_windows": int(y_true.size),
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "macro_f1": _macro_f1(scores, ACTIVITY_NAMES),
        "transition_macro_f1": _macro_f1(scores, TRANSITION_NAMES),
        "per_class": scores,
        "confusion_matrix": matrix.tolist(),
        "classes": list(ACTIVITY_NAMES),
        "transition_classes": list(TRANSITION_NAMES),
    }
    if extra:
        payload.update(extra)
    return payload


def subject_overlap_report(splits: dict[str, set[int]]) -> dict[str, Any]:
    """Report pairwise subject overlaps between splits.

    Args:
        splits: Mapping of split name to the subject ids observed in its windows.

    Returns:
        ``{"subjects": {split: sorted ids}, "overlaps": {pair: count}, "disjoint": bool}``.
    """
    names = sorted(splits)
    overlaps: dict[str, int] = {}
    disjoint = True
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            size = len(splits[first] & splits[second])
            overlaps[f"{first}_vs_{second}"] = size
            disjoint = disjoint and size == 0
    return {
        "subjects": {name: sorted(splits[name]) for name in names},
        "overlaps": overlaps,
        "disjoint": disjoint,
    }


REJECT_INDEX = N_CLASSES
"""Prediction column for generations that cannot be parsed into a class label."""


def confusion_matrix_with_reject(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES
) -> np.ndarray:
    """Compute a ``n_classes`` x ``n_classes + 1`` matrix with a reject column.

    ``y_pred`` values of ``REJECT_INDEX`` land in the last column: they are errors for their
    true class but are not false positives for any real class.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices, possibly containing ``REJECT_INDEX``.
        n_classes: Number of real classes.

    Returns:
        The rectangular confusion matrix.
    """
    matrix = np.zeros((n_classes, n_classes + 1), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def evaluate_with_reject(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metrics payload for predictions that may include ``REJECT_INDEX``.

    Per-class precision/recall/F1 treat reject predictions as errors for the true class. The
    reported ``confusion_matrix`` keeps the reject column so nothing is silently dropped.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices, possibly containing ``REJECT_INDEX``.
        extra: Additional top-level fields merged into the payload.

    Returns:
        A JSON-serializable metrics dict.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_pred.min(initial=0) < 0 or y_pred.max(initial=-1) > REJECT_INDEX:
        raise ValueError(f"y_pred must be in [0, {REJECT_INDEX}]")
    matrix = confusion_matrix_with_reject(y_true, y_pred)
    scores = per_class_scores(matrix)
    n_reject = int(matrix[:, REJECT_INDEX].sum())
    payload: dict[str, Any] = {
        "n_windows": int(y_true.size),
        "n_unparseable": n_reject,
        "parse_rate": 1.0 - (n_reject / y_true.size) if y_true.size else 0.0,
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "macro_f1": _macro_f1(scores, ACTIVITY_NAMES),
        "transition_macro_f1": _macro_f1(scores, TRANSITION_NAMES),
        "per_class": scores,
        "confusion_matrix": matrix.tolist(),
        "classes": list(ACTIVITY_NAMES),
        "transition_classes": list(TRANSITION_NAMES),
        "reject_column": "unparseable",
    }
    if extra:
        payload.update(extra)
    return payload


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    """Write a metrics payload as pretty JSON.

    Args:
        path: Destination file.
        payload: The metrics dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def transition_indices() -> frozenset[int]:
    """Return the label indices of the six postural transitions."""
    return TRANSITION_INDICES
