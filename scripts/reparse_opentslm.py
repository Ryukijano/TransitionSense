"""Re-parse stored OpenTSLM generations and rewrite test metrics — no GPU needed.

Reads ``<result-dir>/predictions.jsonl`` (written by ``train_opentslm``), applies the
current ``_parse_answer`` rule to each ``generated`` text, recomputes metrics with
``evaluate_with_reject``, and rewrites ``opentslm_test_metrics.json`` and
``test_predictions.npz``. Preserves the original ``extra`` metadata fields.

Example:
    python -m scripts.reparse_opentslm --result-dir results/opentslm-gemma270m-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from transitionsense.labels import ACTIVITY_NAMES
from transitionsense.metrics import REJECT_INDEX, evaluate_with_reject, write_metrics
from transitionsense.train_opentslm import _parse_answer


def reparse(result_dir: Path) -> dict:
    """Recompute metrics for one result directory from its stored generations.

    Args:
        result_dir: Directory containing ``predictions.jsonl`` and the old
            ``opentslm_test_metrics.json`` (for preserved metadata).

    Returns:
        The new metrics payload.
    """
    pred_path = result_dir / "predictions.jsonl"
    metrics_path = result_dir / "opentslm_test_metrics.json"
    if not pred_path.is_file():
        raise FileNotFoundError(f"missing {pred_path}")
    old = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}

    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_to_index = {name: index for index, name in enumerate(ACTIVITY_NAMES)}
    y_true = np.asarray([int(row["label_index"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray(
        [label_to_index.get(_parse_answer(row["generated"]), REJECT_INDEX) for row in rows],
        dtype=np.int64,
    )

    # preserve every top-level field that is not a metric output
    metric_keys = {
        "n_windows", "n_unparseable", "parse_rate", "accuracy", "macro_f1",
        "transition_macro_f1", "per_class", "confusion_matrix", "classes",
        "transition_classes", "reject_column",
    }
    extra = {k: v for k, v in old.items() if k not in metric_keys}
    extra["parse_rule"] = "answer_span_else_first_label_in_text"
    payload = evaluate_with_reject(y_true, y_pred, extra=extra)
    write_metrics(metrics_path, payload)
    np.savez(
        result_dir / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        subject_id=np.asarray([int(row["subject_id"]) for row in rows], dtype=np.int64),
        experiment_id=np.asarray([int(row["experiment_id"]) for row in rows], dtype=np.int64),
        window_start=np.asarray([int(row["window_start"]) for row in rows], dtype=np.int64),
    )
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = reparse(args.result_dir)
    print(
        f"{args.result_dir.name}: macro-F1 {payload['macro_f1']:.4f} "
        f"transition-F1 {payload['transition_macro_f1']:.4f} parse-rate {payload['parse_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
