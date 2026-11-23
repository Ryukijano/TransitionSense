"""Build subject-disjoint HAPT window NPZ files from the raw release.

Example:
    uv run python -m transitionsense.prepare --hapt-root ../data/hapt --output-dir data/windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from transitionsense.data import (
    CHANNEL_NAMES,
    SAMPLE_RATE_HZ,
    STRIDE,
    WINDOW,
    build_windows,
    load_trials,
    observed_subjects_per_split,
    split_summary,
    verify_no_subject_overlap,
)
from transitionsense.labels import ACTIVITY_NAMES, N_CLASSES, SPLIT_NAMES


def main() -> None:
    """Extract windows for all three splits and write NPZ files plus ``splits.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hapt-root", type=Path, default=Path("../data/hapt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/windows"))
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument(
        "--timef-registry",
        type=Path,
        default=None,
        help="Local registry root holding the built uci/hapt release; enables the TimeF cross-check.",
    )
    args = parser.parse_args()

    trials = load_trials(args.hapt_root)
    print(f"loaded {len(trials)} trials")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        data = build_windows(trials, split, window=args.window, stride=args.stride)
        arrays[split] = data
        np.savez(args.output_dir / f"windows_{split}.npz", **data)
        counts = np.bincount(data["y"], minlength=N_CLASSES)
        class_counts[split] = {ACTIVITY_NAMES[i]: int(counts[i]) for i in range(N_CLASSES)}
        print(
            f"{split}: {len(data['y'])} windows from {len(np.unique(data['subject_id']))} subjects",
            flush=True,
        )

    observed = observed_subjects_per_split(arrays)
    overlaps = verify_no_subject_overlap(observed)

    splits_payload = {
        "hapt_root": str(args.hapt_root),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window": args.window,
        "stride": args.stride,
        "channel_names": list(CHANNEL_NAMES),
        "classes": list(ACTIVITY_NAMES),
        "split_summary": split_summary(trials),
        "observed_subjects": {split: sorted(subjects) for split, subjects in observed.items()},
        "subject_overlaps": overlaps,
        "window_counts": {split: int(len(arrays[split]["y"])) for split in SPLIT_NAMES},
        "class_counts": class_counts,
    }
    (args.output_dir / "splits.json").write_text(json.dumps(splits_payload, indent=2) + "\n", encoding="utf-8")
    print(f"subject overlaps: {overlaps}")

    if args.timef_registry is not None:
        from transitionsense.timef_verify import verify_windows_against_timef

        report = verify_windows_against_timef(args.output_dir, args.timef_registry, window=args.window)
        (args.output_dir / "timef_verify.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"TimeF cross-check: {report['checked']} windows checked, {report['n_mismatches']} mismatches"
        )
        if not report["ok"]:
            raise SystemExit("TimeF cross-check failed; see timef_verify.json")


if __name__ == "__main__":
    main()
