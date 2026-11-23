"""Load raw UCI HAPT trials and extract fixed windows under subject-disjoint splits.

This module reads the same pinned release that the ``uci/hapt`` TimeNet connector converts:
``RawData/acc_expXX_userYY.txt`` and ``gyro_expXX_userYY.txt`` hold three-axis samples at 50 Hz,
and ``RawData/labels.txt`` marks contiguous labeled segments with 1-indexed inclusive bounds.

Each window is labelled by the segment containing its centre sample, which is the standard
sliding-window convention for continuous activity recognition and keeps brief postural
transitions visible to the model.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transitionsense.labels import (
    ACTIVITY_ID_TO_INDEX,
    N_CLASSES,
    SPLIT_NAMES,
    split_for_subject,
    subjects_for_split,
)

SAMPLE_RATE_HZ = 50
WINDOW = 128
STRIDE = 64
N_CHANNELS = 6
CHANNEL_NAMES: tuple[str, ...] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


@dataclass(frozen=True)
class Segment:
    """One contiguous labeled segment of a trial, in 0-indexed half-open sample bounds."""

    start: int
    end: int
    label: int


@dataclass(frozen=True)
class Trial:
    """One continuous HAPT trial: six IMU channels and its labeled segments."""

    experiment_id: int
    subject_id: int
    signals: np.ndarray
    segments: tuple[Segment, ...]

    @property
    def split(self) -> str:
        """The subject-disjoint split of the trial."""
        return split_for_subject(self.subject_id)


def _read_labels(path: Path) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    """Read ``RawData/labels.txt`` grouped by ``(experiment_id, subject_id)``.

    Args:
        path: The labels file.

    Returns:
        ``(experiment_id, subject_id)`` to a list of ``(start, end, activity_id)`` rows with
        1-indexed inclusive bounds, in file order.

    Raises:
        ValueError: If a row is malformed or names an unknown activity id.
    """
    grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path.name} line {lineno}: expected 5 columns, got {len(parts)}")
        experiment_id, subject_id, activity_id, start, end = (int(part) for part in parts)
        if activity_id not in ACTIVITY_ID_TO_INDEX:
            raise ValueError(f"{path.name} line {lineno}: unknown activity id {activity_id}")
        grouped.setdefault((experiment_id, subject_id), []).append((start, end, activity_id))
    return grouped


def _read_imu_pair(raw_dir: Path, experiment_id: int, subject_id: int) -> np.ndarray:
    """Read one trial's accelerometer and gyroscope files into an ``(n, 6)`` array.

    Args:
        raw_dir: The ``RawData`` directory.
        experiment_id: The experiment id.
        subject_id: The subject id.

    Returns:
        ``float32`` samples with columns ``acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z``.

    Raises:
        ValueError: If the files are missing, malformed, or of different lengths.
    """
    stem = f"exp{experiment_id:02d}_user{subject_id:02d}"
    arrays = []
    for kind in ("acc", "gyro"):
        path = raw_dir / f"{kind}_{stem}.txt"
        try:
            data = np.loadtxt(path, dtype=np.float64, ndmin=2)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] == 0:
            raise ValueError(f"{path.name} must be a non-empty (n, 3) array")
        arrays.append(data.astype(np.float32))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(
            f"experiment {experiment_id} subject {subject_id}: accelerometer and gyroscope lengths differ"
        )
    return np.concatenate(arrays, axis=1)


def load_trials(hapt_root: Path) -> list[Trial]:
    """Load every labeled trial of the extracted HAPT release.

    Args:
        hapt_root: The extraction root containing ``RawData/``.

    Returns:
        The trials sorted by ``(experiment_id, subject_id)``.

    Raises:
        ValueError: If ``RawData/labels.txt`` or a trial file is missing or malformed.
    """
    raw_dir = hapt_root / "RawData"
    grouped = _read_labels(raw_dir / "labels.txt")
    trials: list[Trial] = []
    for (experiment_id, subject_id), rows in sorted(grouped.items()):
        signals = _read_imu_pair(raw_dir, experiment_id, subject_id)
        n_samples = signals.shape[0]
        segments = tuple(
            sorted(
                (
                    Segment(
                        start=start - 1,
                        end=end,
                        label=ACTIVITY_ID_TO_INDEX[activity_id],
                    )
                    for start, end, activity_id in rows
                ),
                key=lambda segment: segment.start,
            )
        )
        for segment in segments:
            if segment.start < 0 or segment.end > n_samples or segment.end <= segment.start:
                raise ValueError(
                    f"labels.txt segment exp{experiment_id} user{subject_id} "
                    f"[{segment.start}, {segment.end}] leaves the {n_samples}-sample trial"
                )
        trials.append(
            Trial(
                experiment_id=experiment_id,
                subject_id=subject_id,
                signals=signals,
                segments=segments,
            )
        )
    return trials


def _label_at(segments: tuple[Segment, ...], index: int) -> int | None:
    """Return the label of the segment containing ``index``, or ``None`` in a gap."""
    # Segments are sorted and non-overlapping; bisect on start bounds.
    starts = [segment.start for segment in segments]
    position = bisect.bisect_right(starts, index) - 1
    if position < 0:
        return None
    segment = segments[position]
    return segment.label if index < segment.end else None


def iter_windows(
    trial: Trial,
    *,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> list[tuple[int, int]]:
    """Return ``(window_start, label)`` pairs for one trial.

    A window is kept only when its centre sample falls inside a labeled segment. Windows whose
    centre lands in an unlabeled gap are dropped.

    Args:
        trial: The trial to window.
        window: Window length in samples.
        stride: Stride between consecutive window starts.

    Returns:
        ``(start, label)`` pairs in time order.
    """
    n_samples = trial.signals.shape[0]
    pairs: list[tuple[int, int]] = []
    for start in range(0, n_samples - window + 1, stride):
        label = _label_at(trial.segments, start + window // 2)
        if label is not None:
            pairs.append((start, label))
    return pairs


def build_windows(
    trials: list[Trial],
    split: str,
    *,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> dict[str, np.ndarray]:
    """Extract windows for one split into arrays ready for ``np.savez``.

    Args:
        trials: All loaded trials; only those in ``split`` are used.
        split: One of ``SPLIT_NAMES``.
        window: Window length in samples.
        stride: Stride between consecutive window starts.

    Returns:
        A dict with ``X`` ``[n, 6, window]`` float32, ``y`` int64 labels, ``subject_id`` int64,
        ``experiment_id`` int64, and ``window_start`` int64 provenance columns.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    wanted = subjects_for_split(split)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    experiments: list[int] = []
    starts: list[int] = []
    for trial in trials:
        if trial.subject_id not in wanted:
            continue
        for start, label in iter_windows(trial, window=window, stride=stride):
            xs.append(trial.signals[start : start + window].T)
            ys.append(label)
            subjects.append(trial.subject_id)
            experiments.append(trial.experiment_id)
            starts.append(start)
    return {
        "X": np.stack(xs).astype(np.float32) if xs else np.empty((0, N_CHANNELS, window), np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "subject_id": np.asarray(subjects, dtype=np.int64),
        "experiment_id": np.asarray(experiments, dtype=np.int64),
        "window_start": np.asarray(starts, dtype=np.int64),
    }


def split_summary(trials: list[Trial]) -> dict[str, dict[str, int | list[int]]]:
    """Summarize subjects, trials, and labeled segments per split.

    Args:
        trials: All loaded trials.

    Returns:
        Per-split subject ids, trial count, and segment count.
    """
    summary: dict[str, dict[str, int | list[int]]] = {}
    for split in SPLIT_NAMES:
        wanted = subjects_for_split(split)
        split_trials = [trial for trial in trials if trial.subject_id in wanted]
        summary[split] = {
            "subjects": sorted(wanted),
            "n_trials": len(split_trials),
            "n_segments": sum(len(trial.segments) for trial in split_trials),
        }
    return summary


def verify_no_subject_overlap(subject_sets: dict[str, set[int]]) -> dict[str, int]:
    """Assert pairwise disjointness of split subject sets and return intersection sizes.

    Args:
        subject_sets: Mapping of split name to its subject ids, derived from the data.

    Returns:
        Pairwise intersection sizes (all zero for a valid subject-disjoint split).

    Raises:
        ValueError: If any two splits share a subject.
    """
    names = sorted(subject_sets)
    overlaps: dict[str, int] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            size = len(subject_sets[first] & subject_sets[second])
            overlaps[f"{first}_{second}"] = size
            if size:
                raise ValueError(f"subject overlap between {first} and {second}: {size} subjects")
    return overlaps


def observed_subjects_per_split(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, set[int]]:
    """Collect the subject ids actually present in each split's window arrays.

    Args:
        arrays: Mapping of split name to the dict returned by :func:`build_windows`.

    Returns:
        Split name to the set of subject ids seen in its windows.
    """
    return {
        split: {int(subject) for subject in data["subject_id"].tolist()}
        for split, data in arrays.items()
    }


def n_classes() -> int:
    """Return the number of activity classes."""
    return N_CLASSES
