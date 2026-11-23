"""Optional cross-check of experiment windows against the built ``uci/hapt`` TimeF release.

This module is imported lazily by ``scripts/prepare_windows.py`` so the training path never
requires ``timenet`` to be installed.
"""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from transitionsense.labels import ACTIVITY_NAMES

US_PER_SAMPLE = 1_000_000 // 50


def verify_windows_against_timef(
    window_dir: Path,
    registry_root: Path,
    *,
    window: int,
) -> dict:
    """Check every window's centre against the TimeF task that covers it.

    For each window the function finds the ``uci/hapt`` classification task whose scope contains
    the window centre and asserts that the task's split and label match the experiment split and
    label. This proves the experiment pipeline consumes the same subject-disjoint release that
    the TimeNet connector builds.

    Args:
        window_dir: Directory holding ``windows_{train,validation,test}.npz``.
        registry_root: Root of the local registry containing ``uci/hapt``.
        window: Window length in samples (to compute the centre sample).

    Returns:
        ``{"checked": n, "n_mismatches": m, "mismatches": [...], "ok": bool}``.

    Raises:
        ImportError: If ``timenet`` is not installed.
        ValueError: If the registry does not contain ``uci/hapt``.
    """
    import timenet  # noqa: PLC0415 - optional dependency, imported lazily
    from timenet.types import ClassificationTask, read_value  # noqa: PLC0415

    with timenet.TimeNet(sources=[str(registry_root)]).open("uci/hapt") as reader:
        dataset = reader.read()

    tasks_by_trial: dict[tuple[int, int], list] = {}
    for task in dataset.tasks_of(ClassificationTask):
        (record,) = task.inputs
        key = (int(record.metadata["experiment_id"]), int(record.metadata["subject_id"]))
        tasks_by_trial.setdefault(key, []).append(task)
    for tasks in tasks_by_trial.values():
        tasks.sort(key=lambda task: task.scope.start_us)

    mismatches: list[dict] = []
    checked = 0
    for split in ("train", "validation", "test"):
        path = window_dir / f"windows_{split}.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        for index in range(len(data["y"])):
            experiment_id = int(data["experiment_id"][index])
            subject_id = int(data["subject_id"][index])
            centre_us = (int(data["window_start"][index]) + window // 2) * US_PER_SAMPLE
            tasks = tasks_by_trial.get((experiment_id, subject_id))
            if not tasks:
                mismatches.append({"split": split, "index": index, "reason": "no TimeF record"})
                continue
            bounds = [task.scope.start_us for task in tasks]
            position = bisect.bisect_right(bounds, centre_us) - 1
            task = tasks[position] if position >= 0 else None
            if task is None or centre_us >= task.scope.exclusive_end:
                mismatches.append({"split": split, "index": index, "reason": "centre in unlabeled gap"})
                continue
            task_split = str(task.split)
            task_label = read_value(task.targets[0])
            expected_label = ACTIVITY_NAMES[int(data["y"][index])]
            if task_split != split or task_label != expected_label:
                mismatches.append(
                    {
                        "split": split,
                        "index": index,
                        "experiment_id": experiment_id,
                        "subject_id": subject_id,
                        "expected": {"split": split, "label": expected_label},
                        "timef": {"split": task_split, "label": task_label},
                    }
                )
            checked += 1
    return {
        "checked": checked,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "ok": not mismatches,
    }
