"""OpenTSLM batch adapter for HAPT windows.

``HaptWindowDataset`` serves rows from the precomputed ``windows_{split}.npz`` files written by
``transitionsense.prepare``. ``HaptPromptDataset`` formats each window into the dict contract that
``OpenTSLMSP.compute_loss``/``generate`` consume: ``pre_prompt``, ``time_series_text``,
``time_series``, ``post_prompt``, and ``answer``.

Each of the six IMU channels becomes its own univariate series with its own caption, because
``TransformerCNNEncoder`` only accepts ``[B, L]`` inputs. Formatting is lazy per ``__getitem__``:
OpenTSLM's ``QADataset`` formats and caches all three splits eagerly, which is wasteful for
~40 MB of window arrays and, worse, caches formatted variants at class level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from transitionsense.labels import ACTIVITY_NAMES, N_CLASSES

CHANNEL_CAPTIONS: tuple[str, ...] = (
    "accelerometer x-axis in g",
    "accelerometer y-axis in g",
    "accelerometer z-axis in g",
    "gyroscope x-axis in rad/s",
    "gyroscope y-axis in rad/s",
    "gyroscope z-axis in rad/s",
)

LABEL_ALIASES: dict[str, str] = {
    "lying": "laying",
    "laying_down": "laying",
    "walk": "walking",
    "walk_upstairs": "walking_upstairs",
    "walk_downstairs": "walking_downstairs",
    "stand_to_sitting": "stand_to_sit",
    "sit_to_standing": "sit_to_stand",
    "sit_to_lying": "sit_to_lie",
    "lie_to_sitting": "lie_to_sit",
    "stand_to_lying": "stand_to_lie",
    "lie_to_standing": "lie_to_stand",
}

PRE_PROMPT = (
    "You are given six inertial measurement unit channels from a smartphone "
    "worn on the waist, sampled at 50 Hz over 2.56 seconds: three "
    "accelerometer axes in g and three gyroscope axes in rad/s."
)

POST_PROMPT_TEMPLATE = (
    "Classify the activity or postural transition. Answer ONLY with the "
    "activity label. The following labels are possible: {labels}. "
    "You MUST end your response with 'Answer: <class label>'."
)


def normalize_label(text: str) -> str:
    """Normalize a generated label fragment to the dataset vocabulary.

    Args:
        text: Raw generated text fragment.

    Returns:
        The canonical label name, or ``""`` when nothing matches.
    """
    cleaned = text.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = cleaned.strip("_.,;:!?()[]{}\"'")
    if cleaned in ACTIVITY_NAMES:
        return cleaned
    if cleaned in LABEL_ALIASES:
        return LABEL_ALIASES[cleaned]
    for name in ACTIVITY_NAMES:
        if name in cleaned or cleaned in name:
            return name
    return ""


class HaptWindowDataset(Dataset):
    """Torch dataset over one split's precomputed HAPT windows."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        """Load ``windows_{split}.npz`` from ``data_dir``.

        Args:
            data_dir: Directory written by ``transitionsense.prepare``.
            split: ``"train"``, ``"validation"``, or ``"test"``.

        Raises:
            FileNotFoundError: If the NPZ file does not exist.
        """
        path = Path(data_dir) / f"windows_{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run `python -m transitionsense.prepare` first")
        data = np.load(path)
        self.x = data["X"]
        self.y = data["y"]
        self.subject_id = data["subject_id"]
        self.experiment_id = data["experiment_id"]
        self.window_start = data["window_start"]

    def __len__(self) -> int:
        """Return the number of windows."""
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> dict:
        """Return raw row fields plus six channel views (no formatting)."""
        return {
            "channels": self.x[index],
            "label": ACTIVITY_NAMES[int(self.y[index])],
            "label_index": int(self.y[index]),
            "subject_id": int(self.subject_id[index]),
            "experiment_id": int(self.experiment_id[index]),
            "window_start": int(self.window_start[index]),
        }


class HaptPromptDataset(Dataset):
    """Format ``HaptWindowDataset`` rows into OpenTSLM prompt dicts on the fly."""

    def __init__(self, base: HaptWindowDataset, *, eos_token: str) -> None:
        """Wrap a window dataset.

        Args:
            base: The underlying window dataset.
            eos_token: Token appended to training answers, matching ``QADataset`` behavior.
        """
        self.base = base
        self.eos_token = eos_token
        self.post_prompt = POST_PROMPT_TEMPLATE.format(labels=", ".join(ACTIVITY_NAMES))

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        """Return one OpenTSLM-format sample dict."""
        row = self.base[index]
        channels = row["channels"]
        answer = row["label"]
        if self.eos_token and not answer.endswith(self.eos_token):
            answer = answer + self.eos_token
        return {
            "pre_prompt": PRE_PROMPT,
            "post_prompt": self.post_prompt,
            "time_series_text": list(CHANNEL_CAPTIONS),
            "time_series": [channels[channel] for channel in range(len(CHANNEL_CAPTIONS))],
            "answer": answer,
            "label": row["label"],
            "label_index": row["label_index"],
            "subject_id": row["subject_id"],
            "experiment_id": row["experiment_id"],
            "window_start": row["window_start"],
        }


def load_prompt_datasets(
    data_dir: str | Path, *, eos_token: str
) -> dict[str, HaptPromptDataset]:
    """Build train/validation/test prompt datasets for one ``data_dir``.

    Args:
        data_dir: Directory written by ``transitionsense.prepare``.
        eos_token: Token appended to training answers.

    Returns:
        Mapping of split name to ``HaptPromptDataset``.
    """
    return {
        split: HaptPromptDataset(HaptWindowDataset(data_dir, split), eos_token=eos_token)
        for split in ("train", "validation", "test")
    }


def n_classes() -> int:
    """Return the number of HAPT classes."""
    return N_CLASSES
