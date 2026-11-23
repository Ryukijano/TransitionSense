"""Label vocabulary and subject-disjoint split rules for UCI HAPT.

The names match the ``uci/hapt`` TimeNet connector exactly: index ``i`` is activity id ``i + 1``.
The six postural transitions are the last six entries.
"""

ACTIVITY_NAMES: tuple[str, ...] = (
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "sitting",
    "standing",
    "laying",
    "stand_to_sit",
    "sit_to_stand",
    "sit_to_lie",
    "lie_to_sit",
    "stand_to_lie",
    "lie_to_stand",
)

N_CLASSES = len(ACTIVITY_NAMES)
ACTIVITY_ID_TO_INDEX = {activity_id: activity_id - 1 for activity_id in range(1, N_CLASSES + 1)}
TRANSITION_NAMES: tuple[str, ...] = ACTIVITY_NAMES[6:]
TRANSITION_INDICES: frozenset[int] = frozenset(range(6, N_CLASSES))
BASIC_NAMES: tuple[str, ...] = ACTIVITY_NAMES[:6]

# The official UCI HAPT partition: 21 train subjects and 9 test subjects.
OFFICIAL_TEST_SUBJECTS: frozenset[int] = frozenset({2, 4, 9, 10, 12, 13, 18, 20, 24})
OFFICIAL_TRAIN_SUBJECTS: frozenset[int] = frozenset(
    subject for subject in range(1, 31) if subject not in OFFICIAL_TEST_SUBJECTS
)
# Deterministic validation carve-out: official train subjects whose id is divisible by five.
VALIDATION_SUBJECTS: frozenset[int] = frozenset(
    subject for subject in OFFICIAL_TRAIN_SUBJECTS if subject % 5 == 0
)
TRAIN_SUBJECTS: frozenset[int] = OFFICIAL_TRAIN_SUBJECTS - VALIDATION_SUBJECTS

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


def split_for_subject(subject_id: int) -> str:
    """Return the subject-disjoint split name for a HAPT subject id.

    Args:
        subject_id: The subject id from ``labels.txt``.

    Returns:
        ``"train"``, ``"validation"``, or ``"test"``.

    Raises:
        ValueError: If the subject is in neither official UCI split.
    """
    if subject_id in OFFICIAL_TEST_SUBJECTS:
        return "test"
    if subject_id in OFFICIAL_TRAIN_SUBJECTS:
        return "validation" if subject_id in VALIDATION_SUBJECTS else "train"
    raise ValueError(f"subject {subject_id} is in neither the official train nor test split")


def subjects_for_split(split: str) -> frozenset[int]:
    """Return the subject ids assigned to a split.

    Args:
        split: One of ``SPLIT_NAMES``.

    Returns:
        The subject ids of the split.
    """
    if split == "train":
        return TRAIN_SUBJECTS
    if split == "validation":
        return VALIDATION_SUBJECTS
    if split == "test":
        return OFFICIAL_TEST_SUBJECTS
    raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
