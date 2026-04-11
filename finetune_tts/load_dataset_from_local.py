import json
from pathlib import Path
from typing import Callable

from datasets import Dataset

METADATA_FILE = Path("dataset/segments_metadata.jsonl")
SAMPLE_RATE = 16000


def load_dataset(filter_fn: Callable[[dict], bool] | None = None) -> Dataset:
    """Load the local segmented dataset.

    The ``audio`` column is kept as a plain file-path string — no HF Audio cast.
    Audio is loaded on demand in the training Dataset class using soundfile,
    which avoids pulling in torchcodec entirely.

    Args:
        filter_fn: optional callable applied to each raw metadata dict.
    """
    segments = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                segments.append(json.loads(line))

    print(f"📋 {len(segments)} segments loaded from {METADATA_FILE}")

    if filter_fn is not None:
        before = len(segments)
        segments = [s for s in segments if filter_fn(s)]
        print(f"🔍 Filtered: {before} → {len(segments)} segments")

    return Dataset.from_list(segments)
