import json
from pathlib import Path
from typing import Callable

from datasets import Dataset, Audio

METADATA_FILE = Path("dataset/segments_metadata.jsonl")
SAMPLE_RATE = 16000


def load_dataset(
    filter_fn: Callable[[dict], bool] | None = None,
    cast_audio: bool = False,
) -> Dataset:
    """Load the local segmented dataset.

    Args:
        filter_fn: optional callable applied to each raw metadata dict.
        cast_audio: if True, cast the ``audio`` column to HF Audio feature
                    (required for push_to_hub so audio files are uploaded).
                    Leave False for training (audio loaded on demand via soundfile).
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

    dataset = Dataset.from_list(segments)

    if cast_audio:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    return dataset
