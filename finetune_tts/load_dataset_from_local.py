import json
from pathlib import Path

from datasets import Audio, Dataset

METADATA_FILE = Path("dataset/segments_metadata.jsonl")
SAMPLE_RATE = 16000


def load_dataset():
    segments = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                segments.append(json.loads(line))

    print(f"📋 {len(segments)} segments loaded from {METADATA_FILE}")

    # Build dataset
    ds = Dataset.from_list(segments)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return ds
