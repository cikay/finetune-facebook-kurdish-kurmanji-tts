#!/usr/bin/env python3
"""
Preprocess long news audio into short utterance-level segments for TTS training.

Uses faster-whisper to get sentence timestamps, then splits audio accordingly.
Outputs a HuggingFace Dataset ready for finetune-hf-vits.

Usage:
    pip install faster-whisper soundfile datasets
    python preprocess.py
"""

import json
import os
import re
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from faster_whisper import WhisperModel

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
AUDIO_DIR = DATASET_DIR / "audio"
TEXT_DIR = DATASET_DIR / "text"
META_FILE = DATASET_DIR / "metadata.jsonl"
OUTPUT_DIR = Path("processed_dataset")
SEGMENTS_DIR = OUTPUT_DIR / "segments"

WHISPER_MODEL = "large-v3"       # best accuracy for Kurdish
WHISPER_DEVICE = "auto"          # "cuda" / "cpu" / "auto"
WHISPER_COMPUTE = "float16"  # "float16" for GPU, "int8" for CPU

MIN_DURATION = 2.0   # seconds - skip very short segments
MAX_DURATION = 15.0   # seconds - skip very long segments
SAMPLE_RATE = 16000

# Kurdish Kurmanji text cleaning
def clean_text(text: str) -> str:
    """Clean text for TTS: normalize whitespace, remove junk."""
    text = text.strip()
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing punctuation noise
    text = text.strip("- –—…")
    return text


def load_metadata() -> list[dict]:
    """Load metadata.jsonl entries."""
    entries = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def segment_audio(model: WhisperModel, audio_path: Path, output_prefix: str) -> list[dict]:
    """
    Transcribe audio with timestamps and split into segments.
    Returns list of {path, text, duration} dicts.
    """
    segments_out = []

    # Transcribe with word timestamps
    segments, info = model.transcribe(
        str(audio_path),
        language="ku",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        ),
    )

    # Read full audio
    audio_data, sr = sf.read(audio_path)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"

    for i, seg in enumerate(segments):
        start = seg.start
        end = seg.end
        duration = end - start
        text = clean_text(seg.text)

        # Filter by duration
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue

        # Filter by text quality
        if not text or len(text) < 5:
            continue

        # Extract audio segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio_data[start_sample:end_sample]

        # Save segment
        seg_filename = f"{output_prefix}_{i:04d}.wav"
        seg_path = SEGMENTS_DIR / seg_filename
        sf.write(seg_path, segment_audio, sr)

        segments_out.append({
            "audio": str(seg_path),
            "text": text,
            "duration": round(duration, 2),
        })

    return segments_out


def main():
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    entries = load_metadata()
    print(f"📋 {len(entries)} entries in metadata")

    # Init Whisper
    print(f"🔄 Loading Whisper {WHISPER_MODEL}...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("✅ device:", model.device, "| compute:", model.compute_type)

    # Process each audio file
    all_segments = []
    for idx, entry in enumerate(entries):
        audio_path = DATASET_DIR / entry["audio_file"]
        video_id = entry["id"]

        if not audio_path.exists():
            print(f"  ⚠️  Missing audio: {audio_path}")
            continue

        print(f"[{idx+1}/{len(entries)}] {entry['title'][:60]}...")
        segs = segment_audio(model, audio_path, video_id)
        all_segments.extend(segs)
        print(f"  → {len(segs)} segments")

    print(f"\n✅ Total segments: {len(all_segments)}")

    # Calculate stats
    total_dur = sum(s["duration"] for s in all_segments)
    avg_dur = total_dur / len(all_segments) if all_segments else 0
    print(f"📊 Total duration: {total_dur/3600:.1f}h | Avg: {avg_dur:.1f}s")

    # Create HuggingFace Dataset
    print("📦 Creating HuggingFace Dataset...")
    ds = Dataset.from_list(all_segments)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # Train/eval split (95/5)
    ds_split = ds.train_test_split(test_size=0.05, seed=42)
    ds_dict = DatasetDict({
        "train": ds_split["train"],
        "test": ds_split["test"],
    })

    ds_dict.save_to_disk(str(OUTPUT_DIR / "hf_dataset"))
    print(f"💾 Saved to {OUTPUT_DIR / 'hf_dataset'}")
    print(f"   Train: {len(ds_dict['train'])} | Eval: {len(ds_dict['test'])}")

    # Also save a metadata JSON for reference
    with open(OUTPUT_DIR / "segments_metadata.jsonl", "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
