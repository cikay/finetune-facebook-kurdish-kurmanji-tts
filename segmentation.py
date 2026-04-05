#!/usr/bin/env python3
"""
Preprocess long Kurdish Kurmanji audio into short utterance-level segments
using CTC forced alignment with the MMS model.

Unlike the Whisper-based approach, this aligns the *ground truth* text to the
audio, producing accurate timestamps without ASR errors.

Usage:
   python segmentation.py
"""

import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch

import numpy as np
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_spans,
    postprocess_results,
)
from ctc_forced_aligner.alignment_utils import (
    forced_align,
    merge_repeats,
)

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
AUDIO_DIR = DATASET_DIR / "clean_audio"
TEXT_DIR = DATASET_DIR / "text"
META_FILE = DATASET_DIR / "metadata.jsonl"
OUTPUT_DIR = Path("ctc_processed_dataset")
SEGMENTS_DIR = DATASET_DIR / "segments"

LANGUAGE = "kmr"  # ISO 639-3 for Kurmanji Kurdish
SAMPLE_RATE = 16000

# Alignment params
BATCH_SIZE = 16  # batch size for emission generation
WINDOW_SIZE = 30  # seconds per alignment window
CONTEXT_SIZE = 2  # seconds of overlap between windows

# Segment filtering
MIN_DURATION = 2.0  # seconds
MAX_DURATION = 15.0  # seconds
MIN_WORDS = 3  # minimum words per segment
MIN_SCORE = -7.0  # minimum average alignment score (log-prob; more negative = worse)

# Sentence-ending punctuation for Kurdish text splitting
SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


# ── Fixed alignment (star token index bug workaround) ────────────────────────


def get_alignments_fixed(
    emissions: torch.Tensor,
    tokens: list,
    tokenizer,
):
    """
    Like ctc_forced_aligner.get_alignments but fixes the <star> token index.

    The HuggingFace tokenizer has extra special tokens (<pad>, </s>, <unk>)
    that inflate the vocab size. generate_emissions adds the star column at
    the last emissions index, so <star> must map there, not to len(vocab).
    """
    assert len(tokens) > 0, "Empty transcript"

    dictionary = tokenizer.get_vocab()
    dictionary = {k.lower(): v for k, v in dictionary.items()}
    # Fix: star index = last column of emissions (added by generate_emissions)
    star_idx = emissions.shape[-1] - 1
    dictionary["<star>"] = star_idx

    token_indices = [
        dictionary[c]
        for c in " ".join(tokens).split(" ")
        if c in dictionary
    ]

    blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)

    if not emissions.is_cpu:
        emissions = emissions.cpu()
    targets = np.asarray([token_indices], dtype=np.int64)

    path, scores = forced_align(
        emissions.unsqueeze(0).float().numpy(),
        targets,
        blank=blank_id,
    )
    path = path.squeeze().tolist()

    idx_to_token_map = {v: k for k, v in dictionary.items()}
    segments = merge_repeats(path, idx_to_token_map)
    return segments, scores, idx_to_token_map[blank_id]


# ── Text utilities ───────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Normalize whitespace and light cleanup for Kurdish Kurmanji text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_into_sentences(text: str) -> list[str]:
    """Split Kurdish text into sentences using punctuation boundaries."""
    parts = SENT_END_RE.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


# ── Core pipeline ────────────────────────────────────────────────────────────


def load_metadata() -> list[dict]:
    """Load metadata.jsonl entries."""
    entries = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def align_and_segment(
    alignment_model,
    alignment_tokenizer,
    audio_path: Path,
    text: str,
    output_prefix: str,
    device: str,
    dtype: torch.dtype,
) -> list[dict]:
    """
    Perform forced alignment of ground truth text against audio,
    then split into sentence-level segments.

    Returns list of {audio, text, duration, score} dicts.
    """
    segments_out = []

    # Load audio via ctc-forced-aligner (handles resampling to 16kHz mono)
    audio_waveform = load_audio(str(audio_path), dtype, device)

    # Generate CTC emissions from the alignment model
    emissions, stride = generate_emissions(
        alignment_model,
        audio_waveform,
        window_length=WINDOW_SIZE,
        context_length=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Split text into sentences for sentence-level alignment
    sentences = split_into_sentences(text)
    if not sentences:
        sentences = [text]

    # Preprocess full text for alignment (romanization + tokenization)
    # We align the full text at once, then map timestamps back to sentences
    full_text = " ".join(sentences)
    tokens_starred, text_starred = preprocess_text(
        full_text,
        romanize=True,
        language=LANGUAGE,
    )

    if not tokens_starred:
        print(f"    ⚠️  No alignable tokens for {output_prefix}")
        return []

    # Run forced alignment (using fixed version for HF tokenizer compat)
    segments, scores, blank_token = get_alignments_fixed(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    # Get word-level timestamps
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    if not word_timestamps:
        print(f"    ⚠️  No word timestamps for {output_prefix}")
        return []

    # Read original audio for slicing
    audio_data, sr = sf.read(audio_path)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"

    # Map word timestamps back to sentence boundaries
    # Build a list of (sentence_text, start_time, end_time, avg_score) tuples
    sentence_segments = _map_words_to_sentences(
        sentences, full_text, word_timestamps
    )

    # Slice audio for each sentence segment
    for i, (sent_text, start_sec, end_sec, avg_score) in enumerate(sentence_segments):
        duration = end_sec - start_sec

        # Filter by duration
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue

        # Filter by word count
        word_count = len(sent_text.split())
        if word_count < MIN_WORDS:
            continue

        # Filter by alignment quality
        if avg_score < MIN_SCORE:
            continue

        # Extract audio segment
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        # Clamp to audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        segment_audio = audio_data[start_sample:end_sample]

        if len(segment_audio) < int(MIN_DURATION * sr):
            continue

        # Save segment
        seg_filename = f"{output_prefix}_{i:04d}.wav"
        seg_path = SEGMENTS_DIR / seg_filename
        sf.write(seg_path, segment_audio, sr)

        segments_out.append(
            {
                "audio": str(seg_path),
                "text": sent_text,
                "duration": round(duration, 2),
                "score": round(avg_score, 2),
            }
        )

    return segments_out


def _map_words_to_sentences(
    sentences: list[str],
    full_text: str,
    word_timestamps: list,
) -> list[tuple[str, float, float, float]]:
    """
    Map word-level timestamps back to sentence boundaries.

    Returns: list of (sentence_text, start_sec, end_sec, avg_score)
    """
    result = []

    # Build word list from full_text to match alignment output
    full_words = full_text.split()
    total_aligned_words = len(word_timestamps)

    if total_aligned_words == 0:
        return result

    # Track position in the aligned word list
    word_idx = 0

    for sentence in sentences:
        sent_words = sentence.split()
        n_words = len(sent_words)

        if word_idx >= total_aligned_words:
            break

        # Determine word range for this sentence
        start_word_idx = word_idx
        end_word_idx = min(word_idx + n_words, total_aligned_words)

        # Get the word timestamps for this sentence
        sent_word_ts = word_timestamps[start_word_idx:end_word_idx]

        if not sent_word_ts:
            word_idx = end_word_idx
            continue

        # Extract start/end times and scores
        # word_timestamps are dicts with 'start', 'end' in seconds, 'score' as log-prob
        starts = []
        ends = []
        scores_list = []
        for wt in sent_word_ts:
            starts.append(wt["start"])
            ends.append(wt["end"])
            scores_list.append(wt.get("score", 0))

        if starts and ends:
            start_sec = min(starts)  # already in seconds
            end_sec = max(ends)
            avg_score = sum(scores_list) / len(scores_list) if scores_list else 0
            result.append((sentence, start_sec, end_sec, avg_score))

        word_idx = end_word_idx

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load metadata
    entries = load_metadata()
    print(f"📋 {len(entries)} entries to process")

    # Load alignment model
    print(f"🔄 Loading MMS forced alignment model on {device}...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=dtype,
    )
    print("✅ Alignment model loaded")

    # Process each audio file
    all_segments = []
    for idx, entry in enumerate(entries):
        audio_path = DATASET_DIR / entry["audio_file"]
        text_path = DATASET_DIR / entry["text_file"]
        video_id = entry["id"]

        if not audio_path.exists():
            print(f"  ⚠️  Missing audio: {audio_path}")
            continue

        if not text_path.exists():
            print(f"  ⚠️  Missing text: {text_path}")
            continue

        # Read ground truth text
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = clean_text(text)

        if not text or len(text) < 20:
            print(f"  ⚠️  Text too short for {video_id}")
            continue

        print(f"[{idx + 1}/{len(entries)}] {entry['title'][:60]}...")

        try:
            segs = align_and_segment(
                alignment_model,
                alignment_tokenizer,
                audio_path,
                text,
                video_id,
                device,
                dtype,
            )
            all_segments.extend(segs)
            print(f"  → {len(segs)} segments")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print(f"\n✅ Total segments: {len(all_segments)}")

    if not all_segments:
        print("⚠️  No segments produced. Exiting.")
        sys.exit(1)

    # Stats
    total_dur = sum(s["duration"] for s in all_segments)
    avg_dur = total_dur / len(all_segments)
    avg_score = sum(s["score"] for s in all_segments) / len(all_segments)
    print(f"📊 Total duration: {total_dur / 3600:.1f}h | Avg: {avg_dur:.1f}s | Avg score: {avg_score:.2f}")

    # Save metadata JSONL
    with open(DATASET_DIR / "segments_metadata.jsonl", "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
