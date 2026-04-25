#!/usr/bin/env python3
"""
Preprocess long Kurdish Kurmanji audio into short utterance-level segments
using CTC forced alignment with the MMS model.

Unlike the Whisper-based approach, this aligns the *ground truth* text to the
audio, producing accurate timestamps without ASR errors.

Usage:
   python segmentation.py --audio-subdir audio
"""

from collections import Counter
import json
import re
import sys
import unicodedata
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
from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score

# ── Config ───────────────────────────────────────────────────────────────────
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
END_PADDING = 0.15  # seconds of silence padding added after each segment end

# Regex for splitting text into sentences
# Keeps ending punctuation (.!?) with each sentence
SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+")
SENTENCE_LIKE_RE = re.compile(r"[^;:]*[;:]+|[^;:]+")


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
        dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary
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

# Abbreviation / initialism patterns
# - Matches dotted initialisms like "D.Y.A." or "A.B.D."
# - Matches short title-style abbreviations like "Dr.", "Mr.", "Prof."
# - Matches all-caps initialisms like "DYA", "PDK", "UNESCO"
# - Uses a lookahead on the dotted branch to avoid matching normal sentence-final
#   punctuation like "Ahmed."
ABBR_RE = re.compile(
    r"(?:"
    r"\b(?:(?:[A-ZÇĞÎÛŞ]\.){2,}|[A-ZÇĞÎÛŞ][a-zçğıîûş]{1,3}\.)(?=\s*[A-Za-zÇĞÎÛŞçğıîûş])"
    r"|"
    r"\b[A-ZÇĞÎÛŞ]{2,}\b"
    r")"
)
DIGIT_RE = re.compile(r"\d")


def normalize_text(text: str) -> str:
    """Normalize whitespace and light cleanup for Kurdish Kurmanji text."""
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


# ── Audio quality metrics ────────────────────────────────────────────────────


def calculate_dns_mos(waveform: np.ndarray, device: str = "cpu") -> dict[str, float]:
    """
    Calculate Deep Noise Suppression Mean Opinion Score for an audio waveform.

    Args:
        waveform: audio waveform as numpy array
        device: device to use for computation ("cpu" or "cuda")

    Returns:
        dict with keys: p808_mos, mos_sig, mos_bak, mos_ovr
    """
    # Convert numpy array to torch tensor
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    # Ensure waveform is on the correct device for computation
    waveform = waveform.to(device)

    # Calculate DNS MOS (returns [p808_mos, mos_sig, mos_bak, mos_ovr])
    dns_scores = deep_noise_suppression_mean_opinion_score(
        preds=waveform,
        fs=SAMPLE_RATE,
        personalized=False,
        device=device,
    )

    # Convert to dict for easier access
    score_dict = {
        "p808_mos": round(float(dns_scores[0]), 2),
        "mos_sig": round(float(dns_scores[1]), 2),
        "mos_bak": round(float(dns_scores[2]), 2),
        "mos_ovr": round(float(dns_scores[3]), 2),
    }
    return score_dict


# ── Core pipeline ────────────────────────────────────────────────────────────


def load_metadata(meta_file: Path) -> list[dict]:
    """Load metadata.jsonl entries."""
    entries = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def should_discard(
    sentence: str, duration: float, align_score: float
) -> tuple[bool, str]:
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return True, "duration"
    elif len(sentence.split()) < MIN_WORDS:
        return True, "too_few_words"
    # Filter by alignment quality
    elif align_score < MIN_SCORE:
        return True, "low_score"
    elif bool(ABBR_RE.search(sentence)):
        return True, "abbreviations"
    elif bool(DIGIT_RE.search(sentence)):
        return True, "digits"

    return False, ""


def _process_sentence_segment(
    chunk_text: str,
    start_sec: float,
    end_sec: float,
    align_score: float,
    audio_data,
    sr: int,
    device: str,
    segments_dir: Path,
    output_prefix: str,
    seg_idx: int,
) -> tuple[list[dict], Counter, int]:
    """
    Filter one chunk and, if kept, extract and save its audio.

    Returns (new_segments, discard_counts, updated_seg_idx).
    """
    segments_out = []
    discard_counts: Counter = Counter()
    duration = end_sec - start_sec

    should_discard_result, reason = should_discard(chunk_text, duration, align_score)
    if should_discard_result:
        print(f"    ⚠️  Discarding segment: '{chunk_text}' | Reason: {reason}")
        discard_counts[reason] += 1
        return segments_out, discard_counts, seg_idx

    print(
        f"    ✅ Keeping segment: '{chunk_text}' | Duration: {duration:.1f}s | Align Score: {align_score:.2f}"
    )
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(len(audio_data), int((end_sec + END_PADDING) * sr))
    segment_audio = audio_data[start_sample:end_sample]

    if len(segment_audio) < int(MIN_DURATION * sr):
        return segments_out, discard_counts, seg_idx

    dns_mos_scores = calculate_dns_mos(segment_audio, device)

    seg_filename = f"{output_prefix}_{seg_idx:04d}.wav"
    seg_path = segments_dir / seg_filename
    sf.write(seg_path, segment_audio, sr)

    segments_out.append(
        {
            "audio": str(seg_path),
            "text": chunk_text,
            "duration": round(duration, 2),
            "align_score": round(align_score, 2),
            "dns_mos": dns_mos_scores,
            "word_count": len(chunk_text.split()),
        }
    )
    return segments_out, discard_counts, seg_idx + 1


def align_and_segment(
    alignment_model,
    alignment_tokenizer,
    audio_path: Path,
    text: str,
    output_prefix: str,
    segments_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[dict], dict[str, int]]:
    """
    Perform forced alignment of ground truth text against audio,
    then split into sentence-level segments.

    Returns:
      - list of {audio, text, duration, score} dicts
      - discard reason counts for this source item
    """
    segments_out = []
    discard_counts = Counter()

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
    print(f"    Text split into {len(sentences)} sentences")
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
        return [], dict(discard_counts)

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
        return [], dict(discard_counts)

    # Read original audio for slicing
    audio_data, sr = sf.read(audio_path)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"

    chunks = _map_words_to_sentence_like_chunks(sentences, word_timestamps)

    seg_idx = 0
    for chunk_text, start_sec, end_sec, align_score in chunks:
        new_segs, chunk_discards, seg_idx = _process_sentence_segment(
            chunk_text,
            start_sec,
            end_sec,
            align_score,
            audio_data,
            sr,
            device,
            segments_dir,
            output_prefix,
            seg_idx,
        )
        segments_out.extend(new_segs)
        discard_counts.update(chunk_discards)

    return segments_out, dict(discard_counts)


def _map_words_to_sentence_like_chunks(
    sentences: list[str],
    word_timestamps: list,
) -> list[tuple[str, float, float, float]]:
    """
    Map word-level timestamps to sentence boundaries, then sub-split any sentence
    exceeding MAX_DURATION by -, ;, : into smaller chunks.

    Returns: list of (chunk_text, start_sec, end_sec, align_score)
    """
    result = []
    total_aligned_words = len(word_timestamps)

    if total_aligned_words == 0:
        return result

    word_idx = 0
    for sentence in sentences:
        n_words = len(sentence.split())

        if word_idx >= total_aligned_words:
            break

        end_word_idx = min(word_idx + n_words, total_aligned_words)
        sent_word_ts = word_timestamps[word_idx:end_word_idx]

        if not sent_word_ts:
            word_idx = end_word_idx
            continue

        starts = [wt["start"] for wt in sent_word_ts]
        ends = [wt["end"] for wt in sent_word_ts]
        scores = [wt.get("score", 0) for wt in sent_word_ts]
        start_sec = min(starts)
        end_sec = max(ends)
        align_score = sum(scores) / len(scores)
        duration = end_sec - start_sec

        if duration <= MAX_DURATION:
            result.append((sentence, start_sec, end_sec, align_score))
            word_idx = end_word_idx
            continue

        sub_chunks = [c.strip() for c in SENTENCE_LIKE_RE.findall(sentence) if c.strip()]
        if len(sub_chunks) <= 1:
            # this will be discarded in _process_sentence_segment
            result.append((sentence, start_sec, end_sec, align_score))
            word_idx = end_word_idx
            continue

        sub_word_idx = 0
        for chunk in sub_chunks:
            chunk_n_words = len(chunk.split())
            chunk_end = min(sub_word_idx + chunk_n_words, len(sent_word_ts))
            chunk_wts = sent_word_ts[sub_word_idx:chunk_end]
            if chunk_wts:
                chunk_scores = [wt.get("score", 0) for wt in chunk_wts]
                result.append(
                    (
                        chunk,
                        min(wt["start"] for wt in chunk_wts),
                        max(wt["end"] for wt in chunk_wts),
                        sum(chunk_scores) / len(chunk_scores),
                    )
                )
            sub_word_idx = chunk_end
        word_idx = end_word_idx

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def run_segmentation(input_dirs: dict[str, Path], output_dirs: dict[str, Path]) -> None:
    audio_dir = Path(input_dirs["audio"]).expanduser()
    text_dir = Path(input_dirs["text"]).expanduser()
    meta_file = Path(input_dirs["metadata"]).expanduser()
    segments_dir = Path(output_dirs["audio_segments"]).expanduser()
    segments_meta_file = Path(output_dirs["metadata"]).expanduser()

    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"⚠️  Invalid audio directory: {audio_dir}")
        sys.exit(1)
    if not text_dir.exists() or not text_dir.is_dir():
        print(f"⚠️  Invalid text directory: {text_dir}")
        sys.exit(1)
    if not meta_file.exists() or not meta_file.is_file():
        print(f"⚠️  Invalid metadata file: {meta_file}")
        sys.exit(1)

    segments_dir.mkdir(parents=True, exist_ok=True)
    segments_meta_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using audio directory: {audio_dir}")

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load metadata
    entries = load_metadata(meta_file)
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
    total_discard_counts = Counter()
    for idx, entry in enumerate(entries):
        audio_name = Path(entry["audio_file"]).name
        text_name = Path(entry["text_file"]).name
        audio_path = audio_dir / audio_name
        text_path = text_dir / text_name
        video_id = entry["id"]

        if not audio_path.exists():
            print(f"  ⚠️  Missing audio: {audio_path}")
            continue

        if not text_path.exists():
            print(f"  ⚠️  Missing text: {text_path}")
            continue

        # Read ground truth text
        with open(text_path, "r", encoding="utf-8") as f:
            text = normalize_text(f.read())

        if not text or len(text) < 20:
            print(f"  ⚠️  Text too short for {video_id}")
            continue

        print(f"[{idx + 1}/{len(entries)}] {entry['title'][:60]}...")

        try:
            segs, discard_counts = align_and_segment(
                alignment_model,
                alignment_tokenizer,
                audio_path,
                text,
                video_id,
                segments_dir,
                device,
                dtype,
            )
            all_segments.extend(segs)
            total_discard_counts.update(discard_counts)
            print(f"  → {len(segs)} segments")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print(f"\n✅ Total segments: {len(all_segments)}")
    total_discarded = sum(total_discard_counts.values())
    print(f"🗑️  Total discarded: {total_discarded}")
    if total_discard_counts:
        print("🧾 Discard reasons:")
        for reason, count in sorted(total_discard_counts.items()):
            print(f"   - {reason}: {count}")

    if not all_segments:
        print("⚠️  No segments produced. Exiting.")
        sys.exit(1)

    # Stats
    total_dur = sum(s["duration"] for s in all_segments)
    avg_dur = total_dur / len(all_segments)
    align_score = sum(s["align_score"] for s in all_segments) / len(all_segments)
    avg_word_count = sum(s["word_count"] for s in all_segments) / len(all_segments)
    print(
        f"📊 Total duration: {total_dur / 3600:.1f}h | Avg duration: {avg_dur:.1f}s | Avg Align score: {align_score:.2f} | Avg word count: {avg_word_count:.1f}"
    )

    # Save metadata JSONL
    with open(segments_meta_file, "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    print("✅ Preprocessing complete!")


class SegmentationBlock:
    name: str = "segmentation"

    def __init__(
        self, input_dirs: dict[str, Path], output_dirs: dict[str, Path]
    ) -> None:
        self.name = "segmentation"
        self.input_dirs = input_dirs
        self.output_dirs = output_dirs

    def run(self) -> None:
        required_inputs = {"audio", "text", "metadata"}
        required_outputs = {"audio_segments", "metadata"}

        missing_inputs = required_inputs - set(self.input_dirs)
        missing_outputs = required_outputs - set(self.output_dirs)
        if missing_inputs:
            raise ValueError(
                f"Missing segmentation input_dirs keys: {sorted(missing_inputs)}"
            )
        if missing_outputs:
            raise ValueError(
                f"Missing segmentation output_dirs keys: {sorted(missing_outputs)}"
            )

        run_segmentation(self.input_dirs, self.output_dirs)
