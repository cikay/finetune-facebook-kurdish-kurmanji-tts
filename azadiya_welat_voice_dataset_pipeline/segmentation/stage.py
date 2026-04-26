from collections import Counter
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Union

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

SAMPLE_RATE = 16000

SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+")
SENTENCE_LIKE_RE = re.compile(r"[^;:]*[;:]+|[^;:]+")

ABBR_RE = re.compile(
    r"(?:"
    r"\b(?:(?:[A-ZÇĞÎÛŞ]\.){2,}|[A-ZÇĞÎÛŞ][a-zçğıîûş]{1,3}\.)(?=\s*[A-Za-zÇĞÎÛŞçğıîûş])"
    r"|"
    r"\b[A-ZÇĞÎÛŞ]{2,}\b"
    r")"
)
DIGIT_RE = re.compile(r"\d")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


class SegmentationStage:
    name = "segmentation"

    def __init__(self, config: dict) -> None:
        self.input_dir = Path(config["input_dir"])
        self.output_dir = Path(config["output_dir"])
        self.language = config.get("align_language", "kmr")
        self.min_duration = float(config.get("min_duration", 2.0))
        self.max_duration = float(config.get("max_duration", 15.0))
        self.min_words = int(config.get("min_words", 3))
        self.min_align_score = float(config.get("min_align_score", -7.0))
        self.end_padding = float(config.get("end_padding", 0.15))
        self.batch_size = int(config.get("batch_size", 16))
        self.window_size = int(config.get("window_size", 30))
        self.context_size = int(config.get("context_size", 2))

    def run(self) -> None:
        audio_dir = self.input_dir / "audio"
        text_dir = self.input_dir / "text"
        meta_file = self.input_dir / "metadata.jsonl"
        segments_dir = self.output_dir / "audio_segments"
        segments_meta_file = self.output_dir / "metadata.jsonl"

        self._validate_paths(audio_dir, text_dir, meta_file)

        segments_dir.mkdir(parents=True, exist_ok=True)
        segments_meta_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Using audio directory: {audio_dir}")

        device, dtype = self._resolve_device()
        entries = self._load_metadata(meta_file)
        print(f"📋 {len(entries)} entries to process")

        alignment_model, alignment_tokenizer = self._load_alignment_model(device, dtype)

        all_segments, total_discard_counts = self._segment_entries(
            entries,
            alignment_model,
            alignment_tokenizer,
            audio_dir,
            text_dir,
            segments_dir,
            device,
            dtype,
        )

        self._print_run_stats(all_segments, total_discard_counts)
        self._save_segments_metadata(all_segments, segments_meta_file)

    def _validate_paths(self, audio_dir: Path, text_dir: Path, meta_file: Path) -> None:
        if not audio_dir.exists() or not audio_dir.is_dir():
            print(f"⚠️  Invalid audio directory: {audio_dir}")
            sys.exit(1)
        if not text_dir.exists() or not text_dir.is_dir():
            print(f"⚠️  Invalid text directory: {text_dir}")
            sys.exit(1)
        if not meta_file.exists() or not meta_file.is_file():
            print(f"⚠️  Invalid metadata file: {meta_file}")
            sys.exit(1)

    def _resolve_device(self) -> tuple[str, torch.dtype]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Using device: {device}")
        return device, dtype

    def _load_alignment_model(self, device: str, dtype: torch.dtype):
        print(f"🔄 Loading MMS forced alignment model on {device}...")
        model, tokenizer = load_alignment_model(device, dtype=dtype)
        print("✅ Alignment model loaded")
        return model, tokenizer

    def _load_metadata(self, meta_file: Path) -> list[dict]:
        entries = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def _segment_entries(
        self,
        entries: list[dict],
        alignment_model,
        alignment_tokenizer,
        audio_dir: Path,
        text_dir: Path,
        segments_dir: Path,
        device: str,
        dtype: torch.dtype,
    ) -> tuple[list[dict], Counter]:
        all_segments = []
        total_discard_counts = Counter()

        for idx, entry in enumerate(entries):
            audio_path = audio_dir / Path(entry["audio_file"]).name
            text_path = text_dir / Path(entry["text_file"]).name
            video_id = entry["id"]

            if not audio_path.exists():
                print(f"  ⚠️  Missing audio: {audio_path}")
                continue
            if not text_path.exists():
                print(f"  ⚠️  Missing text: {text_path}")
                continue

            text = self._read_entry_text(text_path)
            if text is None:
                print(f"  ⚠️  Text too short for {video_id}")
                continue

            print(f"[{idx + 1}/{len(entries)}] {entry['title'][:60]}...")

            try:
                segments, discard_counts = self._segment_entry(
                    alignment_model,
                    alignment_tokenizer,
                    audio_path,
                    text,
                    video_id,
                    segments_dir,
                    device,
                    dtype,
                )
                all_segments.extend(segments)
                total_discard_counts.update(discard_counts)
                print(f"  → {len(segments)} segments")
            except Exception as e:
                print(f"  ❌ Error: {e}")

        return all_segments, total_discard_counts

    def _read_entry_text(self, text_path: Path) -> str | None:
        with open(text_path, "r", encoding="utf-8") as f:
            text = normalize_text(f.read())
        return text if text and len(text) >= 20 else None

    def _segment_entry(
        self,
        alignment_model,
        alignment_tokenizer,
        audio_path: Path,
        text: str,
        output_prefix: str,
        segments_dir: Path,
        device: str,
        dtype: torch.dtype,
    ) -> tuple[list[dict], dict[str, int]]:
        discard_counts = Counter()

        audio_waveform = load_audio(str(audio_path), dtype, device)
        emissions, stride = generate_emissions(
            alignment_model,
            audio_waveform,
            window_length=self.window_size,
            context_length=self.context_size,
            batch_size=self.batch_size,
        )

        sentences, tokens_starred, text_starred = self._prepare_text(text)
        if not tokens_starred:
            print(f"    ⚠️  No alignable tokens for {output_prefix}")
            return [], dict(discard_counts)

        word_timestamps = self._run_alignment(
            emissions, tokens_starred, alignment_tokenizer, text_starred, stride
        )
        if not word_timestamps:
            print(f"    ⚠️  No word timestamps for {output_prefix}")
            return [], dict(discard_counts)

        audio_data, sr = sf.read(audio_path)
        assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"

        chunks = self._map_words_to_sentence_like_chunks(sentences, word_timestamps)

        segments_out = []
        seg_idx = 0
        for chunk_text, start_sec, end_sec, align_score in chunks:
            new_seg, chunk_discards, seg_idx = self._build_segment(
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
            if new_seg:
                segments_out.append(new_seg)
            discard_counts.update(chunk_discards)

        return segments_out, dict(discard_counts)

    def _prepare_text(self, text: str) -> tuple[list[str], list, str]:
        sentences = split_into_sentences(text)
        if not sentences:
            sentences = [text]
        print(f"    Text split into {len(sentences)} sentences")
        full_text = " ".join(sentences)
        tokens_starred, text_starred = preprocess_text(
            full_text, romanize=True, language=self.language
        )
        return sentences, tokens_starred, text_starred

    def _run_alignment(
        self,
        emissions: torch.Tensor,
        tokens_starred: list,
        alignment_tokenizer,
        text_starred: str,
        stride,
    ) -> list:
        segments, scores, blank_token = self._get_alignments_fixed(
            emissions, tokens_starred, alignment_tokenizer
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        return postprocess_results(text_starred, spans, stride, scores)

    def _get_alignments_fixed(self, emissions: torch.Tensor, tokens: list, tokenizer):
        """
        Like ctc_forced_aligner.get_alignments but fixes the <star> token index.

        The HuggingFace tokenizer has extra special tokens that inflate the vocab
        size. generate_emissions adds the star column at the last emissions index,
        so <star> must map there, not to len(vocab).
        """
        assert len(tokens) > 0, "Empty transcript"

        dictionary = tokenizer.get_vocab()
        dictionary = {k.lower(): v for k, v in dictionary.items()}
        dictionary["<star>"] = emissions.shape[-1] - 1

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

    def _sentence_time_span(self, word_ts: list) -> tuple[float, float, float]:
        starts = [wt["start"] for wt in word_ts]
        ends = [wt["end"] for wt in word_ts]
        scores = [wt.get("score", 0) for wt in word_ts]
        return min(starts), max(ends), sum(scores) / len(scores)

    def _sub_split_sentence(
        self, sentence: str, sent_word_ts: list
    ) -> list[tuple[str, float, float, float]]:
        sub_chunks = [
            c.strip() for c in SENTENCE_LIKE_RE.findall(sentence) if c.strip()
        ]
        if len(sub_chunks) <= 1:
            return []

        result = []
        sub_word_idx = 0
        for chunk in sub_chunks:
            chunk_end = min(sub_word_idx + len(chunk.split()), len(sent_word_ts))
            chunk_wts = sent_word_ts[sub_word_idx:chunk_end]
            if chunk_wts:
                start, end, score = self._sentence_time_span(chunk_wts)
                result.append((chunk, start, end, score))
            sub_word_idx = chunk_end
        return result

    def _map_words_to_sentence_like_chunks(
        self,
        sentences: list[str],
        word_timestamps: list,
    ) -> list[tuple[str, float, float, float]]:
        result = []
        total_aligned_words = len(word_timestamps)
        if total_aligned_words == 0:
            return result

        word_idx = 0
        for sentence in sentences:
            if word_idx >= total_aligned_words:
                break

            n_words = len(sentence.split())
            end_word_idx = min(word_idx + n_words, total_aligned_words)
            sent_word_ts = word_timestamps[word_idx:end_word_idx]

            if not sent_word_ts:
                word_idx = end_word_idx
                continue

            start_sec, end_sec, align_score = self._sentence_time_span(sent_word_ts)

            if end_sec - start_sec <= self.max_duration:
                result.append((sentence, start_sec, end_sec, align_score))
                word_idx = end_word_idx
                continue

            sub_chunks = self._sub_split_sentence(sentence, sent_word_ts)
            result.extend(
                sub_chunks
                if sub_chunks
                else [(sentence, start_sec, end_sec, align_score)]
            )
            word_idx = end_word_idx

        return result

    def _should_discard(
        self, sentence: str, duration: float, align_score: float
    ) -> tuple[bool, str]:
        if duration < self.min_duration or duration > self.max_duration:
            return True, "duration"
        elif len(sentence.split()) < self.min_words:
            return True, "too_few_words"
        elif align_score < self.min_align_score:
            return True, "low_score"
        elif bool(ABBR_RE.search(sentence)):
            return True, "abbreviations"
        elif bool(DIGIT_RE.search(sentence)):
            return True, "digits"
        return False, ""

    def _extract_audio_slice(
        self, audio_data, sr: int, start_sec: float, end_sec: float
    ) -> np.ndarray:
        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(audio_data), int((end_sec + self.end_padding) * sr))
        return audio_data[start_sample:end_sample]

    def _calculate_dns_mos(self, waveform: np.ndarray, device: str) -> dict[str, float]:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        waveform = waveform.to(device)
        dns_scores = deep_noise_suppression_mean_opinion_score(
            preds=waveform,
            fs=SAMPLE_RATE,
            personalized=False,
            device=device,
        )
        return {
            "p808_mos": round(float(dns_scores[0]), 2),
            "mos_sig": round(float(dns_scores[1]), 2),
            "mos_bak": round(float(dns_scores[2]), 2),
            "mos_ovr": round(float(dns_scores[3]), 2),
        }

    def _build_segment(
        self,
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
    ) -> tuple[Union[dict, None], Counter, int]:
        discard_counts: Counter = Counter()
        duration = end_sec - start_sec

        should_discard_result, reason = self._should_discard(
            chunk_text, duration, align_score
        )
        if should_discard_result:
            print(f"    ⚠️  Discarding segment: '{chunk_text}' | Reason: {reason}")
            discard_counts[reason] += 1
            return None, discard_counts, seg_idx

        print(
            f"    ✅ Keeping segment: '{chunk_text}' | Duration: {duration:.1f}s | Align Score: {align_score:.2f}"
        )

        segment_audio = self._extract_audio_slice(audio_data, sr, start_sec, end_sec)
        if len(segment_audio) < int(self.min_duration * sr):
            return None, discard_counts, seg_idx

        dns_mos_scores = self._calculate_dns_mos(segment_audio, device)

        seg_path = segments_dir / f"{output_prefix}_{seg_idx:04d}.wav"
        sf.write(seg_path, segment_audio, sr)

        segment = {
            "audio": str(seg_path),
            "text": chunk_text,
            "duration": round(duration, 2),
            "align_score": round(align_score, 2),
            "dns_mos": dns_mos_scores,
            "word_count": len(chunk_text.split()),
        }
        return segment, discard_counts, seg_idx + 1

    def _print_run_stats(
        self, all_segments: list[dict], discard_counts: Counter
    ) -> None:
        if not all_segments:
            print("⚠️  No segments produced. Exiting.")
            sys.exit(1)

        print(f"\n✅ Total segments: {len(all_segments)}")
        total_discarded = sum(discard_counts.values())
        print(f"🗑️  Total discarded: {total_discarded}")
        if discard_counts:
            print("🧾 Discard reasons:")
            for reason, count in sorted(discard_counts.items()):
                print(f"   - {reason}: {count}")

        total_dur = sum(s["duration"] for s in all_segments)
        avg_dur = total_dur / len(all_segments)
        avg_score = sum(s["align_score"] for s in all_segments) / len(all_segments)
        avg_words = sum(s["word_count"] for s in all_segments) / len(all_segments)
        print(
            f"📊 Total duration: {total_dur / 3600:.1f}h | Avg duration: {avg_dur:.1f}s"
            f" | Avg Align score: {avg_score:.2f} | Avg word count: {avg_words:.1f}"
        )

    def _save_segments_metadata(self, all_segments: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for seg in all_segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        print("✅ Segmentation complete!")
