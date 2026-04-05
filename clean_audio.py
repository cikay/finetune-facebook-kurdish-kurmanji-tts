#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path

import torch
import torchaudio

from demucs.pretrained import get_model
from demucs.apply import apply_model

AUDIO_DIR = Path("dataset/audio")
CLEAN_DIR = Path("dataset/clean_audio")


def main():
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"📋 {len(wav_files)} audio files to clean")

    if not wav_files:
        print("⚠️  No WAV files found")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load model once
    model = get_model(name="htdemucs")
    model.to(device)
    model.eval()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for wav_file in wav_files:
        print(f"🔄 Processing {wav_file.name}")

        wav, sr = torchaudio.load(wav_file)

        # Convert to stereo if needed (Demucs expects 2 channels)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        wav = wav.to(device)

        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device)[0]

        # Demucs order: [drums, bass, other, vocals]
        vocals = sources[3].cpu()

        out_path = CLEAN_DIR / wav_file.name
        torchaudio.save(out_path, vocals, sr)

    print(f"\n✅ Done! Cleaned files in {CLEAN_DIR}")


if __name__ == "__main__":
    main()
