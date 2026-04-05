#!/usr/bin/env python3

import sys
from pathlib import Path

import soundfile as sf
import numpy as np
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

    model = get_model(name="htdemucs")
    model.to(device)
    model.eval()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for wav_file in wav_files:
        print(f"🔄 Processing {wav_file.name}")

        data, sr = sf.read(wav_file)

        # data shape: (samples,) mono veya (samples, channels) stereo
        if data.ndim == 1:
            data = np.stack([data, data], axis=0)  # (2, samples)
        else:
            data = data.T  # (channels, samples)
            if data.shape[0] == 1:
                data = np.concatenate([data, data], axis=0)  # (2, samples)

        wav = torch.tensor(data, dtype=torch.float32).to(device)

        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0), device=device)[0]

        vocals = sources[3].cpu().numpy().T  # (samples, 2)

        out_path = CLEAN_DIR / wav_file.name
        sf.write(out_path, vocals, sr)

    print(f"\n✅ Done! Cleaned files in {CLEAN_DIR}")


if __name__ == "__main__":
    main()
