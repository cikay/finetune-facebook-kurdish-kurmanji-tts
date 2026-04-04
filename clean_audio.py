#!/usr/bin/env python3
"""
Remove background music from all audio files using Demucs.

Processes all WAV files in dataset/audio/, extracts the vocals (speech) track,
and saves them with the same filename in dataset/clean_audio/.

Usage:
    pipenv run python clean_audio.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_DIR = Path("dataset/audio")
DEMUCS_OUTPUT = Path("dataset/demucs_raw")
CLEAN_DIR = Path("dataset/clean_audio")


def main():
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"📋 {len(wav_files)} audio files to clean")

    if not wav_files:
        print("⚠️  No WAV files found in dataset/audio/")
        sys.exit(1)

    # Run Demucs on all files
    print(f"🔄 Running Demucs (vocals separation)...")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", str(DEMUCS_OUTPUT),
    ] + [str(f) for f in wav_files]

    subprocess.run(cmd, check=True)

    # Flatten: copy vocals.wav files to clean_audio/ with original filenames
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = DEMUCS_OUTPUT / "htdemucs"

    copied = 0
    for wav_file in wav_files:
        stem = wav_file.stem
        vocals_path = model_dir / stem / "vocals.wav"
        if vocals_path.exists():
            dest = CLEAN_DIR / f"{stem}.wav"
            shutil.copy2(vocals_path, dest)
            copied += 1
        else:
            print(f"  ⚠️  Missing vocals for {stem}")

    print(f"\n✅ {copied} cleaned files saved to {CLEAN_DIR}/")

    # Clean up Demucs intermediate output
    print(f"🧹 Removing intermediate Demucs output...")
    shutil.rmtree(DEMUCS_OUTPUT)
    print("✅ Done!")


if __name__ == "__main__":
    main()
