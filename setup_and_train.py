#!/usr/bin/env python3
"""
Fine-tune MMS-TTS-KMR (Kurdish Kurmanji) — end-to-end pipeline.

Run on a machine with a GPU. Assumes dataset/ folder with audio+text is present.

Steps:
    1. Preprocess audio into short segments (calls preprocess.py)
    2. Clone & setup finetune-hf-vits
    3. Convert discriminator checkpoint for MMS-KMR
    4. Launch training

Usage:
    python setup_and_train.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VITS_REPO = PROJECT_DIR / "finetune-hf-vits"
LANGUAGE_CODE = "kmr-script_latin"
TRAIN_MODEL_DIR = PROJECT_DIR / "mms-tts-kmr-train"
HF_DATASET_DIR = PROJECT_DIR / "processed_dataset" / "hf_dataset"
CONFIG_FILE = PROJECT_DIR / "training_config.json"


def run(cmd: list[str], cwd: Path | None = None):
    """Run a command, stream output, raise on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"❌ Command failed with code {result.returncode}")


def pip_install(*packages: str):
    run([sys.executable, "-m", "pip", "install", *packages])


def step1_preprocess():
    """Segment long audio into short utterances using Whisper."""
    if HF_DATASET_DIR.exists():
        print("⏭  Preprocessed dataset already exists, skipping.\n")
        return

    print("▶ Installing preprocessing dependencies...")
    pip_install("faster-whisper", "soundfile", "datasets")

    print("▶ Running preprocess.py...")
    run([sys.executable, str(PROJECT_DIR / "preprocess.py")], cwd=PROJECT_DIR)
    print()


def step2_setup_vits_repo():
    """Clone finetune-hf-vits, install deps, build monotonic_align."""
    if VITS_REPO.exists():
        print("⏭  finetune-hf-vits already cloned, skipping.\n")
        return

    print("▶ Cloning finetune-hf-vits...")
    run(["git", "clone", "https://github.com/ylacombe/finetune-hf-vits.git"], cwd=PROJECT_DIR)

    print("▶ Installing training dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=VITS_REPO)

    print("▶ Building monotonic_align (Cython)...")
    mono_dir = VITS_REPO / "monotonic_align"
    (mono_dir / "monotonic_align").mkdir(exist_ok=True)
    run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=mono_dir)
    print()


def step3_convert_discriminator():
    """Convert MMS checkpoint to include discriminator for GAN training."""
    if TRAIN_MODEL_DIR.exists():
        print("⏭  Training checkpoint already exists, skipping.\n")
        return

    print(f"▶ Converting discriminator for {LANGUAGE_CODE}...")
    run(
        [
            sys.executable, "convert_original_discriminator_checkpoint.py",
            "--language_code", LANGUAGE_CODE,
            "--pytorch_dump_folder_path", str(TRAIN_MODEL_DIR),
        ],
        cwd=VITS_REPO,
    )
    print()


def step4_train():
    """Launch VITS fine-tuning with accelerate."""
    print("▶ Starting training...")
    print(f"  Config: {CONFIG_FILE}")
    print(f"  Model:  {TRAIN_MODEL_DIR}")
    print(f"  Data:   {HF_DATASET_DIR}")
    print()

    run(
        [sys.executable, "-m", "accelerate.commands.launch", "run_vits_finetuning.py", str(CONFIG_FILE)],
        cwd=VITS_REPO,
    )


def main():
    print("═" * 50)
    print("  MMS-TTS Kurdish Kurmanji Fine-tuning Pipeline")
    print("═" * 50)
    print()

    step1_preprocess()
    step2_setup_vits_repo()
    step3_convert_discriminator()
    step4_train()

    print()
    print("═" * 50)
    print(f"  ✅ Training complete!")
    print(f"  Checkpoints: {PROJECT_DIR / 'checkpoints'}")
    print("═" * 50)


if __name__ == "__main__":
    main()
