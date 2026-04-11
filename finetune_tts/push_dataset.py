#!/usr/bin/env python3
"""
Push the segmented Kurdish Kurmanji TTS dataset to HuggingFace Hub.

Reads segments_metadata.jsonl, builds a HuggingFace Dataset with audio + text,
and pushes it to your HuggingFace account.

Usage:
    pipenv run python push_dataset.py
    pipenv run python push_dataset.py --repo muzaffercky/kurdish-kurmanji-tts --private
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from finetune_tts.load_dataset_from_local import load_dataset

load_dotenv()

METADATA_FILE = Path("dataset/segments_metadata.jsonl")
SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(description="Push segmented dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo",
        type=str,
        # default="muzaffercky/azadiya-welat-tts",
        help="HuggingFace repo ID",
    )
    args = parser.parse_args()

    dataset = load_dataset()

    # Push
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Set HF_TOKEN env variable first: export HF_TOKEN=hf_...")
        return
    print(f"🚀 Pushing to {args.repo}")
    dataset.push_to_hub(args.repo, token=token)
    print(f"✅ Done! https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
