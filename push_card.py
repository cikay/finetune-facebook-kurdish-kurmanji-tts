#!/usr/bin/env python3
"""Push dataset card to HuggingFace Hub."""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO = "muzaffercky/azadiya-welat-kurdish-kurmanji-voice"

CARD = """
---
language:
  - kmr
license: cc-by-nc-4.0
task_categories:
  - text-to-speech
  - automatic-speech-recognition
tags:
  - kurmanji
  - kurdish
  - voice
  - speech
  - forced-alignment
  - tts
  - asr
pretty_name: Azadiya Welat Kurdish Kurmanji Voice
size_categories:
  - 10K<n<100K
---

# Azadiya Welat Kurdish Kurmanji Voice

A paired audio-text corpus for Kurdish Kurmanji (Northern Kurdish), built from news readings published by [Azadiya Welat](https://azadyawelat.com).

## Dataset Summary

- **Language:** Kurdish Kurmanji (kmr)
- **Segments:** 17,246
- **Total duration:** 29.4 hours
- **Average duration:** 6.1s (range: 2–15s)
- **Sample rate:** 16 kHz mono WAV
- **Script:** Latin

## How It Was Built

1. Audio was downloaded from YouTube news reading playlists (Azadiya Welat).
2. Corresponding article text was scraped from azadyawelat.com.
3. Audio was segmented at sentence boundaries using **CTC forced alignment** with the [MMS-300M forced alignment model](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner), which supports 1,130+ languages including Kurdish.
4. Segments were filtered by duration (2–15s), minimum word count (3), and alignment confidence score.

The ground truth text is used directly — no ASR-generated transcriptions.

## Dataset Structure

Each example contains:

- `audio`: WAV audio (16 kHz, mono)
- `text`: Kurdish Kurmanji transcript (Latin script)
- `duration`: Segment duration in seconds
- `score`: Alignment confidence (log-probability; closer to 0 is better)

## Usage

```python
from datasets import load_dataset

ds = load_dataset("muzaffercky/azadiya-welat-kurdish-kurmanji-voice")
print(ds["train"][0])
```

## Intended Use

- Fine-tuning TTS models (e.g. [facebook/mms-tts-kmr-script_latin](https://huggingface.co/facebook/mms-tts-kmr-script_latin))
- Fine-tuning ASR models for Kurdish Kurmanji
- Linguistic research on Kurdish Kurmanji

## Source

Audio and text from [Azadiya Welat](https://azadyawelat.com) news articles.

"""

token = os.environ.get("HF_TOKEN")
if not token:
    print("❌ Set HF_TOKEN in .env")
    exit(1)

api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=CARD.encode(),
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="dataset",
)
print(f"✅ Card pushed to https://huggingface.co/datasets/{REPO}")
