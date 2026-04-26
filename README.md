# Voice Dataset Creation using Azadiya Welat audio news

A pipeline to build a paired audio-text dataset for Kurdish Kurmanji, suitable for fine-tuning TTS and ASR models.

It downloads news readings from a YouTube playlist, matches them with article text scraped from [azadyawelat.com](https://azadyawelat.com), then segments long audio into short utterances using CTC forced alignment.

**Published dataset:** [muzaffercky/azadiya-welat-kurdish-kurmanji-voice](https://huggingface.co/datasets/muzaffercky/azadiya-welat-kurdish-kurmanji-voice)

## Clone

Clone the latest code:

```bash
git clone https://github.com/cikay/azadiya-welat-voice-dataset-pipeline.git
```

To clone a specific tag (e.g. `v1.0.0`):

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/cikay/azadiya-welat-voice-dataset-pipeline.git
```

## Setup

```bash
pip install pipenv
pipenv shell
pipenv install
```

### RunPod (CUDA Image) Note

If you use `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, keep PyTorch aligned with the image stack:

```bash
pipenv run pip install --no-cache-dir --force-reinstall torch==2.8.0 torchaudio==2.8.0 nvidia-cusparselt-cu12==0.7.1
pipenv run pip install "fsspec[http]<=2026.2.0,>=2023.1.0"
pipenv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Expected output should start with `2.8.0` and CUDA `12.8`.

## Pipeline

The pipeline is configured via `configs/config.yml`. Run it with:

```bash
python -m azadiya_welat_voice_dataset_pipeline.pipeline --config configs/config.yml
```

Stages run in the order defined under the `stages` key in `configs/config.yml`.

### AcquireStage

Downloads paired audio/text data from YouTube and Azadiya Welat:

1. Fetches playlist metadata with `yt-dlp`.
2. Downloads each video's audio as 16kHz mono WAV.
3. Matches article text using the title slug on `azadyawelat.com`.
4. Filters by language using fastText (`langid_language`).
5. Writes `metadata.jsonl` and `playlist_info.json` to `output_dir`.

Configured under `acquire` in `configs/config.yml`. Supports pluggable strategies:

- **audio strategies:** `youtube_playlist`
- **text strategies:** `web_scrape`

### SegmentationStage

| Key | Default | Description |
|---|---|---|
| `min_duration` | `2.0` | Minimum segment duration (seconds) |
| `max_duration` | `15.0` | Maximum segment duration (seconds) |
| `min_words` | `3` | Minimum words per segment |
| `min_align_score` | `-7.0` | Minimum alignment confidence score |
| `end_padding` | `0.15` | Silence padding after each segment end (seconds) |

Splits long audio (~5 min each) into short utterances using CTC forced alignment:

1. Loads the [MMS-300M forced alignment model](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) (supports 1,130+ languages including Kurdish).
2. Aligns ground truth text to audio — no ASR transcription involved.
3. Maps word-level timestamps back to sentence boundaries.
4. Sub-splits sentences longer than `max_duration` by `;:` punctuation.
5. Filters segments by duration, word count, and alignment confidence.
6. Saves segmented WAV files and `metadata.jsonl` to `output_dir`.

### Publish Dataset

```bash
python -m azadiya_welat_voice_dataset_pipeline.push_dataset --repo your-username/your-dataset-name
```

Uploads the segmented dataset to HuggingFace Hub.

### Output Structure

```text
dataset/
├── acquire/
│   ├── audio/
│   ├── text/
│   ├── metadata.jsonl
│   └── playlist_info.json
└── segments/
    ├── audio_segments/
    └── metadata.jsonl
```
