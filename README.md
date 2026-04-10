# Fine tunning Facebook Kurdish Kurmanji TTS

## Dataset Creation

A pipeline to build a paired audio-text dataset for Kurdish Kurmanji, suitable for fine-tuning TTS and ASR models.

It downloads news readings from a YouTube playlist, matches them with article text scraped from [azadyawelat.com](https://azadyawelat.com), then segments long audio into short utterances using CTC forced alignment.

**Published dataset:** [muzaffercky/azadiya-welat-kurdish-kurmanji-voice](https://huggingface.co/datasets/muzaffercky/azadiya-welat-kurdish-kurmanji-voice)

### Setup

```bash
pip install pipenv
pipenv shell
pipenv install
```

#### RunPod (CUDA Image) Note

If you use `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, keep PyTorch aligned with the image stack:

```bash
pipenv run pip install --no-cache-dir --force-reinstall torch==2.8.0 torchaudio==2.8.0 nvidia-cusparselt-cu12==0.7.1
pipenv run pip install "fsspec[http]<=2026.2.0,>=2023.1.0"
pipenv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Expected output should start with `2.8.0` and CUDA `12.8`.


### Pipeline

Run the end-to-end pipeline:

```bash
python -m finetune_tts.dataset_creator
```

`dataset_creator.py` builds a simple block pipeline with:

1. `DownloadYoutubeAudioAndTextBlock`
2. `SegmentationBlock`

#### DownloadYoutubeAudioAndTextBlock

Downloads paired audio/text data from YouTube and Azadiya Welat:

1. Fetches playlist metadata with `yt-dlp`.
2. Downloads each video's audio as 16kHz mono WAV.
3. Matches article text using title slug on `azadyawelat.com`.
4. Writes `metadata.jsonl` and `playlist_info.json`.

#### SegmentationBlock

| Variable | Default | Description |
|---|---|---|
| `MIN_DURATION` | `2.0` | Minimum segment duration (seconds) |
| `MAX_DURATION` | `15.0` | Maximum segment duration (seconds) |
| `MIN_WORDS` | `3` | Minimum words per segment |
| `MIN_SCORE` | `-7.0` | Minimum alignment confidence score |


Splits long audio (~5 min each) into short utterances using forced alignment:

1. Loads the [MMS-300M forced alignment model](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) (supports 1,130+ languages including Kurdish).
2. Aligns ground truth text to audio using CTC forced alignment — no ASR transcription involved.
3. Maps word-level timestamps back to sentence boundaries.
4. Filters segments by duration (2–15s), word count (≥3), and alignment confidence.
5. Saves segmented WAV files and `segments_metadata.jsonl`.

#### Publish Dataset (`push_dataset.py`)
```bash
python push_dataset.py --repo your-username/your-dataset-name
```

Uploads the segmented dataset to HuggingFace Hub.

### Output Structure

```text
test_dataset/
├── audio/
├── text/
├── metadata.jsonl
├── playlist_info.json
├── audio_segments/
└── segments_metadata.jsonl
```

## Fine tunning

