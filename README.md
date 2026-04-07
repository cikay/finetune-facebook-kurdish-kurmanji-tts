# Kurdish Kurmanji Voice Dataset Pipeline

A pipeline to build a paired audio-text dataset for Kurdish Kurmanji, suitable for fine-tuning TTS and ASR models.

It downloads news readings from a YouTube playlist, matches them with article text scraped from [azadyawelat.com](https://azadyawelat.com), then segments long audio into short utterances using CTC forced alignment.

**Published dataset:** [muzaffercky/azadiya-welat-kurdish-kurmanji-voice](https://huggingface.co/datasets/muzaffercky/azadiya-welat-kurdish-kurmanji-voice)

## Setup

```bash
pip install pipenv
pipenv install
```


## Pipeline

### 1. Download

```bash
python download_dataset.py
```

Fetches audio and text pairs from Azadiya Welat:

1. Fetches video metadata from a YouTube playlist using `yt-dlp`.
2. Downloads each video's audio as a 16kHz mono WAV file.
3. Derives a URL slug from the video title (with Kurdish diacritic normalization) and scrapes the matching article text from azadyawelat.com.
4. Saves each audio/text pair and writes a `metadata.jsonl` index file.

### 2. Clean Audio

```bash
python clean_audio.py
```

Removes background music from audio files using [Demucs](https://github.com/adefossez/demucs) (Meta's source separation model):

1. Runs Demucs vocal separation on all WAV files.
2. Extracts the vocals (speech) track, discarding background music.
3. Saves cleaned files to `dataset/clean_audio/` with matching filenames.

### 3. Segmentation

```bash
python segmentation.py
```

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

### 4. Publish Dataset (`push_dataset.py`)
```bash
python push_dataset.py --repo your-username/your-dataset-name
```

Uploads the segmented dataset to HuggingFace Hub.

## Output Structure

```
dataset/                        # Raw downloaded data
├── audio/                      # WAV files (16kHz, mono)
├── text/                       # Plain text files
├── metadata.jsonl              # Audio/text path mappings
└── playlist_info.json
├── aligned/                   # Aligned and segmented data
    ├── segments/                   # Short WAV utterances
    └── segments_metadata.jsonl     # Segment metadata (audio, text, duration, score)
```
