# Kurdish Kurmanji TTS Dataset Pipeline

A script to build a paired audio-text dataset for Kurdish Kurmanji Text-to-Speech (TTS) fine-tuning.

It downloads news readings from a YouTube playlist and matches them with the corresponding article text scraped from [azadyawelat.com](https://azadyawelat.com).

## How It Works

1. Fetches video metadata from a YouTube playlist using `yt-dlp`.
2. Downloads each video's audio as a 16kHz mono WAV file.
3. Derives a URL slug from the video title (with Kurdish diacritic normalization) and scrapes the matching article text from azadyawelat.com.
4. Saves each audio/text pair and writes a `metadata.jsonl` index file.

## Output Structure

```
dataset/
├── audio/          # WAV files (16kHz, mono), named by YouTube video ID
├── text/           # Plain text files, named by YouTube video ID
├── metadata.jsonl  # One JSON object per pair (id, title, slug, file paths, etc.)
└── playlist_info.json
```

## Requirements

- Python 3.12+
- `yt-dlp` (must be installed and available on your `PATH`)
- `ffmpeg` (required by `yt-dlp` for audio conversion)

## Setup

```bash
pip install pipenv
pipenv install
```

Or install dependencies directly:

```bash
pip install yt-dlp requests beautifulsoup4
```

## Usage

```bash
pipenv run python download_dataset.py
# or
python download_dataset.py
```

The dataset will be written to the `dataset/` directory in the current working directory.

## Configuration

Edit the constants at the top of `download_dataset.py` to change the data source or output settings:

| Variable | Default | Description |
|---|---|---|
| `PLAYLIST_URL` | YouTube playlist URL | Source playlist of news readings |
| `BASE_URL` | `https://azadyawelat.com` | Website to scrape article text from |
| `OUTPUT_DIR` | `dataset/` | Root output directory |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |

## Notes

- If a full slug doesn't match an article, the script progressively trims the last word(s) of the slug and retries (up to 3 times).
- A 1-second delay is added between article requests to be polite to the server.
- Already-downloaded audio files are skipped on re-runs.
