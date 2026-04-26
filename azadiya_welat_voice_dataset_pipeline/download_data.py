#!/usr/bin/env python3
"""
Download Kurdish Kurmanji TTS dataset:
- Audio from YouTube playlist (news readings)
- Text from azadyawelat.com (matching articles via slugified video titles)

Usage:
    pip install yt-dlp trafilatura python-slugify
    python download_data.py
"""

import json
import time
import subprocess
from pathlib import Path

import fasttext
import trafilatura
import yt_dlp
from huggingface_hub import hf_hub_download
from slugify import slugify

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_PLAYLIST_URL = "https://youtube.com/playlist?list=PLpi8IQW8sLlOmmCgJA00ecGLHYMBcS5bu"
DEFAULT_BASE_URL = "https://azadyawelat.com"
SAMPLE_RATE = 16000  # 16kHz for TTS

# fastText language identification (Facebook AI Research) via Hugging Face Hub
FASTTEXT_MIN_PROB = 0.60


def get_playlist_info(
    playlist_url: str, cookies_file: Path | None = None
) -> list[dict]:
    """Fetch video metadata from the YouTube playlist using yt-dlp."""
    print("📥 Fetching playlist metadata...")
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "noplaylist": False,
            "cachedir": False,
        }
        if cookies_file and cookies_file.exists():
            ydl_opts["cookiefile"] = str(cookies_file)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        entries = info.get("entries") or []
        videos: list[dict] = []
        for entry in entries:
            if not entry:
                continue
            video_id = entry.get("id") or entry.get("url")
            title = entry.get("title")
            if not video_id or not title:
                continue
            videos.append(
                {
                    "id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }
            )

        print(f"✅ Found {len(videos)} videos in playlist")
        return videos
    except Exception as e:
        print(f"❌ Playlist fetch error: {e}")
        return []


def load_fasttext_langid_model():
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification", filename="model.bin"
    )
    model = fasttext.load_model(model_path)
    return model


def detect_lang_fasttext(model, text: str) -> tuple[str | None, float]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return None, 0.0

    labels, probs = model.predict(normalized, k=1)
    if not labels:
        return None, 0.0
    lang = labels[0].replace("__label__", "")
    prob = float(probs[0]) if probs else 0.0
    return lang, prob


def download_audio(
    video_url: str,
    video_id: str,
    output_path: Path,
    audio_dir: Path,
    cookies_file: Path | None = None,
) -> bool:
    """Download best audio via yt-dlp and convert to 16kHz mono WAV via FFmpeg."""
    if output_path.exists():
        print(f"  ⏭️  Audio already exists: {output_path.name}")
        return True

    temp_dir = audio_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Download best available audio with yt-dlp (more resilient to YouTube changes)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(temp_dir / f"{video_id}.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "cachedir": False,
        }
        if cookies_file and cookies_file.exists():
            ydl_opts["cookiefile"] = str(cookies_file)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            temp_path = Path(ydl.prepare_filename(info))

        if not temp_path.exists():
            matches = sorted(temp_dir.glob(f"{video_id}.*"))
            temp_path = matches[0] if matches else temp_path

        if not temp_path.exists():
            raise FileNotFoundError(f"yt-dlp did not produce an audio file for {video_id}")

        # 2) Convert to WAV 16kHz mono using FFmpeg directly
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_path),
            "-vn",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]

        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        # 3) Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()

        return output_path.exists()
    except yt_dlp.utils.DownloadError as e:
        print(f"  ❌ Audio download failed (yt-dlp): {e}")
        print("     Hint: try updating yt-dlp (e.g. `pipenv update yt-dlp`).")
        for p in temp_dir.glob(f"{video_id}.*"):
            try:
                p.unlink()
            except OSError:
                pass
        return False
    except Exception as e:
        print(f"  ❌ Audio download/convert failed: {e}")
        for p in temp_dir.glob(f"{video_id}.*"):
            try:
                p.unlink()
            except OSError:
                pass
        return False


def scrape_article(url: str) -> dict | None:
    """Extract article title, author, and text from azadyawelat.com using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print(f"  ❌ Article not found: {url}")
            return None

        output_bytes = trafilatura.extract(
            downloaded, favor_precision=True, output_format="json", with_metadata=True,
        )
        if not output_bytes:
            print(f"  ❌ No text extracted from: {url}")
            return None

        output = json.loads(output_bytes)

        if not output:
            print(f"  ❌ No text extracted from: {url}")
            return None

        return {
            "title": output["title"],
            "author": output["author"],
            "text": output["text"],
        }

    except Exception as e:
        print(f"  ❌ Request failed for {url}: {e}")
        return None


def run_download_data(
    input_dirs: dict[str, Path],
    output_dirs: dict[str, Path],
    playlist_url: str,
    base_url: str,
) -> None:
    audio_dir = Path(output_dirs["audio"]).expanduser()
    text_dir = Path(output_dirs["text"]).expanduser()
    meta_file = Path(output_dirs["metadata"]).expanduser()
    playlist_info_path = Path(output_dirs["playlist_info"]).expanduser()
    cookies_file = (
        Path(input_dirs["cookies"]).expanduser() if "cookies" in input_dirs else None
    )

    # Create output directories
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    playlist_info_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Get playlist info
    if playlist_info_path.exists():
        with open(playlist_info_path, "r", encoding="utf-8") as f:
            videos = json.load(f)
        print(f"📋 Loaded playlist info from {playlist_info_path}")
    else:
        videos = get_playlist_info(playlist_url, cookies_file)
        if not videos:
            print("No videos found. Exiting.")
            return
        with open(playlist_info_path, "w", encoding="utf-8") as f:
            json.dump(videos, f, ensure_ascii=False, indent=2)
        print(f"📋 Saved playlist info to {playlist_info_path}")

    if cookies_file and cookies_file.exists():
        print(f"🍪 Using cookies from {cookies_file}")

    langid_model = load_fasttext_langid_model()

    # Step 2: Process each video
    metadata_entries = []
    success_count = 0
    fail_count = 0

    for i, video in enumerate(videos, 1):
        title = video["title"]
        video_id = video["id"]
        slug = slugify(title)

        print(f"\n[{i}/{len(videos)}] {title}")
        print(f"  Slug: {slug}")

        # Scrape article text
        article_url = f"{base_url}/{slug}/"
        article = scrape_article(article_url)

        if not article:
            print(f"  ❌ Skipping - no matching article found")
            fail_count += 1
            continue

        lang, prob = detect_lang_fasttext(langid_model, article.get("text", ""))
        if lang != "kmr_Latn" or prob < FASTTEXT_MIN_PROB:
            print(
                f"  ⏭️  Skipping - not Kurdish Kurmanji per fastText (lang={lang}, p={prob:.2f})"
            )
            fail_count += 1
            continue

        audio_filename = f"{video_id}.wav"
        audio_path = audio_dir / audio_filename
        audio_ok = download_audio(video["url"], video_id, audio_path, audio_dir, cookies_file)

        if not audio_ok:
            fail_count += 1
            continue

        # Save text
        text_filename = f"{video_id}.txt"
        text_path = text_dir / text_filename
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(article["text"])

        text_len = len(article["text"])
        print(f"  ✅ Audio: {audio_filename} | Text: {text_len} chars")
        print(f"     Title: {article['title']}")
        print(f"     Author: {article['author']}")

        metadata_entries.append(
            {
                "id": video_id,
                "title": article["title"],
                "author": article["author"],
                "slug": slug,
                "audio_file": f"audio/{audio_filename}",
                "text_file": f"text/{text_filename}",
                "text_length": text_len,
                "article_url": article_url,
            }
        )
        success_count += 1

        # Be polite to the server
        time.sleep(1)

    # Save metadata
    with open(meta_file, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Done! {success_count} pairs downloaded, {fail_count} failed")
    print(f"📁 Output root: {meta_file.parent.resolve()}")
    print(f"   Audio: {audio_dir.resolve()}")
    print(f"   Text:  {text_dir.resolve()}")
    print(f"   Meta:  {meta_file.resolve()}")
    print(f"   Playlist: {playlist_info_path.resolve()}")


class DownloadYoutubeAudioAndTextBlock:
    def __init__(
        self,
        input_dirs: dict[str, Path],
        output_dirs: dict[str, Path],
        playlist_url: str = DEFAULT_PLAYLIST_URL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.name = "download_youtube_audio_and_text"
        self.input_dirs = input_dirs
        self.output_dirs = output_dirs
        self.playlist_url = playlist_url
        self.base_url = base_url

    def run(self) -> None:
        required_outputs = {"audio", "text", "metadata", "playlist_info"}
        missing_outputs = required_outputs - set(self.output_dirs)
        if missing_outputs:
            raise ValueError(
                f"Missing download_data output_dirs keys: {sorted(missing_outputs)}"
            )
        run_download_data(
            self.input_dirs,
            self.output_dirs,
            playlist_url=self.playlist_url,
            base_url=self.base_url,
        )
