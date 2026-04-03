#!/usr/bin/env python3
"""
Download Kurdish Kurmanji TTS dataset:
- Audio from YouTube playlist (news readings)
- Text from azadyawelat.com (matching articles via slugified video titles)

Usage:
    pip install pytubefix trafilatura python-slugify
    python download_dataset.py
"""

import json
import time
import subprocess
from pathlib import Path

import yt_dlp
import trafilatura
from slugify import slugify
from pytubefix import Playlist, YouTube

# ── Config ───────────────────────────────────────────────────────────────────
PLAYLIST_URL = "https://youtube.com/playlist?list=PLpi8IQW8sLlOmmCgJA00ecGLHYMBcS5bu"
BASE_URL = "https://azadyawelat.com"
OUTPUT_DIR = Path("dataset")
AUDIO_DIR = OUTPUT_DIR / "audio"
TEXT_DIR = OUTPUT_DIR / "text"
META_FILE = OUTPUT_DIR / "metadata.jsonl"
SAMPLE_RATE = 16000  # 16kHz for TTS
COOKIES_FILE = Path("cookies.txt")


def get_playlist_info() -> list[dict]:
    """Fetch video metadata from the YouTube playlist using pytubefix."""
    print("📥 Fetching playlist metadata...")
    try:
        pl = Playlist(PLAYLIST_URL, use_oauth=True, allow_oauth_cache=True)
        videos = []
        for video in pl.videos:
            videos.append(
                {
                    "id": video.video_id,
                    "title": video.title,
                    "url": video.watch_url,
                }
            )
        print(f"✅ Found {len(videos)} videos in playlist")
        return videos
    except Exception as e:
        print(f"❌ Playlist fetch error: {e}")
        return []


def download_audio(video_url: str, video_id: str, output_path: Path) -> bool:
    """Download audio using pytubefix and convert to WAV via FFmpeg."""
    if output_path.exists():
        print(f"  ⏭️  Audio already exists: {output_path.name}")
        return True

    temp_mp4 = AUDIO_DIR / f"{video_id}.mp4"

    try:
        # 1. Download highest quality audio stream
        yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
        audio_stream = yt.streams.get_audio_only()
        audio_stream.download(output_path=str(AUDIO_DIR), filename=f"{video_id}.mp4")

        # 2. Convert to WAV 16kHz Mono using FFmpeg directly
        # This mirrors your -ar 16000 -ac 1 requirements
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_mp4),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(output_path),
        ]

        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        # 3. Cleanup temp file
        if temp_mp4.exists():
            temp_mp4.unlink()

        return output_path.exists()
    except Exception as e:
        print(f"  ❌ Audio download/convert failed: {e}")
        if temp_mp4.exists():
            temp_mp4.unlink()
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


def build_full_text(article: dict) -> str:
    """Build the full text matching audio reading order: title, author, text."""
    parts = []
    if article["title"]:
        parts.append(article["title"])
    if article["author"]:
        parts.append(article["author"])
    parts.append(article["text"])
    return "\n".join(parts)


def main():
    # Create output directories
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get playlist info
    videos = get_playlist_info()
    if not videos:
        print("No videos found. Exiting.")
        return

    # Save playlist info for reference
    playlist_info_path = OUTPUT_DIR / "playlist_info.json"
    with open(playlist_info_path, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    print(f"📋 Saved playlist info to {playlist_info_path}")

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

        # Download audio
        audio_filename = f"{video_id}.wav"
        audio_path = AUDIO_DIR / audio_filename
        audio_ok = download_audio(video, audio_path)

        if not audio_ok:
            fail_count += 1
            continue

        # Scrape article text
        article_url = f"{BASE_URL}/{slug}/"
        article = scrape_article(article_url)
        if not article:
            print(f"  ❌ Skipping - no matching article found")
            fail_count += 1
            continue

        # Build full text: title + author + text (matches audio reading order)
        full_text = build_full_text(article)

        # Save text
        text_filename = f"{video_id}.txt"
        text_path = TEXT_DIR / text_filename
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"  ✅ Audio: {audio_filename} | Text: {len(full_text)} chars")
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
                "text_length": len(full_text),
                "article_url": article_url,
            }
        )
        success_count += 1

        # Be polite to the server
        time.sleep(1)

    # Save metadata
    with open(META_FILE, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Done! {success_count} pairs downloaded, {fail_count} failed")
    print(f"📁 Output: {OUTPUT_DIR.resolve()}")
    print(f"   Audio: {AUDIO_DIR.resolve()}")
    print(f"   Text:  {TEXT_DIR.resolve()}")
    print(f"   Meta:  {META_FILE.resolve()}")


if __name__ == "__main__":
    main()
