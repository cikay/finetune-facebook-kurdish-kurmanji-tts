#!/usr/bin/env python3
"""
Download Kurdish Kurmanji TTS dataset:
- Audio from YouTube playlist (news readings)
- Text from azadyawelat.com (matching articles via slugified video titles)

Usage:
    pip install pytubefix requests beautifulsoup4 python-slugify
    python download_dataset.py
"""

import os
import re
import json
import time
import subprocess
from pathlib import Path

import requests
from bs4 import BeautifulSoup
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


def get_playlist_info() -> list[dict]:
    """Fetch video metadata from the YouTube playlist using pytubefix."""
    print("📥 Fetching playlist metadata...")
    try:
        pl = Playlist(PLAYLIST_URL)
        videos = []
        for video in pl.videos:
            videos.append({
                "id": video.video_id,
                "title": video.title,
                "url": video.watch_url,
            })
        print(f"✅ Found {len(videos)} videos in playlist")
        return videos
    except Exception as e:
        print(f"❌ Playlist fetch error: {e}")
        return []


def download_audio(video: dict, output_path: Path) -> bool:
    """Download audio from a YouTube video as WAV 16kHz mono."""
    if output_path.exists():
        print(f"  ⏭️  Audio already exists: {output_path.name}")
        return True

    try:
        yt = YouTube(video["url"])
        stream = yt.streams.get_audio_only()
        if not stream:
            print("  ❌ No audio stream available")
            return False

        # Download to a temp file first
        temp_path = output_path.with_suffix(".m4a")
        stream.download(output_path=str(output_path.parent), filename=temp_path.name)

        # Convert to WAV 16kHz mono using ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(temp_path),
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(output_path)],
            capture_output=True, text=True,
        )
        temp_path.unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"  ❌ FFmpeg conversion failed: {result.stderr[:200]}")
            return False

        return True
    except Exception as e:
        print(f"  ❌ Audio download failed: {e}")
        return False


def scrape_article_text(slug: str) -> str | None:
    """Scrape article text from azadyawelat.com using the slug."""
    url = f"{BASE_URL}/{slug}/"
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Dataset-Builder/1.0)"
        })
        if resp.status_code != 200:
            print(f"  ❌ Article not found (HTTP {resp.status_code}): {url}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Try to get the main article content
        # Look for common article content containers
        article = (
            soup.find("article")
            or soup.find("div", class_=re.compile(r"entry-content|post-content|article-content|content"))
            or soup.find("div", class_=re.compile(r"single-content|main-content"))
        )

        if article:
            # Remove script/style/nav/footer/related posts elements
            for tag in article.find_all(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            # Remove "related news" section
            for h2 in article.find_all(["h2", "h3"]):
                if h2.text and "Nûçeyên Eleqedar" in h2.text:
                    # Remove everything after related news heading
                    for sibling in h2.find_next_siblings():
                        sibling.decompose()
                    h2.decompose()
                    break

            paragraphs = article.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        else:
            # Fallback: get all paragraph text from the page
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return text.strip() if text.strip() else None

    except requests.RequestException as e:
        print(f"  ❌ Request failed for {url}: {e}")
        return None


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
        text = scrape_article_text(slug)
        if not text:
            print(f"  ⚠️  No text found, trying without trailing parts...")
            # Try shorter slug variants (remove last word progressively)
            slug_parts = slug.split("-")
            found = False
            for trim in range(1, min(4, len(slug_parts))):
                short_slug = "-".join(slug_parts[:-trim])
                if not short_slug:
                    break
                text = scrape_article_text(short_slug)
                if text:
                    slug = short_slug
                    found = True
                    break
            if not found:
                print(f"  ❌ Skipping - no matching article found")
                # Still keep the audio, but mark as no-text
                fail_count += 1
                continue

        # Save text
        text_filename = f"{video_id}.txt"
        text_path = TEXT_DIR / text_filename
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  ✅ Audio: {audio_filename} | Text: {len(text)} chars")

        metadata_entries.append({
            "id": video_id,
            "title": title,
            "slug": slug,
            "audio_file": f"audio/{audio_filename}",
            "text_file": f"text/{text_filename}",
            "text_length": len(text),
            "article_url": f"{BASE_URL}/{slug}/",
        })
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
