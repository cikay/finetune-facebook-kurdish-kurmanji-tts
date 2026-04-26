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

FASTTEXT_MIN_PROB = 0.60


class DownloadYoutubeAudioAndTextBlock:
    name: str = "download_youtube_audio_and_text"

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
        self._validate_output_dirs()

        audio_dir = Path(self.output_dirs["audio"]).expanduser()
        text_dir = Path(self.output_dirs["text"]).expanduser()
        meta_file = Path(self.output_dirs["metadata"]).expanduser()
        playlist_info_path = Path(self.output_dirs["playlist_info"]).expanduser()
        cookies_file = (
            Path(self.input_dirs["cookies"]).expanduser()
            if "cookies" in self.input_dirs
            else None
        )

        self._setup_dirs(audio_dir, text_dir, meta_file, playlist_info_path)

        videos = self._load_playlist_info(playlist_info_path, cookies_file)
        if not videos:
            print("No videos found. Exiting.")
            return

        if cookies_file and cookies_file.exists():
            print(f"🍪 Using cookies from {cookies_file}")

        langid_model = self._load_langid_model()

        metadata_entries, success_count, fail_count = self._download_text_audio_entries(
            videos, audio_dir, text_dir, cookies_file, langid_model
        )

        self._save_metadata(metadata_entries, meta_file)
        self._print_summary(success_count, fail_count, audio_dir, text_dir, meta_file, playlist_info_path)

    def _validate_output_dirs(self) -> None:
        required_outputs = {"audio", "text", "metadata", "playlist_info"}
        missing_outputs = required_outputs - set(self.output_dirs)
        if missing_outputs:
            raise ValueError(
                f"Missing download_data output_dirs keys: {sorted(missing_outputs)}"
            )

    def _setup_dirs(
        self,
        audio_dir: Path,
        text_dir: Path,
        meta_file: Path,
        playlist_info_path: Path,
    ) -> None:
        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        playlist_info_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_playlist_info(
        self, playlist_info_path: Path, cookies_file: Path | None
    ) -> list[dict]:
        if playlist_info_path.exists():
            with open(playlist_info_path, "r", encoding="utf-8") as f:
                videos = json.load(f)
            print(f"📋 Loaded playlist info from {playlist_info_path}")
            return videos

        videos = self._fetch_playlist_info(cookies_file)
        if videos:
            with open(playlist_info_path, "w", encoding="utf-8") as f:
                json.dump(videos, f, ensure_ascii=False, indent=2)
            print(f"📋 Saved playlist info to {playlist_info_path}")
        return videos

    def _fetch_playlist_info(self, cookies_file: Path | None) -> list[dict]:
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
                info = ydl.extract_info(self.playlist_url, download=False)

            videos = []
            for entry in info.get("entries") or []:
                if not entry:
                    continue
                video_id = entry.get("id") or entry.get("url")
                title = entry.get("title")
                if not video_id or not title:
                    continue
                videos.append({
                    "id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                })

            print(f"✅ Found {len(videos)} videos in playlist")
            return videos
        except Exception as e:
            print(f"❌ Playlist fetch error: {e}")
            return []

    def _load_langid_model(self):
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification", filename="model.bin"
        )
        return fasttext.load_model(model_path)

    def _detect_language(self, model, text: str) -> tuple[str | None, float]:
        normalized = " ".join((text or "").split())
        if not normalized:
            return None, 0.0
        labels, probs = model.predict(normalized, k=1)
        if not labels:
            return None, 0.0
        lang = labels[0].replace("__label__", "")
        prob = float(probs[0]) if probs else 0.0
        return lang, prob

    def _download_text_audio_entries(
        self,
        videos: list[dict],
        audio_dir: Path,
        text_dir: Path,
        cookies_file: Path | None,
        langid_model,
    ) -> tuple[list[dict], int, int]:
        metadata_entries = []
        success_count = 0
        fail_count = 0

        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {video['title']}")

            entry = self._download_text_audio(video, audio_dir, text_dir, cookies_file, langid_model)
            if entry is None:
                fail_count += 1
            else:
                metadata_entries.append(entry)
                success_count += 1

            time.sleep(1)

        return metadata_entries, success_count, fail_count

    def _download_text_audio(
        self,
        video: dict,
        audio_dir: Path,
        text_dir: Path,
        cookies_file: Path | None,
        langid_model,
    ) -> dict | None:
        title = video["title"]
        video_id = video["id"]
        slug = slugify(title)
        print(f"  Slug: {slug}")

        article_url = f"{self.base_url}/{slug}/"
        article = self._scrape_article(article_url)
        if not article:
            print("  ❌ Skipping - no matching article found")
            return None

        lang, prob = self._detect_language(langid_model, article.get("text", ""))
        if lang != "kmr_Latn" or prob < FASTTEXT_MIN_PROB:
            print(f"  ⏭️  Skipping - not Kurdish Kurmanji per fastText (lang={lang}, p={prob:.2f})")
            return None

        audio_path = audio_dir / f"{video_id}.wav"
        if not self._download_audio(video["url"], video_id, audio_path, audio_dir, cookies_file):
            return None

        text_path = text_dir / f"{video_id}.txt"
        self._save_text(article["text"], text_path)

        print(f"  ✅ Audio: {audio_path.name} | Text: {len(article['text'])} chars")
        print(f"     Title: {article['title']}")
        print(f"     Author: {article['author']}")

        return self._build_metadata_entry(video_id, article, slug, audio_path, text_path, article_url)

    def _scrape_article(self, url: str) -> dict | None:
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

    def _download_audio(
        self,
        video_url: str,
        video_id: str,
        output_path: Path,
        audio_dir: Path,
        cookies_file: Path | None,
    ) -> bool:
        if output_path.exists():
            print(f"  ⏭️  Audio already exists: {output_path.name}")
            return True

        temp_dir = audio_dir / "_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = self._download_raw_audio(video_url, video_id, temp_dir, cookies_file)
        if temp_path is None:
            return False

        ok = self._convert_to_wav(temp_path, output_path)
        self._cleanup_temp_files(temp_dir, video_id)
        return ok

    def _download_raw_audio(
        self,
        video_url: str,
        video_id: str,
        temp_dir: Path,
        cookies_file: Path | None,
    ) -> Path | None:
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

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                temp_path = Path(ydl.prepare_filename(info))

            if not temp_path.exists():
                matches = sorted(temp_dir.glob(f"{video_id}.*"))
                temp_path = matches[0] if matches else temp_path

            if not temp_path.exists():
                raise FileNotFoundError(f"yt-dlp did not produce an audio file for {video_id}")

            return temp_path
        except yt_dlp.utils.DownloadError as e:
            print(f"  ❌ Audio download failed (yt-dlp): {e}")
            print("     Hint: try updating yt-dlp (e.g. `pipenv update yt-dlp`).")
            return None
        except Exception as e:
            print(f"  ❌ Audio download failed: {e}")
            return None

    def _convert_to_wav(self, temp_path: Path, output_path: Path) -> bool:
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(temp_path),
                    "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1",
                    "-acodec", "pcm_s16le", str(output_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return output_path.exists()
        except Exception as e:
            print(f"  ❌ Audio conversion failed: {e}")
            return False

    def _cleanup_temp_files(self, temp_dir: Path, video_id: str) -> None:
        for p in temp_dir.glob(f"{video_id}.*"):
            try:
                p.unlink()
            except OSError:
                pass

    def _save_text(self, text: str, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _build_metadata_entry(
        self,
        video_id: str,
        article: dict,
        slug: str,
        audio_path: Path,
        text_path: Path,
        article_url: str,
    ) -> dict:
        return {
            "id": video_id,
            "title": article["title"],
            "author": article["author"],
            "slug": slug,
            "audio_file": f"audio/{audio_path.name}",
            "text_file": f"text/{text_path.name}",
            "text_length": len(article["text"]),
            "article_url": article_url,
        }

    def _save_metadata(self, entries: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _print_summary(
        self,
        success_count: int,
        fail_count: int,
        audio_dir: Path,
        text_dir: Path,
        meta_file: Path,
        playlist_info_path: Path,
    ) -> None:
        print(f"\n{'='*60}")
        print(f"✅ Done! {success_count} pairs downloaded, {fail_count} failed")
        print(f"📁 Output root: {meta_file.parent.resolve()}")
        print(f"   Audio: {audio_dir.resolve()}")
        print(f"   Text:  {text_dir.resolve()}")
        print(f"   Meta:  {meta_file.resolve()}")
        print(f"   Playlist: {playlist_info_path.resolve()}")
