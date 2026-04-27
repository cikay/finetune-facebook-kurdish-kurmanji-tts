import subprocess
from pathlib import Path

import yt_dlp

SAMPLE_RATE = 16000


class YoutubePlaylistAudioStrategy:
    def __init__(self, config: dict) -> None:
        self.playlist_urls = config["playlist_urls"]
        self.cookies_file = Path(config["cookies_file"]) if "cookies_file" in config else None

    def list_items(self) -> list[dict]:
        items = []
        for url in self.playlist_urls:
            items.extend(self._fetch_playlist(url))
        return items

    def download(self, item: dict, output_path: Path) -> bool:
        if output_path.exists():
            print(f"  ⏭️  Audio already exists: {output_path.name}")
            return True

        temp_dir = output_path.parent / "_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = self._download_raw(item["url"], item["id"], temp_dir)
        if temp_path is None:
            return False

        ok = self._convert_to_wav(temp_path, output_path)
        self._cleanup_temp(temp_dir, item["id"])
        return ok

    def _fetch_playlist(self, url: str) -> list[dict]:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": True,
            "noplaylist": False,
            "ignoreerrors": True,
            "cachedir": False,
        }
        if self.cookies_file and self.cookies_file.exists():
            ydl_opts["cookiefile"] = str(self.cookies_file)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            items = []
            for entry in info.get("entries") or []:
                if not entry:
                    continue
                video_id = entry.get("id") or entry.get("url")
                title = entry.get("title")
                if not video_id or not title:
                    continue
                items.append({
                    "id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                })
            print(f"✅ Found {len(items)} videos in playlist")
            return items
        except Exception as e:
            print(f"❌ Playlist fetch error: {e}")
            return []

    def _download_raw(self, url: str, video_id: str, temp_dir: Path) -> Path | None:
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
        if self.cookies_file and self.cookies_file.exists():
            ydl_opts["cookiefile"] = str(self.cookies_file)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                temp_path = Path(ydl.prepare_filename(info))

            if not temp_path.exists():
                matches = sorted(temp_dir.glob(f"{video_id}.*"))
                temp_path = matches[0] if matches else temp_path

            if not temp_path.exists():
                raise FileNotFoundError(f"yt-dlp did not produce a file for {video_id}")

            return temp_path
        except yt_dlp.utils.DownloadError as e:
            print(f"  ❌ Download failed (yt-dlp): {e}")
            print("     Hint: try updating yt-dlp (e.g. `pip install -U yt-dlp`).")
            return None
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
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
            print(f"  ❌ Conversion failed: {e}")
            return False

    def _cleanup_temp(self, temp_dir: Path, video_id: str) -> None:
        for p in temp_dir.glob(f"{video_id}.*"):
            try:
                p.unlink()
            except OSError:
                pass
