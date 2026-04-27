import json
import time
from pathlib import Path

import fasttext
from huggingface_hub import hf_hub_download

from .audio.youtube_playlist import YoutubePlaylistAudioStrategy
from .text.web_scrape import WebScrapeTextStrategy

AUDIO_STRATEGIES = {
    "youtube_playlist": YoutubePlaylistAudioStrategy,
}
TEXT_STRATEGIES = {
    "web_scrape": WebScrapeTextStrategy,
}


class AcquireStage:
    name = "acquire"

    def __init__(self, config: dict) -> None:
        self.output_dir = Path(config["output_dir"])
        self.language = config.get("langid_language", "kmr_Latn")
        self.min_lang_confidence = float(config.get("min_lang_confidence", 0.60))
        self.sources = config.get("sources", [])

    def run(self) -> None:
        audio_dir = self.output_dir / "audio"
        text_dir = self.output_dir / "text"
        meta_file = self.output_dir / "metadata.jsonl"
        playlist_info_path = self.output_dir / "playlist_info.json"

        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        langid_model = self._load_langid_model()

        all_entries = []
        success_count = 0
        fail_count = 0

        for source in self.sources:
            audio_strategy = self._build_audio_strategy(source["audio"])
            text_strategy = self._build_text_strategy(source["text"])

            items = self._load_items(audio_strategy, playlist_info_path)

            for i, item in enumerate(items, 1):
                print(f"\n[{i}/{len(items)}] {item['title']}")
                entry = self._process_item(item, audio_strategy, text_strategy, audio_dir, text_dir, langid_model)
                if entry:
                    all_entries.append(entry)
                    success_count += 1
                else:
                    fail_count += 1

                time.sleep(1)

        self._save_metadata(all_entries, meta_file)
        print(f"\n✅ Done! {success_count} pairs acquired, {fail_count} failed")

    def _build_audio_strategy(self, config: dict):
        return AUDIO_STRATEGIES[config["strategy"]](config)

    def _build_text_strategy(self, config: dict):
        return TEXT_STRATEGIES[config["strategy"]](config)

    def _load_items(self, audio_strategy, playlist_info_path: Path) -> list[dict]:
        if playlist_info_path.exists():
            with open(playlist_info_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            print(f"📋 Loaded {len(items)} items from {playlist_info_path}")
            return items

        print("📥 Fetching playlist metadata...")
        items = audio_strategy.list_items()
        if items:
            with open(playlist_info_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"📋 Saved playlist info to {playlist_info_path}")
        return items

    def _process_item(
        self,
        item: dict,
        audio_strategy,
        text_strategy,
        audio_dir: Path,
        text_dir: Path,
        langid_model,
    ) -> dict | None:
        article = text_strategy.fetch(item)
        if not article:
            print("  ❌ Skipping - no article found")
            return None

        lang, prob = self._detect_language(langid_model, article.get("text", ""))
        if lang != self.language or prob < self.min_lang_confidence:
            print(f"  ⏭️  Skipping - wrong language (lang={lang}, p={prob:.2f})")
            return None

        audio_path = audio_dir / f"{item['id']}.wav"
        if not audio_strategy.download(item, audio_path):
            return None

        text_path = text_dir / f"{item['id']}.txt"
        text_path.write_text(article["text"], encoding="utf-8")

        print(f"  ✅ {audio_path.name} | {len(article['text'])} chars")
        print(f"     Title: {article['title']}")
        print(f"     Author: {article['author']}")

        return self._build_metadata_entry(item, article, audio_path, text_path)

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

    def _build_metadata_entry(
        self, item: dict, article: dict, audio_path: Path, text_path: Path
    ) -> dict:
        return {
            "id": item["id"],
            "title": article["title"],
            "author": article["author"],
            "slug": article["slug"],
            "audio_file": f"audio/{audio_path.name}",
            "text_file": f"text/{text_path.name}",
            "text_length": len(article["text"]),
            "article_url": article["url"],
        }

    def _save_metadata(self, entries: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
