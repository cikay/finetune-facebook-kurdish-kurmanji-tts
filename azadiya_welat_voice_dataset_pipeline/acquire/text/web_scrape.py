import json

import trafilatura
from slugify import slugify


class WebScrapeTextStrategy:
    def __init__(self, config: dict) -> None:
        self.base_url = config["base_url"].rstrip("/")
        self.endpoint_method = config.get("endpoint_method", "slugify-audio-name")

    def fetch(self, item: dict) -> dict | None:
        url = self._build_url(item)
        article = self._scrape(url)
        if article:
            article["url"] = url
            article["slug"] = slugify(item["title"])
        return article

    def _build_url(self, item: dict) -> str:
        if self.endpoint_method == "slugify-audio-name":
            return f"{self.base_url}/{slugify(item['title'])}/"
        raise ValueError(f"Unknown endpoint_method: {self.endpoint_method}")

    def _scrape(self, url: str) -> dict | None:
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
                return None

            return {
                "title": output["title"],
                "author": output["author"],
                "text": output["text"],
            }
        except Exception as e:
            print(f"  ❌ Scrape failed for {url}: {e}")
            return None
