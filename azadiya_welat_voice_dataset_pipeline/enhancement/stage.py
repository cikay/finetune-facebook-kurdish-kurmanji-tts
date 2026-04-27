import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score

from .tools.deepfilternet import DeepFilterNetTool

SAMPLE_RATE = 16000

TOOL_REGISTRY = {
    "deepfilternet": DeepFilterNetTool,
}


class EnhancementStage:
    name = "enhancement"

    def __init__(self, config: dict) -> None:
        self.input_dir = Path(config["input_dir"])
        self.output_dir = Path(config["output_dir"])
        self.tools = self._build_tools(config.get("tools", {}))

    def run(self) -> None:
        audio_dir = self.input_dir / "audio"
        text_dir = self.input_dir / "text"
        meta_file = self.input_dir / "metadata.jsonl"

        out_audio_dir = self.output_dir / "audio"
        out_meta_file = self.output_dir / "metadata.jsonl"
        out_audio_dir.mkdir(parents=True, exist_ok=True)

        self._symlink(text_dir, self.output_dir / "text")

        entries = self._load_metadata(meta_file)
        updated_entries = []
        enhanced_count = 0
        skipped_count = 0

        for i, entry in enumerate(entries, 1):
            audio_path = audio_dir / Path(entry["audio_file"]).name
            out_audio_path = out_audio_dir / audio_path.name

            if not audio_path.exists():
                print(f"  ⚠️  Missing audio: {audio_path}")
                continue

            print(f"[{i}/{len(entries)}] {entry['title'][:60]}...")

            enhanced = self._apply_tools(audio_path, out_audio_path)
            if enhanced:
                enhanced_count += 1
            else:
                self._symlink(audio_path, out_audio_path)
                skipped_count += 1

            updated_entries.append({**entry, "enhanced": enhanced})

        self._save_metadata(updated_entries, out_meta_file)
        print(f"\n✅ Enhanced: {enhanced_count} | Skipped (already clean): {skipped_count}")

    def _build_tools(self, tools_config: dict) -> list:
        tools = []
        for name, tool_config in tools_config.items():
            cls = TOOL_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown enhancement tool: '{name}'. Available: {list(TOOL_REGISTRY)}")
            tools.append(cls(tool_config or {}))
        return tools

    def _apply_tools(self, input_path: Path, output_path: Path) -> bool:
        original_audio, _ = sf.read(str(input_path))
        original_metrics = self._compute_dns_mos(original_audio)
        applied_any = False

        for tool in self.tools:
            if not tool.should_run(original_metrics):
                print(f"  ⏭️  Skipping {type(tool).__name__} (run_if not met)")
                self._log_original_metrics(input_path.name, original_metrics)
                continue

            print(f"  🔧 Applying {type(tool).__name__}...")
            enhanced_audio = tool.enhance(input_path)
            enhanced_metrics = self._compute_dns_mos(enhanced_audio)
            self._log_metrics(input_path.name, original_metrics, enhanced_metrics)

            if tool.should_replace(original_metrics, enhanced_metrics):
                print(f"  ✅ Keeping enhanced audio")
                sf.write(str(output_path), enhanced_audio, SAMPLE_RATE)
                applied_any = True
            else:
                print(f"  ↩️  Reverting — enhancement caused damage")

        return applied_any

    def _log_original_metrics(self, filename: str, metrics: dict) -> None:
        m = metrics["dns_mos"]
        print(f"  📊 {filename}")
        print(f"     {'metric':<8} {'value':>8}")
        for key in ("bak", "sig", "ovr", "p808"):
            print(f"     {key:<8} {m[key]:>8.2f}")

    def _log_metrics(self, filename: str, original: dict, enhanced: dict) -> None:
        orig = original["dns_mos"]
        enh = enhanced["dns_mos"]
        print(f"  📊 {filename}")
        print(f"     {'metric':<8} {'original':>8} {'enhanced':>8} {'change':>8}")
        for key in ("bak", "sig", "ovr", "p808"):
            change = enh[key] - orig[key]
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"     {key:<8} {orig[key]:>8.2f} {enh[key]:>8.2f} {change:>+8.2f} {arrow}")

    def _compute_dns_mos(self, audio: np.ndarray) -> dict:
        waveform = torch.from_numpy(audio).float()
        scores = deep_noise_suppression_mean_opinion_score(
            preds=waveform, fs=SAMPLE_RATE, personalized=False,
        )
        return {
            "dns_mos": {
                "p808": round(float(scores[0]), 2),
                "sig": round(float(scores[1]), 2),
                "bak": round(float(scores[2]), 2),
                "ovr": round(float(scores[3]), 2),
            }
        }

    def _load_metadata(self, meta_file: Path) -> list[dict]:
        entries = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def _save_metadata(self, entries: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _symlink(self, src: Path, dst: Path) -> None:
        if not dst.exists():
            dst.symlink_to(src.resolve())
