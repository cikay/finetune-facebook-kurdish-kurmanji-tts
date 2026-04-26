from pathlib import Path

import yaml

from .acquire.stage import AcquireStage
from .segmentation.stage import SegmentationStage
from .stage import PipelineStage

STAGE_REGISTRY: dict[str, type[PipelineStage]] = {
    "acquire": AcquireStage,
    "segmentation": SegmentationStage,
}


def run_pipeline(config_path: Path = Path("configs/config.yml")) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stages_order: list[str] = config.get("stages", [])

    for idx, stage_name in enumerate(stages_order, 1):
        stage_config = config.get(stage_name, {})
        stage_cls = STAGE_REGISTRY.get(stage_name)
        if stage_cls is None:
            raise ValueError(f"Unknown stage: '{stage_name}'. Available: {list(STAGE_REGISTRY)}")
        stage = stage_cls(stage_config)
        print(f"\n[{idx}/{len(stages_order)}] Running stage: {stage_name}")
        stage.run()

    print("\nPipeline completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yml"))
    args = parser.parse_args()
    run_pipeline(args.config)
