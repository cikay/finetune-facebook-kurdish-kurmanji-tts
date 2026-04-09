#!/usr/bin/env python3
"""
Simple pipeline protocol and runner.
"""

from typing import Protocol


class PipelineBlock(Protocol):
    name: str

    def run(self) -> None:
        ...


def run_pipeline(pipeline: list[PipelineBlock]) -> None:
    for idx, block in enumerate(pipeline, start=1):
        print(f"[{idx}/{len(pipeline)}] Running block: {block.name}")
        block.run()
    print("Pipeline completed.")
