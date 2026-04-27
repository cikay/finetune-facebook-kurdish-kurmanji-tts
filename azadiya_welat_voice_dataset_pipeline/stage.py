from typing import Protocol


class PipelineStage(Protocol):
    name: str

    def run(self) -> None:
        ...
