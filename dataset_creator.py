#!/usr/bin/env python3
"""
Simple dataset creation pipeline runner.
"""

from pathlib import Path

from download_data import DownloadYoutubeAudioAndTextBlock
from pipeline import PipelineBlock, run_pipeline
from segmentation import SegmentationBlock


def build_pipeline() -> list[PipelineBlock]:
    dataset_dir = Path("test_dataset")
    return [
        DownloadYoutubeAudioAndTextBlock(
            input_dirs={
                "cookies": Path("cookies.txt"),
            },
            output_dirs={
                "audio": dataset_dir / "audio",
                "text": dataset_dir / "text",
                "metadata": dataset_dir / "metadata.jsonl",
                "playlist_info": dataset_dir / "playlist_info.json",
            },
            playlist_url="https://youtube.com/playlist?list=PLpi8IQW8sLlOmmCgJA00ecGLHYMBcS5bu",
            base_url="https://azadyawelat.com",
        ),
        SegmentationBlock(
            input_dirs={
                "audio": dataset_dir / "audio",
                "text": dataset_dir / "text",
                "metadata": dataset_dir / "metadata.jsonl",
            },
            output_dirs={
                "audio_segments": dataset_dir / "audio_segments",
                "metadata": dataset_dir / "segments_metadata.jsonl",
            },
        ),
    ]


def main() -> None:
    pipeline = build_pipeline()
    run_pipeline(pipeline)


if __name__ == "__main__":
    main()
