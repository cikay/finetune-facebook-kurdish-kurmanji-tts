#!/usr/bin/env python3
"""
Inference with a fine-tuned MMS TTS checkpoint.

Usage:
    python infer.py
"""

import logging
from pathlib import Path

import torch
import soundfile as sf
from transformers import AutoTokenizer, VitsModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_ID = "facebook/mms-tts-kmr-script_latin"
SAMPLE_RATE = 16000


def load_from_checkpoint(checkpoint_path: Path, device: torch.device):
    log.info("Loading base model: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID)

    log.info("Loading fine-tuned weights from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss")
    if loss is not None:
        log.info("Checkpoint: epoch=%s  loss=%.4f", epoch, loss)
    else:
        log.info("Checkpoint: epoch=%s", epoch)

    model.eval()
    model.to(device)
    return model, tokenizer


def load_from_hf_dir(hf_dir: Path, device: torch.device):
    log.info("Loading HF model from %s", hf_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
    model = VitsModel.from_pretrained(str(hf_dir))
    model.eval()
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def synthesize(model: VitsModel, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """Returns a 1-D float32 waveform tensor."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model(**inputs)
    waveform = output.waveform.squeeze()  # (T,)
    return waveform.cpu()


def save_wav(waveform: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.numpy(), SAMPLE_RATE)
    log.info("Saved → %s  (%.2f s)", path, len(waveform) / SAMPLE_RATE)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model, tokenizer = load_from_checkpoint("checkpoints/checkpoint_best.pt", device)

    sentences = ["Ez diçim dibistanê"]

    out_dir = Path("outputs")
    for i, sentence in enumerate(sentences, start=1):
        waveform = synthesize(model, tokenizer, sentence, device)
        save_wav(waveform, out_dir / f"{i:04d}.wav")

    log.info("Done. %d files written to %s", len(sentences), out_dir)


if __name__ == "__main__":
    main()
