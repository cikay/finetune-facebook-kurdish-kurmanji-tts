#!/usr/bin/env python3
"""
Fine-tune facebook/mms-tts-kmr-script_latin on a local Kurdish Kurmanji dataset.

Implements the full VITS training loop (without GAN discriminator) using the
HuggingFace VitsModel components directly:
  - KL divergence loss   (posterior vs. prior after monotonic alignment)
  - Mel reconstruction loss (L1 on mel spectrograms)
  - Duration loss         (stochastic duration predictor NLL)

Training without the discriminator is standard practice for fine-tuning / speaker
adaptation when the decoder has already been pre-trained; it avoids the need for a
separate discriminator checkpoint and halves GPU memory.

Usage (local smoke-test):
    python -m finetune_tts.finetune --epochs 1 --save-every 10

Usage (full server run):
    python -m finetune_tts.finetune
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, VitsModel

from finetune_tts.load_dataset_from_local import load_dataset

# ── Fixed audio/STFT constants (tied to the MMS model architecture) ─────────
SAMPLE_RATE = 16000

# STFT — must match the model:
#   spectrogram_bins = n_fft/2 + 1 = 513  →  n_fft = 1024
#   hop = product(upsample_rates) = 8*8*2*2 = 256
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256

# Mel spectrogram used only for the reconstruction loss
N_MELS = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0

# ── Config helpers ────────────────────────────────────────────────────────────

DEFAULT_CONFIG_FILE = Path(__file__).parent.parent / "finetune_configs.json"


def load_config(config_name: str, config_file: Path = DEFAULT_CONFIG_FILE) -> dict:
    """Load a named config from the JSON file."""
    with open(config_file) as f:
        all_configs = json.load(f)
    if config_name not in all_configs:
        available = ", ".join(all_configs.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    return all_configs[config_name]


def list_configs(config_file: Path = DEFAULT_CONFIG_FILE) -> list[str]:
    with open(config_file) as f:
        return list(json.load(f).keys())


# These are read from the active config at runtime — initialised to None here
# so that train_step (which references them as module globals) always has a value.
MIN_ALIGN_SCORE: float = -5.0
MIN_DURATION: float = 2.0
MAX_DURATION: float = 14.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Spectrogram helpers ──────────────────────────────────────────────────────

def linear_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """
    Compute the magnitude STFT used by the VITS posterior encoder.

    Args:
        waveform: (T,) or (1, T) float32

    Returns:
        spec: (spectrogram_bins, frames) = (513, ?)  float32
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    window = torch.hann_window(WIN_LENGTH, device=waveform.device)
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )                       # (freq, frames)
    spec = torch.abs(stft)  # magnitude, no log
    return spec             # (513, T_spec)


def mel_spectrogram(waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Log-mel spectrogram used for the reconstruction loss.

    Args:
        waveform: (T,) float32

    Returns:
        (N_MELS, frames) float32
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        f_min=MEL_FMIN,
        f_max=MEL_FMAX,
        n_mels=N_MELS,
        power=1.0,
    ).to(device)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = mel_transform(waveform).squeeze(0)             # (N_MELS, frames)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


# ── Monotonic Alignment Search (pure Python, no Cython) ─────────────────────

@torch.no_grad()
def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Dynamic-programming monotonic alignment search.

    This is the Python/NumPy equivalent of the Cython `monotonic_align` in the
    original VITS repo.  For fine-tuning on a few hundred samples it is fast
    enough even on CPU.

    Args:
        neg_cent: (B, T_text, T_spec)  — per-cell log-likelihood contribution
        mask:     (B, T_text, T_spec)  — 1 where valid

    Returns:
        path: (B, T_text, T_spec)  — one-hot alignment
    """
    device, dtype = neg_cent.device, neg_cent.dtype
    b, t_x, t_y = neg_cent.shape

    neg_np = neg_cent.detach().float().cpu().numpy()
    mask_np = mask.detach().float().cpu().numpy()

    path_np = np.zeros((b, t_x, t_y), dtype=np.float32)

    for batch_idx in range(b):
        v = np.full((t_x,), -np.inf, dtype=np.float32)
        direction = np.zeros((t_x, t_y), dtype=np.int8)  # 0=stay, 1=advance

        # Determine the number of valid text tokens for this sample
        # (padding tokens have mask==0 everywhere in the text dim)
        text_len = int(np.sum(mask_np[batch_idx, :, 0]))
        spec_len = int(np.sum(mask_np[batch_idx, 0, :]))

        if text_len == 0 or spec_len == 0:
            continue

        v[0] = neg_np[batch_idx, 0, 0]

        for j in range(1, spec_len):
            # v_prev[i] = best cumulative score when advancing from text pos i-1
            v_stay = v.copy()
            v_advance = np.pad(v[: text_len - 1], (1, t_x - text_len), constant_values=-np.inf)
            # Actually shift: advancing means text pos i → comes from text pos i-1
            v_advance_full = np.full_like(v, -np.inf)
            v_advance_full[1:text_len] = v[: text_len - 1]

            advance_better = v_advance_full > v_stay
            v_new = np.where(advance_better, v_advance_full, v_stay)
            v_new += neg_np[batch_idx, :, j]
            v_new[text_len:] = -np.inf  # keep out-of-range invalid

            direction[:, j] = advance_better.astype(np.int8)
            v = v_new

        # Backtrack
        i = text_len - 1
        for j in range(spec_len - 1, -1, -1):
            path_np[batch_idx, i, j] = 1.0
            if j > 0 and direction[i, j]:
                i = max(0, i - 1)

    return torch.tensor(path_np, dtype=dtype, device=device)


# ── Dataset ──────────────────────────────────────────────────────────────────

class VitsTTSDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_text_length: int = 200):
        self.items = []
        skipped = 0
        for item in hf_dataset:
            if not Path(item["audio"]).exists():
                skipped += 1
                continue
            self.items.append(item)
        if skipped:
            log.warning("Skipped %d items with missing audio files.", skipped)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # ── Audio ──
        waveform, sr = sf.read(item["audio"], dtype="float32")
        assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} Hz in {item['audio']}"
        waveform = torch.from_numpy(waveform)  # (T,)

        # ── Text ──
        enc = self.tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            padding=False,
        )
        input_ids = enc["input_ids"].squeeze(0)          # (T_text,)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "waveform": waveform,
            "text": item["text"],
            "audio_path": item["audio"],
        }


def collate_fn(batch):
    """Pad all sequences to the longest in the batch."""
    # Text
    text_lens = [b["input_ids"].shape[0] for b in batch]
    max_text = max(text_lens)
    input_ids = torch.zeros(len(batch), max_text, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_text, dtype=torch.long)
    for i, b in enumerate(batch):
        tl = text_lens[i]
        input_ids[i, :tl] = b["input_ids"]
        attention_mask[i, :tl] = b["attention_mask"]

    # Audio
    audio_lens = [b["waveform"].shape[0] for b in batch]
    max_audio = max(audio_lens)
    waveforms = torch.zeros(len(batch), max_audio)
    for i, b in enumerate(batch):
        al = audio_lens[i]
        waveforms[i, :al] = b["waveform"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "waveforms": waveforms,
        "audio_lens": torch.tensor(audio_lens, dtype=torch.long),
        "text_lens": torch.tensor(text_lens, dtype=torch.long),
    }


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(
    model: VitsModel,
    batch: dict,
    device: torch.device,
    mel_lambda: float = 45.0,
) -> dict[str, torch.Tensor]:
    """
    One forward + loss computation step.

    Losses (no discriminator):
      - L_kl  : KL(posterior || prior)  after MAS alignment
      - L_mel : L1 on log-mel spectrograms
      - L_dur : stochastic duration predictor NLL

    Returns a dict of scalar losses.
    """
    input_ids = batch["input_ids"].to(device)         # (B, T_text)
    attention_mask = batch["attention_mask"].to(device)
    waveforms = batch["waveforms"].to(device)          # (B, T_audio)
    audio_lens = batch["audio_lens"].to(device)        # (B,)
    text_lens = batch["text_lens"].to(device)          # (B,)

    B = input_ids.shape[0]
    mask_dtype = model.text_encoder.embed_tokens.weight.dtype

    # ── 1. Text encoder ──────────────────────────────────────────────────────
    text_padding_mask = attention_mask.unsqueeze(-1).to(mask_dtype)  # (B, T_text, 1)

    text_out = model.text_encoder(
        input_ids=input_ids,
        padding_mask=text_padding_mask,
        attention_mask=attention_mask,
        return_dict=True,
    )
    # (B, T_text, hidden) → transpose → (B, hidden, T_text)
    text_hidden = text_out.last_hidden_state.transpose(1, 2)
    text_padding_mask_t = text_padding_mask.transpose(1, 2)          # (B, 1, T_text)
    prior_mean = text_out.prior_means.transpose(1, 2)                # (B, flow_size, T_text)
    prior_log_var = text_out.prior_log_variances.transpose(1, 2)     # (B, flow_size, T_text)

    # ── 2. Linear spectrogram + posterior encoder ─────────────────────────────
    # Compute one spec per sample (different lengths), then pad
    specs = []
    spec_lens = []
    for i in range(B):
        wav_i = waveforms[i, : audio_lens[i]]
        spec_i = linear_spectrogram(wav_i)   # (513, T_spec_i)
        specs.append(spec_i)
        spec_lens.append(spec_i.shape[1])

    spec_lens_t = torch.tensor(spec_lens, device=device, dtype=torch.long)
    max_spec = max(spec_lens)
    specs_padded = torch.zeros(B, N_FFT // 2 + 1, max_spec, device=device)
    for i, spec_i in enumerate(specs):
        specs_padded[i, :, : spec_lens[i]] = spec_i  # (B, 513, T_spec)

    # Posterior encoder padding mask: (B, 1, T_spec)
    spec_idx = torch.arange(max_spec, device=device).unsqueeze(0)    # (1, T_spec)
    spec_mask = (spec_idx < spec_lens_t.unsqueeze(1)).unsqueeze(1).to(mask_dtype)  # (B,1,T_spec)

    z_q, post_mean, post_log_stddev = model.posterior_encoder(
        specs_padded, spec_mask
    )  # all (B, flow_size, T_spec)

    # ── 3. Flow: posterior → prior space ─────────────────────────────────────
    z_p = model.flow(z_q, spec_mask, global_conditioning=None)  # (B, flow_size, T_spec)

    # ── 4. Monotonic Alignment Search ─────────────────────────────────────────
    # Compute per-cell log-likelihood: z_p[j] under N(m_p[i], exp(logs_p[i]))
    # neg_cent[b, i, j] = -0.5*(log2π + logs_p[i] + (z_p[j]-m_p[i])^2 / exp(2*logs_p[i]))
    # We sum over the flow_size dimension.

    with torch.no_grad():
        # z_p:          (B, C, T_spec)
        # prior_mean:   (B, C, T_text)
        # prior_log_var:(B, C, T_text)

        neg_cent = (
            -0.5 * math.log(2 * math.pi)
            - 0.5 * prior_log_var.unsqueeze(-1)                          # (B, C, T_text, 1)
            - 0.5 * (
                z_p.unsqueeze(2) - prior_mean.unsqueeze(-1)
            ) ** 2 / torch.exp(2.0 * prior_log_var.unsqueeze(-1))
        )  # (B, C, T_text, T_spec)
        neg_cent = neg_cent.sum(1)   # (B, T_text, T_spec)

        # Build alignment mask: text_i ≤ text_len  &  spec_j ≤ spec_len
        text_idx = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)  # (1, T_text)
        text_valid = (text_idx < text_lens.unsqueeze(1))                          # (B, T_text)
        attn_mask = text_valid.unsqueeze(2) & (spec_idx < spec_lens_t.unsqueeze(1)).unsqueeze(1)
        attn_mask = attn_mask.float()   # (B, T_text, T_spec)

        path = maximum_path(neg_cent, attn_mask)   # (B, T_text, T_spec) one-hot

    # Aligned prior: weighted sum of text vectors by the path
    # path: (B, T_text, T_spec) → align prior stats to spec frames
    path_t = path.transpose(1, 2)                          # (B, T_spec, T_text)
    prior_mean_aligned = torch.bmm(
        path_t, prior_mean.transpose(1, 2)
    ).transpose(1, 2)                                      # (B, C, T_spec)
    prior_log_var_aligned = torch.bmm(
        path_t, prior_log_var.transpose(1, 2)
    ).transpose(1, 2)

    # ── 5. KL loss ────────────────────────────────────────────────────────────
    # KL(q || p) where q ~ N(post_mean, exp(post_log_stddev))
    #                   p ~ N(prior_mean_aligned, exp(prior_log_var_aligned))
    kl = (
        prior_log_var_aligned
        - post_log_stddev
        - 0.5
        + 0.5 * (torch.exp(2.0 * post_log_stddev) + (post_mean - prior_mean_aligned) ** 2)
        / torch.exp(2.0 * prior_log_var_aligned)
    )
    kl = kl * spec_mask
    loss_kl = kl.sum() / spec_mask.sum()

    # ── 6. Duration predictor loss ────────────────────────────────────────────
    # Ground-truth durations from the MAS path: sum over spec axis → (B, T_text)
    gt_durations = path.sum(2).unsqueeze(1)   # (B, 1, T_text)

    loss_dur = model.duration_predictor(
        text_hidden,
        text_padding_mask_t,
        global_conditioning=None,
        durations=gt_durations,
        reverse=False,
    )  # returns NLL per sample, shape (B,) or scalar
    loss_dur = loss_dur.mean()

    # ── 7. Decode ─────────────────────────────────────────────────────────────
    gen_waveform = model.decoder(z_q * spec_mask)  # (B, 1, T_gen)
    gen_waveform = gen_waveform.squeeze(1)   # (B, T_gen)

    # ── 8. Mel reconstruction loss ────────────────────────────────────────────
    loss_mel_total = torch.tensor(0.0, device=device)
    for i in range(B):
        # Ground truth mel
        gt_wav_i = waveforms[i, : audio_lens[i]]
        mel_gt = mel_spectrogram(gt_wav_i, device)          # (N_MELS, T_mel_gt)

        # Generated mel (trim to same spec length)
        gen_len = spec_lens[i] * HOP_LENGTH
        gen_wav_i = gen_waveform[i, :gen_len]
        mel_gen = mel_spectrogram(gen_wav_i, device)        # (N_MELS, T_mel_gen)

        # Align lengths
        min_t = min(mel_gt.shape[1], mel_gen.shape[1])
        loss_mel_total = loss_mel_total + F.l1_loss(
            mel_gen[:, :min_t], mel_gt[:, :min_t]
        )

    loss_mel = loss_mel_total / B

    total = loss_kl + mel_lambda * loss_mel + loss_dur

    return {
        "loss": total,
        "loss_kl": loss_kl.detach(),
        "loss_mel": loss_mel.detach(),
        "loss_dur": loss_dur.detach(),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, epoch, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = out_dir / f"checkpoint_step{step:07d}.pt"
    torch.save(ckpt, path)
    # Also overwrite a fixed "latest" pointer so inference scripts can find it easily
    torch.save(ckpt, out_dir / "checkpoint_latest.pt")
    log.info("Saved checkpoint → %s", path)


def load_checkpoint(model, optimizer, scheduler, ckpt_path: Path):
    log.info("Resuming from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["epoch"]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune MMS TTS Kurdish Kurmanji",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available configs are defined in finetune_configs.json.",
    )
    p.add_argument(
        "--config", default="default",
        help="Named config from finetune_configs.json (default: 'default')",
    )
    p.add_argument(
        "--config-file", type=Path, default=DEFAULT_CONFIG_FILE,
        help="Path to the JSON config file",
    )
    p.add_argument(
        "--list-configs", action="store_true",
        help="Print available config names and exit",
    )
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint_*.pt to resume from")
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_configs:
        print("Available configs:", ", ".join(list_configs(args.config_file)))
        return

    cfg = load_config(args.config, args.config_file)
    log.info("Config '%s': %s", args.config, cfg)

    # Expose filter thresholds as module-level names so train_step can read them
    global MIN_ALIGN_SCORE, MIN_DURATION, MAX_DURATION
    MIN_ALIGN_SCORE = cfg["min_align_score"]
    MIN_DURATION = cfg["min_duration"]
    MAX_DURATION = cfg["max_duration"]

    output_dir = Path(cfg["output_dir"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    log.info("Device: %s", device)

    # ── Load model & tokenizer ──
    log.info("Loading model: %s", cfg["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = VitsModel.from_pretrained(cfg["model_id"])
    model.train()
    model.to(device)

    if cfg["freeze_decoder"]:
        log.info("Freezing decoder (HiFi-GAN).")
        for p in model.decoder.parameters():
            p.requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %s", f"{trainable_params:,}")

    # ── Dataset ──
    def quality_filter(item):
        return (
            item.get("align_score", -999) >= MIN_ALIGN_SCORE
            and MIN_DURATION <= item.get("duration", 0) <= MAX_DURATION
        )

    log.info("Loading dataset...")
    hf_dataset = load_dataset(filter_fn=quality_filter)
    log.info("Dataset size: %d segments", len(hf_dataset))

    tts_dataset = VitsTTSDataset(hf_dataset, tokenizer, max_text_length=cfg["max_text_len"])
    loader = DataLoader(
        tts_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )
    log.info("Steps per epoch: %d", len(loader))

    # ── Optimizer & scheduler ──
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        betas=(0.8, 0.99),
        eps=1e-9,
    )
    scheduler = ExponentialLR(optimizer, gamma=cfg["lr_decay"])

    # ── Resume ──
    global_step = 0
    start_epoch = 0
    if args.resume is not None:
        global_step, start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1

    # ── Training loop ──
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Starting training for %d epochs -> %s", cfg["epochs"], output_dir)

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()

            losses = train_step(model, batch, device, mel_lambda=cfg["mel_lambda"])
            loss = losses["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                log.warning("NaN/Inf loss at step %d — skipping batch.", global_step)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                log.info(
                    "epoch %3d  step %6d  loss=%.4f  kl=%.4f  mel=%.4f  dur=%.4f  lr=%.2e",
                    epoch,
                    global_step,
                    losses["loss"].item(),
                    losses["loss_kl"].item(),
                    losses["loss_mel"].item(),
                    losses["loss_dur"].item(),
                    scheduler.get_last_lr()[0],
                )

            if global_step % cfg["save_every"] == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, epoch, output_dir)

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)
        log.info("-- Epoch %d done  avg_loss=%.4f --", epoch, avg)

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, cfg["epochs"] - 1, output_dir)

    # ── Save HF-compatible model for inference ──
    hf_out = output_dir / "hf_model"
    log.info("Saving HuggingFace model to %s", hf_out)
    model.eval()
    model.save_pretrained(hf_out)
    tokenizer.save_pretrained(hf_out)
    log.info("Done. Load with: VitsModel.from_pretrained('%s')", hf_out)


if __name__ == "__main__":
    main()
