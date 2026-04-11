#!/usr/bin/env python3
"""
Fine-tune facebook/mms-tts-kmr-script_latin on local Kurdish Kurmanji dataset.

Architecture: VITS (Variational Inference Text-to-Speech)
Training objective: ELBO + adversarial (MPD + MSD discriminators)
Data: loaded from local dataset/segments_metadata.jsonl via load_dataset_from_local.py

Configs are defined in finetune_configs.json. Available configs: default, fast, quality, debug.

Usage:
    python -m finetune_tts.finetune                        # uses 'default' config
    python -m finetune_tts.finetune fast                   # uses 'fast' config
    python -m finetune_tts.finetune quality --resume checkpoints/kmr_quality/checkpoint_epoch_0010.pt
    python -m finetune_tts.finetune --list                 # list available configs
"""

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from transformers import VitsModel, VitsTokenizer

from finetune_tts.load_dataset_from_local import load_dataset as load_local_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "facebook/mms-tts-kmr-script_latin"
SAMPLE_RATE = 16000

# STFT parameters — n_fft=1024 → 513 frequency bins (matches model.config.spectrogram_bins)
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

# Mel spectrogram for reconstruction loss
N_MELS = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0

# Training limits
MAX_WAV_SAMPLES = int(15.0 * SAMPLE_RATE)
MAX_TEXT_TOKENS = 300


# ── Audio utilities ───────────────────────────────────────────────────────────


def compute_linear_spec(waveform: torch.Tensor) -> torch.Tensor:
    """Linear magnitude spectrogram [513, T_frames] from mono waveform [T]."""
    window = torch.hann_window(WIN_LENGTH, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)


def compute_mel_spec(waveform: torch.Tensor, mel_transform) -> torch.Tensor:
    """Log-mel spectrogram [N_MELS, T_frames] from mono waveform [T]."""
    return torch.log(torch.clamp(mel_transform(waveform), min=1e-5))


def build_mel_transform(device: str):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        f_min=MEL_FMIN,
        f_max=MEL_FMAX,
        power=1.0,
    ).to(device)


# ── Monotonic Alignment Search ────────────────────────────────────────────────


def _mas_numpy(log_p: np.ndarray) -> np.ndarray:
    T_text, T_spec = log_p.shape
    Q = np.full((T_text, T_spec), -np.inf, dtype=np.float64)
    Q[0, 0] = log_p[0, 0]
    for j in range(1, T_spec):
        Q[0, j] = Q[0, j - 1] + log_p[0, j]
    for i in range(1, T_text):
        Q[i, i] = Q[i - 1, i - 1] + log_p[i, i]
        for j in range(i + 1, T_spec):
            Q[i, j] = max(Q[i, j - 1], Q[i - 1, j - 1]) + log_p[i, j]
    attn = np.zeros((T_spec, T_text), dtype=np.float32)
    i = T_text - 1
    for j in range(T_spec - 1, -1, -1):
        attn[j, i] = 1.0
        if i > 0 and (j == 0 or Q[i - 1, j - 1] > Q[i, j - 1]):
            i -= 1
    return attn


def maximum_path_batch(log_p: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    B = log_p.shape[0]
    lp_np = log_p.detach().cpu().numpy()
    mask_np = attn_mask.detach().cpu().numpy().astype(bool)
    results = []
    for b in range(B):
        t_text = int(mask_np[b].any(axis=1).sum())
        t_spec = int(mask_np[b].any(axis=0).sum())
        attn_b = _mas_numpy(lp_np[b, :t_text, :t_spec])
        full = np.zeros((log_p.shape[2], log_p.shape[1]), dtype=np.float32)
        full[:t_spec, :t_text] = attn_b
        results.append(full)
    return torch.from_numpy(np.stack(results)).to(log_p.device)


# ── Dataset ───────────────────────────────────────────────────────────────────


class KurmanjiTTSDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.data = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        audio_array, _ = sf.read(item["audio"], dtype="float32")
        audio_array = audio_array[:MAX_WAV_SAMPLES]
        waveform = torch.from_numpy(audio_array)

        token_out = self.tokenizer(item["text"], return_tensors="pt")
        input_ids = token_out.input_ids.squeeze(0)
        if len(input_ids) > MAX_TEXT_TOKENS:
            input_ids = input_ids[:MAX_TEXT_TOKENS]

        return {
            "input_ids": input_ids,
            "spec": compute_linear_spec(waveform),
            "waveform": waveform,
        }


def collate_fn(batch: list[dict]) -> dict:
    text_lens = [x["input_ids"].shape[0] for x in batch]
    spec_lens = [x["spec"].shape[1] for x in batch]
    wav_lens  = [x["waveform"].shape[0] for x in batch]
    B = len(batch)

    input_ids    = torch.zeros(B, max(text_lens), dtype=torch.long)
    attention_mask = torch.zeros(B, max(text_lens), dtype=torch.long)
    spec         = torch.zeros(B, batch[0]["spec"].shape[0], max(spec_lens))
    waveform     = torch.zeros(B, max(wav_lens))

    for i, x in enumerate(batch):
        tl, sl, wl = text_lens[i], spec_lens[i], wav_lens[i]
        input_ids[i, :tl] = x["input_ids"]
        attention_mask[i, :tl] = 1
        spec[i, :, :sl] = x["spec"]
        waveform[i, :wl] = x["waveform"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spec": spec,
        "spec_lengths": torch.tensor(spec_lens, dtype=torch.long),
        "waveform": waveform,
        "wav_lengths": torch.tensor(wav_lens, dtype=torch.long),
        "text_lengths": torch.tensor(text_lens, dtype=torch.long),
    }


# ── Discriminators (MPD + MSD) ────────────────────────────────────────────────


class DiscriminatorP(nn.Module):
    """One sub-discriminator of the Multi-Period Discriminator."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        pad = (kernel_size - 1) // 2
        wn = nn.utils.weight_norm
        self.convs = nn.ModuleList([
            wn(nn.Conv2d(1,    32,   (kernel_size, 1), (stride, 1), (pad, 0))),
            wn(nn.Conv2d(32,   128,  (kernel_size, 1), (stride, 1), (pad, 0))),
            wn(nn.Conv2d(128,  512,  (kernel_size, 1), (stride, 1), (pad, 0))),
            wn(nn.Conv2d(512,  1024, (kernel_size, 1), (stride, 1), (pad, 0))),
            wn(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,           (pad, 0))),
        ])
        self.conv_post = wn(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor):
        # x: [B, 1, T]
        B, C, T = x.shape
        if T % self.period != 0:
            x = F.pad(x, (0, self.period - T % self.period), "reflect")
        T2 = x.shape[-1]
        x = x.view(B, C, T2 // self.period, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p) for p in [2, 3, 5, 7, 11]]
        )

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real_outs, fake_outs, real_fmaps, fake_fmaps = [], [], [], []
        for d in self.discriminators:
            r, rf = d(real)
            f, ff = d(fake)
            real_outs.append(r);  real_fmaps.append(rf)
            fake_outs.append(f);  fake_fmaps.append(ff)
        return real_outs, fake_outs, real_fmaps, fake_fmaps


class DiscriminatorS(nn.Module):
    """One sub-discriminator of the Multi-Scale Discriminator."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128,  15, 1,  padding=7)),
            norm(nn.Conv1d(128,  128,  41, 2,  groups=4,  padding=20)),
            norm(nn.Conv1d(128,  256,  41, 2,  groups=16, padding=20)),
            norm(nn.Conv1d(256,  512,  41, 4,  groups=16, padding=20)),
            norm(nn.Conv1d(512,  1024, 41, 4,  groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1,  groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5,  1,  padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real_outs, fake_outs, real_fmaps, fake_fmaps = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                real = self.pooling[i - 1](real)
                fake = self.pooling[i - 1](fake)
            r, rf = d(real)
            f, ff = d(fake)
            real_outs.append(r);  real_fmaps.append(rf)
            fake_outs.append(f);  fake_fmaps.append(ff)
        return real_outs, fake_outs, real_fmaps, fake_fmaps


# ── Loss functions ────────────────────────────────────────────────────────────


def kl_divergence_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_q) + (z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)


def discriminator_loss(real_outs: list, fake_outs: list) -> torch.Tensor:
    """Least-squares GAN discriminator loss: real → 1, fake → 0."""
    return sum(
        torch.mean((1 - r) ** 2) + torch.mean(f ** 2)
        for r, f in zip(real_outs, fake_outs)
    )


def generator_adversarial_loss(fake_outs: list) -> torch.Tensor:
    """Generator wants discriminator to output 1 for fake samples."""
    return sum(torch.mean((1 - f) ** 2) for f in fake_outs)


def feature_matching_loss(real_fmaps: list, fake_fmaps: list) -> torch.Tensor:
    """L1 distance between discriminator feature maps of real and fake audio."""
    loss = 0.0
    for rf, ff in zip(real_fmaps, fake_fmaps):
        for r, f in zip(rf, ff):
            loss = loss + F.l1_loss(r.detach(), f)
    return loss * 2


# ── Generator forward pass ────────────────────────────────────────────────────


def forward_generator_pass(
    model: VitsModel,
    batch: dict,
    device: str,
    mel_transform,
    kl_weight: float = 1.0,
) -> dict:
    """
    Full VITS generator forward pass.

    Returns a dict with individual losses and the generated waveform so the
    training loop can feed it to the discriminator without re-running the generator.
    """
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    spec           = batch["spec"].to(device)
    spec_lengths   = batch["spec_lengths"].to(device)
    waveform       = batch["waveform"].to(device)
    text_lengths   = batch["text_lengths"].to(device)

    B, _, T_spec = spec.shape
    dtype = next(model.parameters()).dtype

    # Padding masks
    text_pad_tl1 = attention_mask.unsqueeze(-1).to(dtype)   # [B, T_text, 1]
    text_pad     = text_pad_tl1.transpose(1, 2)              # [B, 1, T_text]
    spec_mask    = torch.zeros(B, 1, T_spec, device=device, dtype=dtype)
    for i, sl in enumerate(spec_lengths):
        spec_mask[i, 0, :sl] = 1.0

    # Text encoder
    text_enc      = model.text_encoder(input_ids=input_ids, padding_mask=text_pad_tl1,
                                        attention_mask=attention_mask, return_dict=True)
    hidden_states = text_enc.last_hidden_state.transpose(1, 2)
    prior_means   = text_enc.prior_means.transpose(1, 2)
    prior_log_vars = text_enc.prior_log_variances.transpose(1, 2)

    # Posterior encoder
    z, mean_q, logs_q = model.posterior_encoder(spec, spec_mask)

    # Flow: z → z_p
    z_p = model.flow(z, spec_mask)

    # MAS
    with torch.no_grad():
        z_p_exp = z_p.unsqueeze(2)
        m_p_exp = prior_means.unsqueeze(3)
        lp_exp  = prior_log_vars.unsqueeze(3)
        log_p   = -0.5 * (
            math.log(2 * math.pi) + 2.0 * lp_exp
            + (z_p_exp - m_p_exp) ** 2 * torch.exp(-2.0 * lp_exp)
        ).sum(dim=1)
        attn_mask   = text_pad.transpose(1, 2) * spec_mask
        attn        = maximum_path_batch(log_p * attn_mask, attn_mask)

    attn_t         = attn.transpose(1, 2)
    m_p_aligned    = torch.bmm(prior_means,    attn_t)
    logs_p_aligned = torch.bmm(prior_log_vars, attn_t)

    # KL loss
    loss_kl = kl_divergence_loss(z_p, logs_q, m_p_aligned, logs_p_aligned, spec_mask)

    # Duration predictor loss
    dur_target = attn.sum(dim=1).unsqueeze(1)
    dur_nll    = model.duration_predictor(hidden_states, text_pad,
                                           durations=dur_target, reverse=False)
    loss_dur   = (dur_nll / text_lengths.float()).mean()

    # Decode
    waveform_gen = model.decoder(z * spec_mask)  # [B, 1, T_gen]

    # Mel reconstruction loss
    mel_ref = torch.stack([compute_mel_spec(waveform[b], mel_transform) for b in range(B)])
    mel_gen = torch.stack([compute_mel_spec(waveform_gen[b, 0], mel_transform) for b in range(B)])
    min_t   = min(mel_ref.shape[2], mel_gen.shape[2])
    loss_mel = F.l1_loss(mel_gen[:, :, :min_t], mel_ref[:, :, :min_t])

    return {
        "loss_mel":      loss_mel,
        "loss_kl":       loss_kl,
        "loss_dur":      loss_dur,
        "kl_weight":     kl_weight,
        "waveform_gen":  waveform_gen,          # [B, 1, T_gen]  — kept in graph
        "waveform_real": waveform.unsqueeze(1),  # [B, 1, T_real] — reference
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(
    epoch: int,
    model: VitsModel,
    mpd: MultiPeriodDiscriminator,
    msd: MultiScaleDiscriminator,
    optim_gen: torch.optim.Optimizer,
    optim_disc: torch.optim.Optimizer,
    sched_gen,
    sched_disc,
    train_loss: float,
    val_loss: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":     model.state_dict(),
            "mpd_state_dict":       mpd.state_dict(),
            "msd_state_dict":       msd.state_dict(),
            "optim_gen_state_dict":  optim_gen.state_dict(),
            "optim_disc_state_dict": optim_disc.state_dict(),
            "sched_gen_state_dict":  sched_gen.state_dict() if sched_gen  else None,
            "sched_disc_state_dict": sched_disc.state_dict() if sched_disc else None,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        },
        path,
    )
    logger.info("Saved checkpoint → %s", path)
    return path


def load_checkpoint(
    path: Path,
    model: VitsModel,
    mpd: MultiPeriodDiscriminator,
    msd: MultiScaleDiscriminator,
    optim_gen: torch.optim.Optimizer,
    optim_disc: torch.optim.Optimizer,
    sched_gen,
    sched_disc,
    device: str,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "mpd_state_dict" in ckpt:
        mpd.load_state_dict(ckpt["mpd_state_dict"])
        msd.load_state_dict(ckpt["msd_state_dict"])
    optim_gen.load_state_dict(ckpt["optim_gen_state_dict"])
    if "optim_disc_state_dict" in ckpt:
        optim_disc.load_state_dict(ckpt["optim_disc_state_dict"])
    if sched_gen  and ckpt.get("sched_gen_state_dict"):
        sched_gen.load_state_dict(ckpt["sched_gen_state_dict"])
    if sched_disc and ckpt.get("sched_disc_state_dict"):
        sched_disc.load_state_dict(ckpt["sched_disc_state_dict"])
    epoch = ckpt["epoch"]
    logger.info("Resumed from %s (epoch %d, val_loss=%.4f)", path, epoch, ckpt.get("val_loss", 0))
    return epoch


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ── Main training loop ────────────────────────────────────────────────────────


def train(args: SimpleNamespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s  |  VRAM total: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Generator (VITS) ─────────────────────────────────────────────────────
    logger.info("Loading model: %s", MODEL_ID)
    t0 = time.time()
    tokenizer = VitsTokenizer.from_pretrained(MODEL_ID)
    model     = VitsModel.from_pretrained(MODEL_ID).to(device)
    model.train()
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Generator loaded in %.1fs  |  %s total params, %s trainable",
                time.time() - t0, f"{total_params:,}", f"{trainable_params:,}")

    # ── Discriminators ────────────────────────────────────────────────────────
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mpd.train(); msd.train()
    disc_params = list(mpd.parameters()) + list(msd.parameters())
    disc_total  = sum(p.numel() for p in disc_params)
    logger.info("Discriminators (MPD+MSD): %s params", f"{disc_total:,}")

    if device == "cuda":
        logger.info("VRAM after models: %.1f GB", torch.cuda.memory_allocated() / 1e9)

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading local dataset…")
    t0 = time.time()

    min_align = args.min_align_score
    min_p808  = args.min_p808_mos
    min_ovr   = args.min_mos_ovr
    filters_active = (
        ([f"align_score>={min_align}"] if min_align is not None else []) +
        ([f"p808_mos>={min_p808}"]     if min_p808  is not None else []) +
        ([f"mos_ovr>={min_ovr}"]       if min_ovr   is not None else [])
    )

    def _keep(x):
        if min_align is not None and x.get("align_score", 0) < min_align:
            return False
        dns = x.get("dns_mos") or {}
        if min_p808 is not None and dns.get("p808_mos", 0) < min_p808:
            return False
        if min_ovr  is not None and dns.get("mos_ovr",  0) < min_ovr:
            return False
        return True

    logger.info("Loading local dataset%s…",
                f" (filters: {', '.join(filters_active)})" if filters_active else "")
    hf_ds = load_local_dataset(filter_fn=_keep if filters_active else None)
    logger.info("Dataset ready in %.1fs  |  %d samples", time.time() - t0, len(hf_ds))

    dataset   = KurmanjiTTSDataset(hf_ds, tokenizer)
    val_size  = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    logger.info("Train: %d  |  Val: %d  |  Steps/epoch: %d",
                train_size, val_size, math.ceil(train_size / args.batch_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn,
                               pin_memory=(device == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2),
                               shuffle=False, num_workers=args.num_workers,
                               collate_fn=collate_fn, pin_memory=(device == "cuda"))

    # ── Optimizers & schedulers ───────────────────────────────────────────────
    optim_gen  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    betas=(0.8, 0.99), weight_decay=args.weight_decay)
    optim_disc = torch.optim.AdamW(disc_params, lr=args.disc_lr,
                                    betas=(0.8, 0.99), weight_decay=args.weight_decay)
    sched_gen  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_gen, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    sched_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_disc, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    # Mixed precision
    use_amp    = (device == "cuda" and args.fp16)
    scaler_gen  = GradScaler("cuda", enabled=use_amp)
    scaler_disc = GradScaler("cuda", enabled=use_amp)

    mel_transform = build_mel_transform(device)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    output_dir  = Path(args.output_dir)
    if args.resume:
        start_epoch = load_checkpoint(
            Path(args.resume), model, mpd, msd,
            optim_gen, optim_disc, sched_gen, sched_disc, device,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    train_start      = time.time()
    kl_warmup_steps  = int(args.kl_warmup_epochs * len(train_loader))
    global_step      = start_epoch * len(train_loader)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        model.train(); mpd.train(); msd.train()
        epoch_counts: dict[str, float] = {}
        epoch_loss_gen = 0.0

        logger.info("── Epoch %d/%d ──────────────────────────────", epoch, args.epochs)

        for step, batch in enumerate(train_loader):
            global_step += 1

            # KL weight ramp
            if kl_warmup_steps > 0:
                kl_weight = args.kl_weight_start + (1.0 - args.kl_weight_start) * min(
                    global_step / kl_warmup_steps, 1.0
                )
            else:
                kl_weight = 1.0

            optim_gen.zero_grad()
            optim_disc.zero_grad()

            # ── Generator forward (single pass, graph kept) ───────────────────
            with autocast("cuda", enabled=use_amp):
                out = forward_generator_pass(model, batch, device, mel_transform, kl_weight)

            waveform_gen  = out["waveform_gen"]
            waveform_real = out["waveform_real"].to(device)

            # Align lengths for discriminator
            min_len = min(waveform_gen.shape[2], waveform_real.shape[2])
            wav_gen_cut  = waveform_gen[:, :, :min_len]
            wav_real_cut = waveform_real[:, :, :min_len]

            # ── Discriminator step ────────────────────────────────────────────
            with autocast("cuda", enabled=use_amp):
                r_mpd, f_mpd, _, _ = mpd(wav_real_cut, wav_gen_cut.detach())
                r_msd, f_msd, _, _ = msd(wav_real_cut, wav_gen_cut.detach())
                loss_disc = discriminator_loss(r_mpd + r_msd, f_mpd + f_msd)

            scaler_disc.scale(loss_disc).backward()
            scaler_disc.unscale_(optim_disc)
            torch.nn.utils.clip_grad_norm_(disc_params, args.grad_clip)
            scaler_disc.step(optim_disc)
            scaler_disc.update()
            optim_disc.zero_grad()  # clear disc grads before gen backward

            # ── Generator step ────────────────────────────────────────────────
            with autocast("cuda", enabled=use_amp):
                _, f_mpd, rf_mpd, ff_mpd = mpd(wav_real_cut, wav_gen_cut)
                _, f_msd, rf_msd, ff_msd = msd(wav_real_cut, wav_gen_cut)

                loss_adv = generator_adversarial_loss(f_mpd + f_msd)
                loss_fm  = feature_matching_loss(rf_mpd + rf_msd, ff_mpd + ff_msd)

                loss_gen = (
                    out["loss_mel"]
                    + kl_weight * out["loss_kl"]
                    + out["loss_dur"]
                    + loss_adv
                    + loss_fm
                )

            scaler_gen.scale(loss_gen).backward()
            scaler_gen.unscale_(optim_gen)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler_gen.step(optim_gen)
            scaler_gen.update()

            # ── Logging ───────────────────────────────────────────────────────
            epoch_loss_gen += loss_gen.item()
            log = {
                "loss/gen":  loss_gen.item(),
                "loss/disc": loss_disc.item(),
                "loss/mel":  out["loss_mel"].item(),
                "loss/kl":   out["loss_kl"].item(),
                "loss/dur":  out["loss_dur"].item(),
                "loss/adv":  loss_adv.item(),
                "loss/fm":   loss_fm.item(),
                "kl_weight": kl_weight,
            }
            for k, v in log.items():
                epoch_counts[k] = epoch_counts.get(k, 0.0) + v

            if (step + 1) % args.log_interval == 0:
                avg = {k: v / (step + 1) for k, v in epoch_counts.items()}
                elapsed    = time.time() - epoch_start
                steps_done = step + 1
                eta        = elapsed / steps_done * (len(train_loader) - steps_done)
                mem        = f"  VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB" if device == "cuda" else ""
                logger.info(
                    "  step %d/%d | %s | grad_norm=%.3f | ETA %ds%s",
                    steps_done, len(train_loader),
                    "  ".join(f"{k}={v:.4f}" for k, v in avg.items()),
                    grad_norm, int(eta), mem,
                )

        avg_train  = epoch_loss_gen / len(train_loader)
        epoch_time = time.time() - epoch_start

        # ── Validation ────────────────────────────────────────────────────────
        model.eval(); mpd.eval(); msd.eval()
        val_mel = 0.0
        val_kl  = 0.0
        val_dur = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with autocast("cuda", enabled=use_amp):
                    out = forward_generator_pass(model, batch, device, mel_transform)
                    val_mel += out["loss_mel"].item()
                    val_kl  += out["loss_kl"].item()
                    val_dur += out["loss_dur"].item()
        val_mel /= len(val_loader)
        val_kl  /= len(val_loader)
        val_dur /= len(val_loader)

        # Step schedulers on val_mel (the meaningful quality metric)
        sched_gen.step(val_mel)
        sched_disc.step(val_mel)

        eta_total = (time.time() - train_start) / (epoch - start_epoch) * (args.epochs - epoch)
        logger.info(
            "Epoch %d/%d — gen=%.4f  val_mel=%.4f  val_kl=%.2f  val_dur=%.4f  lr_gen=%.2e  epoch_time=%ds  ETA %s",
            epoch, args.epochs, avg_train, val_mel, val_kl, val_dur,
            optim_gen.param_groups[0]["lr"], int(epoch_time), _fmt_duration(eta_total),
        )

        # ── Checkpointing ─────────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            save_checkpoint(epoch, model, mpd, msd, optim_gen, optim_disc,
                            sched_gen, sched_disc, avg_train, val_mel, output_dir)

        if val_mel < best_val_loss:
            best_val_loss = val_mel
            best_path = output_dir / "best_model"
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info("New best val_mel=%.4f — saved to %s", best_val_loss, best_path)

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Training complete in %s. Final model → %s",
                _fmt_duration(time.time() - train_start), final_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

_CONFIGS_PATH = Path(__file__).parent / "finetune_configs.json"


def load_config(config_name: str, resume_override: str | None = None) -> SimpleNamespace:
    with open(_CONFIGS_PATH) as f:
        all_configs = json.load(f)
    if config_name not in all_configs:
        raise ValueError(f"Config '{config_name}' not found. Available: {', '.join(all_configs)}")
    cfg = all_configs[config_name]
    if resume_override is not None:
        cfg["resume"] = resume_override
    logger.info("Using config '%s': %s", config_name, cfg)
    return SimpleNamespace(**cfg)


def parse_args() -> tuple[str, str | None]:
    p = argparse.ArgumentParser(
        description="Fine-tune facebook/mms-tts-kmr-script_latin on Kurdish Kurmanji data"
    )
    p.add_argument("config", nargs="?", default="default",
                   help="Config name from finetune_configs.json (default: 'default')")
    p.add_argument("--resume", default=None,
                   help="Override the resume checkpoint path from the config")
    p.add_argument("--list", action="store_true", help="List available configs and exit")
    args = p.parse_args()

    if args.list:
        with open(_CONFIGS_PATH) as f:
            all_configs = json.load(f)
        print("Available configs:")
        for name, cfg in all_configs.items():
            print(f"  {name:12s}  epochs={cfg['epochs']}  batch_size={cfg['batch_size']}  lr={cfg['lr']}")
        raise SystemExit(0)

    return args.config, args.resume


if __name__ == "__main__":
    config_name, resume = parse_args()
    train(load_config(config_name, resume))
