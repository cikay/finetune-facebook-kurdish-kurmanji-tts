#!/usr/bin/env python3
"""
Fine-tune facebook/mms-tts-kmr-script_latin on local Kurdish Kurmanji dataset.

Architecture: VITS (Variational Inference Text-to-Speech)
Training objective: ELBO = mel reconstruction loss + KL divergence + duration predictor NLL
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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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

# Mel spectrogram for reconstruction loss (does not need to match model internals)
N_MELS = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0

# Training limits
MAX_WAV_SAMPLES = int(15.0 * SAMPLE_RATE)  # clip audio at 15 seconds
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
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)  # [F, T]


def compute_mel_spec(waveform: torch.Tensor, mel_transform) -> torch.Tensor:
    """Log-mel spectrogram [N_MELS, T_frames] from mono waveform [T]."""
    mel = mel_transform(waveform)
    return torch.log(torch.clamp(mel, min=1e-5))  # [N_MELS, T]


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
    """
    Pure-NumPy Monotonic Alignment Search (MAS).

    Args:
        log_p: [T_text, T_spec] log probability matrix
    Returns:
        attn: [T_spec, T_text] binary alignment (one-hot per row)
    """
    T_text, T_spec = log_p.shape

    # Dynamic programming
    Q = np.full((T_text, T_spec), -np.inf, dtype=np.float64)
    Q[0, 0] = log_p[0, 0]
    for j in range(1, T_spec):
        Q[0, j] = Q[0, j - 1] + log_p[0, j]
    for i in range(1, T_text):
        Q[i, i] = Q[i - 1, i - 1] + log_p[i, i]
        for j in range(i + 1, T_spec):
            Q[i, j] = max(Q[i, j - 1], Q[i - 1, j - 1]) + log_p[i, j]

    # Traceback
    attn = np.zeros((T_spec, T_text), dtype=np.float32)
    i = T_text - 1
    for j in range(T_spec - 1, -1, -1):
        attn[j, i] = 1.0
        if i > 0 and (j == 0 or Q[i - 1, j - 1] > Q[i, j - 1]):
            i -= 1

    return attn


def maximum_path_batch(
    log_p: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """
    Batched MAS — runs on CPU, detached from the computation graph.

    Args:
        log_p:    [B, T_text, T_spec] log probability matrix
        attn_mask:[B, T_text, T_spec] valid-region mask (1 = valid)
    Returns:
        attn:     [B, T_spec, T_text] binary alignment
    """
    B = log_p.shape[0]
    lp_np = log_p.detach().cpu().numpy()
    mask_np = attn_mask.detach().cpu().numpy().astype(bool)

    results = []
    for b in range(B):
        t_text = int(mask_np[b].any(axis=1).sum())
        t_spec = int(mask_np[b].any(axis=0).sum())
        lp = lp_np[b, :t_text, :t_spec]
        attn_b = _mas_numpy(lp)
        full = np.zeros((log_p.shape[2], log_p.shape[1]), dtype=np.float32)
        full[:t_spec, :t_text] = attn_b
        results.append(full)

    return torch.from_numpy(np.stack(results)).to(log_p.device)  # [B, T_spec, T_text]


# ── Dataset ───────────────────────────────────────────────────────────────────


class KurmanjiTTSDataset(Dataset):
    """
    Wraps the HuggingFace Dataset produced by load_dataset_from_local.load_dataset().

    Each item returns pre-computed linear + mel spectrograms and tokenized text.
    Spectrograms are computed on-the-fly to avoid storing large tensors.
    """

    def __init__(self, hf_dataset, tokenizer):
        self.data = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        text: str = item["text"]
        audio_array: np.ndarray = item["audio"]["array"].astype(np.float32)

        # Clip to max duration
        audio_array = audio_array[:MAX_WAV_SAMPLES]
        waveform = torch.from_numpy(audio_array)  # [T]

        # Tokenize (no device placement — stays on CPU for collation)
        token_out = self.tokenizer(text, return_tensors="pt")
        input_ids = token_out.input_ids.squeeze(0)  # [T_text]
        if len(input_ids) > MAX_TEXT_TOKENS:
            input_ids = input_ids[:MAX_TEXT_TOKENS]

        # Spectrograms (CPU)
        spec = compute_linear_spec(waveform)  # [513, T_frames]

        return {
            "input_ids": input_ids,
            "spec": spec,
            "waveform": waveform,
            "text": text,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad batch items to uniform lengths."""
    text_lens = [x["input_ids"].shape[0] for x in batch]
    spec_lens = [x["spec"].shape[1] for x in batch]
    wav_lens = [x["waveform"].shape[0] for x in batch]

    B = len(batch)
    spec_bins = batch[0]["spec"].shape[0]  # 513

    max_text = max(text_lens)
    max_spec = max(spec_lens)
    max_wav = max(wav_lens)

    input_ids = torch.zeros(B, max_text, dtype=torch.long)
    attention_mask = torch.zeros(B, max_text, dtype=torch.long)
    spec = torch.zeros(B, spec_bins, max_spec)
    waveform = torch.zeros(B, max_wav)
    spec_lengths = torch.tensor(spec_lens, dtype=torch.long)
    text_lengths = torch.tensor(text_lens, dtype=torch.long)
    wav_lengths = torch.tensor(wav_lens, dtype=torch.long)

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
        "spec_lengths": spec_lengths,
        "waveform": waveform,
        "wav_lengths": wav_lengths,
        "text_lengths": text_lengths,
    }


# ── Loss helpers ──────────────────────────────────────────────────────────────


def kl_divergence_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form KL divergence KL(q || p), masked and averaged.

    All tensors: [B, C, T].  z_mask: [B, 1, T].
    """
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_q) + (z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)


# ── Training step ─────────────────────────────────────────────────────────────


def forward_generator(
    model: VitsModel,
    batch: dict,
    device: str,
    mel_transform,
) -> tuple[torch.Tensor, dict]:
    """
    One generator forward pass.

    Returns:
        loss:         total scalar loss
        loss_dict:    named breakdown for logging
    """
    input_ids = batch["input_ids"].to(device)             # [B, T_text]
    attention_mask = batch["attention_mask"].to(device)    # [B, T_text]
    spec = batch["spec"].to(device)                        # [B, 513, T_spec]
    spec_lengths = batch["spec_lengths"].to(device)        # [B]
    waveform = batch["waveform"].to(device)                # [B, T_wav]
    text_lengths = batch["text_lengths"].to(device)        # [B]

    B, _, T_spec = spec.shape
    T_text = input_ids.shape[1]
    dtype = next(model.parameters()).dtype

    # Padding masks —————————————————————————————————————————————————————————
    # text_encoder expects [B, T_text, 1]
    text_pad_mask_tl1 = attention_mask.unsqueeze(-1).to(dtype)  # [B, T_text, 1]
    # flow/posterior/decoder expect [B, 1, T]
    text_pad_mask = text_pad_mask_tl1.transpose(1, 2)           # [B, 1, T_text]

    spec_mask = torch.zeros(B, 1, T_spec, device=device, dtype=dtype)
    for i, sl in enumerate(spec_lengths):
        spec_mask[i, 0, :sl] = 1.0

    # ── Text encoder ────────────────────────────────────────────────────────
    text_enc = model.text_encoder(
        input_ids=input_ids,
        padding_mask=text_pad_mask_tl1,
        attention_mask=attention_mask,
        return_dict=True,
    )
    # Shapes after transpose: [B, C, T_text]
    hidden_states = text_enc.last_hidden_state.transpose(1, 2)
    prior_means = text_enc.prior_means.transpose(1, 2)
    prior_log_vars = text_enc.prior_log_variances.transpose(1, 2)

    # ── Posterior encoder: spec → (z, mean_q, logs_q) ───────────────────────
    z, mean_q, logs_q = model.posterior_encoder(spec, spec_mask)
    # z: [B, flow_size, T_spec]

    # ── Flow: z → z_p  (encoder direction, no reverse) ─────────────────────
    z_p = model.flow(z, spec_mask)  # [B, flow_size, T_spec]

    # ── MAS: compute log P(z_p | prior) and find alignment ──────────────────
    with torch.no_grad():
        # [B, C, T_spec] ⊗ [B, C, T_text] → [B, T_text, T_spec] pairwise log-likelihood
        z_p_exp = z_p.unsqueeze(2)              # [B, C, 1,      T_spec]
        m_p_exp = prior_means.unsqueeze(3)      # [B, C, T_text, 1     ]
        lp_exp  = prior_log_vars.unsqueeze(3)   # [B, C, T_text, 1     ]

        log_p = -0.5 * (
            math.log(2 * math.pi)
            + 2.0 * lp_exp
            + (z_p_exp - m_p_exp) ** 2 * torch.exp(-2.0 * lp_exp)
        ).sum(dim=1)  # [B, T_text, T_spec]

        # Valid region mask for MAS
        attn_mask = text_pad_mask.transpose(1, 2) * spec_mask  # [B, T_text, T_spec]
        log_p_masked = log_p * attn_mask

        attn = maximum_path_batch(log_p_masked, attn_mask)  # [B, T_spec, T_text]

    # ── Align prior to audio-frame resolution via attention ─────────────────
    # attn: [B, T_spec, T_text]; prior_means: [B, C, T_text]
    # m_p_aligned[b,c,j] = Σ_i attn[b,j,i] * prior_means[b,c,i]
    attn_t = attn.transpose(1, 2)  # [B, T_text, T_spec]
    m_p_aligned   = torch.bmm(prior_means,   attn_t)  # [B, C, T_spec]
    logs_p_aligned = torch.bmm(prior_log_vars, attn_t) # [B, C, T_spec]

    # ── KL loss ─────────────────────────────────────────────────────────────
    loss_kl = kl_divergence_loss(z_p, logs_q, m_p_aligned, logs_p_aligned, spec_mask)

    # ── Duration predictor loss ──────────────────────────────────────────────
    # Duration target: number of spec frames assigned to each text token
    dur_target = attn.sum(dim=1).unsqueeze(1)  # [B, 1, T_text]

    dur_nll = model.duration_predictor(
        hidden_states,      # [B, C, T_text]
        text_pad_mask,      # [B, 1, T_text]
        durations=dur_target,
        reverse=False,
    )
    # dur_nll: [B] — average over batch and normalise by text length
    loss_dur = (dur_nll / text_lengths.float()).mean()

    # ── Decoder: z → waveform ────────────────────────────────────────────────
    waveform_gen = model.decoder(z * spec_mask)  # [B, 1, T_wav_gen]
    waveform_gen_squeezed = waveform_gen.squeeze(1)  # [B, T_wav_gen]

    # ── Mel reconstruction loss ──────────────────────────────────────────────
    # Compute mel for generated audio; compare against reference mel
    mel_ref = torch.stack(
        [compute_mel_spec(waveform[b], mel_transform) for b in range(B)]
    )  # [B, N_MELS, T_frames_ref]

    mel_gen = torch.stack(
        [compute_mel_spec(waveform_gen_squeezed[b], mel_transform) for b in range(B)]
    )  # [B, N_MELS, T_frames_gen]

    min_t = min(mel_ref.shape[2], mel_gen.shape[2])
    loss_mel = F.l1_loss(mel_gen[:, :, :min_t], mel_ref[:, :, :min_t])

    # ── Total loss ───────────────────────────────────────────────────────────
    total_loss = loss_mel + loss_kl + loss_dur

    return total_loss, {
        "loss/total": total_loss.item(),
        "loss/mel": loss_mel.item(),
        "loss/kl": loss_kl.item(),
        "loss/dur": loss_dur.item(),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(
    epoch: int,
    model: VitsModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loss: float,
    val_loss: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )
    logger.info("Saved checkpoint → %s", path)
    return path


def load_checkpoint(
    path: Path,
    model: VitsModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt["epoch"]
    logger.info(
        "Resumed from %s (epoch %d, val_loss=%.4f)", path, epoch, ckpt.get("val_loss", 0)
    )
    return epoch


# ── Main training loop ────────────────────────────────────────────────────────


def train(args: SimpleNamespace) -> None:
    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Load tokenizer & model ────────────────────────────────────────────────
    logger.info("Loading model: %s", MODEL_ID)
    tokenizer = VitsTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.train()

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading local dataset…")
    hf_ds = load_local_dataset()

    # Optional: filter low-quality segments
    if args.min_align_score is not None:
        before = len(hf_ds)
        hf_ds = hf_ds.filter(
            lambda x: x.get("align_score", 0) >= args.min_align_score
        )
        logger.info(
            "Filtered by align_score >= %.1f: %d → %d samples",
            args.min_align_score,
            before,
            len(hf_ds),
        )

    dataset = KurmanjiTTSDataset(hf_ds, tokenizer)

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info("Train: %d  |  Val: %d", train_size, val_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.8, 0.99),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # Mixed precision scaler (only effective on CUDA)
    scaler = GradScaler(enabled=(device == "cuda" and args.fp16))

    # ── Mel transform (shared) ────────────────────────────────────────────────
    mel_transform = build_mel_transform(device)

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch = 0
    output_dir = Path(args.output_dir)
    if args.resume:
        start_epoch = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_counts: dict[str, float] = {}

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            with autocast(enabled=(device == "cuda" and args.fp16)):
                loss, loss_dict = forward_generator(model, batch, device, mel_transform)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_counts[k] = epoch_counts.get(k, 0.0) + v

            if (step + 1) % args.log_interval == 0:
                avg = {k: v / (step + 1) for k, v in epoch_counts.items()}
                logger.info(
                    "Epoch %d | step %d/%d | %s",
                    epoch,
                    step + 1,
                    len(train_loader),
                    "  ".join(f"{k}={v:.4f}" for k, v in avg.items()),
                )

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with autocast(enabled=(device == "cuda" and args.fp16)):
                    loss, _ = forward_generator(model, batch, device, mel_transform)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  lr=%.2e",
            epoch,
            args.epochs,
            avg_train,
            val_loss,
            optimizer.param_groups[0]["lr"],
        )

        # ── Checkpointing ─────────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, avg_train, val_loss, output_dir)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model"
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info("New best val_loss=%.4f — saved to %s", best_val_loss, best_path)

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Training complete. Final model saved to %s", final_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

_CONFIGS_PATH = Path(__file__).parent / "finetune_configs.json"


def load_config(config_name: str, resume_override: str | None = None) -> SimpleNamespace:
    with open(_CONFIGS_PATH) as f:
        all_configs = json.load(f)

    if config_name not in all_configs:
        available = ", ".join(all_configs)
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")

    cfg = all_configs[config_name]
    if resume_override is not None:
        cfg["resume"] = resume_override

    logger.info("Using config '%s': %s", config_name, cfg)
    return SimpleNamespace(**cfg)


def parse_args() -> tuple[str, str | None]:
    p = argparse.ArgumentParser(
        description="Fine-tune facebook/mms-tts-kmr-script_latin on Kurdish Kurmanji data"
    )
    p.add_argument(
        "config",
        nargs="?",
        default="default",
        help="Config name from finetune_configs.json (default: 'default')",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Override the resume checkpoint path from the config",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available configs and exit",
    )
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
