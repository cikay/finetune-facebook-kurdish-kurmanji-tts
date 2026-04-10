# Fine tunning Facebook Kurdish Kurmanji TTS

## Dataset Creation

A pipeline to build a paired audio-text dataset for Kurdish Kurmanji, suitable for fine-tuning TTS and ASR models.

It downloads news readings from a YouTube playlist, matches them with article text scraped from [azadyawelat.com](https://azadyawelat.com), then segments long audio into short utterances using CTC forced alignment.

**Published dataset:** [muzaffercky/azadiya-welat-kurdish-kurmanji-voice](https://huggingface.co/datasets/muzaffercky/azadiya-welat-kurdish-kurmanji-voice)

### Setup

```bash
pip install pipenv
pipenv shell
pipenv install
```

#### RunPod (CUDA Image) Note

If you use `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, keep PyTorch aligned with the image stack:

```bash
pipenv run pip install --no-cache-dir --force-reinstall torch==2.8.0 torchaudio==2.8.0 nvidia-cusparselt-cu12==0.7.1
pipenv run pip install "fsspec[http]<=2026.2.0,>=2023.1.0"
pipenv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Expected output should start with `2.8.0` and CUDA `12.8`.


### Pipeline

Run the end-to-end pipeline:

```bash
python -m finetune_tts.dataset_creator
```

`dataset_creator.py` builds a simple block pipeline with:

1. `DownloadYoutubeAudioAndTextBlock`
2. `SegmentationBlock`

#### DownloadYoutubeAudioAndTextBlock

Downloads paired audio/text data from YouTube and Azadiya Welat:

1. Fetches playlist metadata with `yt-dlp`.
2. Downloads each video's audio as 16kHz mono WAV.
3. Matches article text using title slug on `azadyawelat.com`.
4. Writes `metadata.jsonl` and `playlist_info.json`.

#### SegmentationBlock

| Variable | Default | Description |
|---|---|---|
| `MIN_DURATION` | `2.0` | Minimum segment duration (seconds) |
| `MAX_DURATION` | `15.0` | Maximum segment duration (seconds) |
| `MIN_WORDS` | `3` | Minimum words per segment |
| `MIN_SCORE` | `-7.0` | Minimum alignment confidence score |


Splits long audio (~5 min each) into short utterances using forced alignment:

1. Loads the [MMS-300M forced alignment model](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) (supports 1,130+ languages including Kurdish).
2. Aligns ground truth text to audio using CTC forced alignment — no ASR transcription involved.
3. Maps word-level timestamps back to sentence boundaries.
4. Filters segments by duration (2–15s), word count (≥3), and alignment confidence.
5. Saves segmented WAV files and `segments_metadata.jsonl`.

#### Publish Dataset (`push_dataset.py`)
```bash
python push_dataset.py --repo your-username/your-dataset-name
```

Uploads the segmented dataset to HuggingFace Hub.

### Output Structure

```text
test_dataset/
├── audio/
├── text/
├── metadata.jsonl
├── playlist_info.json
├── audio_segments/
└── segments_metadata.jsonl
```

## Fine tunning

### Setup

Install dependencies (same Pipfile as dataset creation).

### Run

Training is configured via named configs in `finetune_tts/finetune_configs.json`.

```bash
# List available configs
python -m finetune_tts.finetune --list

# Run with default config (100 epochs, batch=8, lr=2e-4)
python -m finetune_tts.finetune

# Run a specific config
python -m finetune_tts.finetune fast
python -m finetune_tts.finetune quality

# Resume from a checkpoint (overrides the config's resume field)
python -m finetune_tts.finetune quality --resume checkpoints/kmr_quality/checkpoint_epoch_0050.pt
```

### Configs

Configs are defined in `finetune_tts/finetune_configs.json`. Edit that file to tune hyperparameters or add new configs.

| Config | Epochs | Batch | LR | fp16 | Notes |
|---|---|---|---|---|---|
| `default` | 100 | 8 | 2e-4 | off | Balanced baseline |
| `fast` | 30 | 16 | 5e-4 | on | Quick iteration |
| `quality` | 200 | 4 | 1e-4 | on | High quality, filters low-score segments |
| `debug` | 2 | 2 | 2e-4 | off | Smoke-test, 0 workers |

### Config fields

| Field | Description |
|---|---|
| `output_dir` | Where to save checkpoints and final model |
| `epochs` | Training epochs |
| `batch_size` | Batch size |
| `lr` | Initial learning rate |
| `lr_decay` | ExponentialLR gamma per epoch |
| `weight_decay` | AdamW weight decay |
| `grad_clip` | Gradient norm clip value |
| `fp16` | Mixed-precision on CUDA |
| `num_workers` | DataLoader worker processes |
| `val_fraction` | Fraction of data used for validation |
| `save_every` | Save checkpoint every N epochs |
| `log_interval` | Log loss every N steps |
| `seed` | Random seed |
| `min_align_score` | Filter segments below this CTC alignment score (null = no filter) |
| `resume` | Path to checkpoint to resume from (null = start fresh) |

### Output

```
checkpoints/kmr/
├── checkpoint_epoch_0005.pt   ← optimizer + model state (resumable)
├── checkpoint_epoch_0010.pt
├── best_model/                ← HuggingFace model dir (best val loss)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── final_model/               ← HuggingFace model dir (end of training)
```

### Training objective

VITS ELBO: mel-spectrogram L1 reconstruction + KL divergence (posterior ‖ prior after flow) + stochastic duration predictor NLL, with Monotonic Alignment Search (MAS) for text↔audio alignment.

### Inference after fine-tuning

```python
from transformers import VitsModel, VitsTokenizer
import torch, soundfile as sf

model = VitsModel.from_pretrained("checkpoints/kmr/best_model")
tokenizer = VitsTokenizer.from_pretrained("checkpoints/kmr/best_model")

inputs = tokenizer("Ez li Kurdistanê dijîm.", return_tensors="pt")
with torch.no_grad():
    output = model(**inputs)

sf.write("output.wav", output.waveform[0].numpy(), samplerate=16000)
```

