# Installation

## System Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| OS | Linux, macOS, or Windows (for development/training) |
| RAM | 8 GB (training), 512 MB (inference on Raspberry Pi) |
| GPU | Optional -- CUDA or MPS for faster training |
| Disk | ~2 GB (for dependencies + datasets) |

---

## Install from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/david/doom-multivec.git
cd doom-multivec
pip install -e .
```

This installs the core dependencies needed for model creation and inference:

| Package | Purpose |
|---|---|
| `torch` | Model computation backend |
| `transformers` | ModernBERT base architecture |
| `sentence-transformers` | Training framework |
| `pylate` | Legacy ColBERT support (optional) |
| `numpy` | Array operations |
| `Pillow` | Image decoding for frame processing |
| `datasets` | HuggingFace dataset loading |

---

## Development Install

For development and training, install with the `dev` extras:

```bash
pip install -e ".[dev]"
```

This adds:

| Package | Purpose |
|---|---|
| `vizdoom` | DOOM game engine for live play and data collection |
| `wandb` | Experiment tracking during training |
| `onnx` | ONNX model export |
| `onnxruntime` | ONNX inference runtime |
| `pytest` | Test runner |
| `mkdocs-material` | Documentation site builder |
| `mkdocstrings[python]` | Auto-generated API docs from docstrings |

---

## Training Install (Remote GPU Server)

For training on a remote GPU server, you typically need CUDA-enabled PyTorch:

```bash
# Install PyTorch with CUDA support first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install the project
pip install -e ".[dev]"
```

!!! tip "Apple Silicon (MPS)"
    On macOS with Apple Silicon, standard `pip install torch` includes MPS support out of the box. Use `--fp16` instead of `--bf16` when training on MPS.

---

## Raspberry Pi Deployment Install

For inference-only deployment on a Raspberry Pi Zero 2W, install the minimal `raspi` extras:

```bash
pip install -e ".[raspi]"
```

This installs only:

| Package | Purpose |
|---|---|
| `onnxruntime` | ONNX Runtime for ARM inference |
| `numpy` | MaxSim computation |

!!! warning "ARM64 Required"
    The Raspberry Pi Zero 2W must run a 64-bit OS (e.g., Raspberry Pi OS Lite 64-bit). The 32-bit armhf builds of ONNX Runtime are not supported.

See the [Deployment Guide](../guide/deployment.md) for full Raspberry Pi setup instructions.

---

## Verification

After installation, verify everything works:

```bash
# Check that the model can be created
python scripts/create_model.py --output /tmp/doom-multivec-test

# Verify the model loads correctly
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('/tmp/doom-multivec-test', trust_remote_code=True)
print(f'Model loaded: {sum(p.numel() for p in m.parameters()):,} parameters')
"
```

Expected output:

```
Model loaded: ~1,300,000 parameters
```

To verify the tokenizer:

```bash
python -c "
from doom_multivec.model.tokenizer import create_ascii_tokenizer
tok = create_ascii_tokenizer()
print(f'Tokenizer vocab size: {tok.vocab_size}')
encoded = tok('E###@\n ...:', return_tensors='pt')
print(f'Encoded shape: {encoded[\"input_ids\"].shape}')
"
```

Expected output:

```
Tokenizer vocab size: 69
Encoded shape: torch.Size([1, 12])
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'vizdoom'`

VizDoom requires system-level dependencies. On Ubuntu/Debian:

```bash
sudo apt-get install cmake libboost-all-dev libsdl2-dev libfreetype6-dev \
    libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev \
    libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm
pip install vizdoom
```

On macOS:

```bash
brew install cmake boost sdl2
pip install vizdoom
```

### `trust_remote_code` errors

When loading the model, always pass `trust_remote_code=True`:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("models/doom-multivec-5L", trust_remote_code=True)
```

This is required because `ModernBertHashModel` is a custom architecture defined in `modeling_doom_hash.py`, which must be present in the model directory alongside `config.json`.

### PyTorch version conflicts

If you encounter version conflicts between `torch`, `transformers`, and `pylate`, pin PyTorch first:

```bash
pip install torch==2.2.0
pip install -e ".[dev]"
```
