# Deployment on Raspberry Pi

This guide covers deploying DOOM MultiVec on a Raspberry Pi Zero 2W for real-time DOOM gameplay.

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Board | Raspberry Pi Zero 2W |
| SoC | BCM2710A1 (Broadcom) |
| CPU | ARM Cortex-A53 Quad-Core @ 1 GHz |
| RAM | 512 MB LPDDR2 |
| Storage | microSD card, 16 GB+ (Class 10 / U1 minimum) |
| Power | 5V / 2.5A micro USB |
| Connectivity | WiFi 802.11 b/g/n, Bluetooth 4.2 |

!!! note "No GPU/NPU"
    The Pi Zero 2W has no hardware accelerator. All inference runs on the quad-core ARM CPU via ONNX Runtime.

### Resource Budgets

**Memory budget** (of 512 MB total):

| Component | Estimated RAM |
|---|---|
| OS + system services | ~80 MB |
| VizDoom engine | ~30 MB |
| Python interpreter + libraries | ~30 MB |
| ONNX Runtime + INT8 model | ~10 MB |
| Inference activations + buffers | ~15 MB |
| MaxSim working memory | ~1 MB |
| Headroom | ~100 MB |
| **Total used** | **~166 MB** |

**Latency budget** (target < 200 ms per frame):

| Step | Estimated |
|---|---|
| VizDoom frame capture | 10 ms |
| ASCII conversion + depth binning | 5 ms |
| Tokenization | 2 ms |
| ONNX inference (INT8, ARM) | 100-120 ms |
| Action selection + dispatch | 1 ms |
| **Total** | **~120-140 ms** |

This yields approximately 7-8 frames per second of reactive gameplay. On CPU (non-ARM), total latency is ~29ms.

---

## OS Setup

### Install Raspberry Pi OS Lite (64-bit)

!!! warning "64-bit is required"
    ONNX Runtime requires a 64-bit ARM (aarch64) OS. The 32-bit Raspberry Pi OS will not work.

1. Download **Raspberry Pi OS Lite (64-bit)** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Flash to microSD using Raspberry Pi Imager
3. Enable SSH and configure WiFi during flashing
4. Boot and connect via SSH

### Initial Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ (should be pre-installed on recent Pi OS)
python3 --version  # Expect 3.11+

# Install pip and venv
sudo apt install -y python3-pip python3-venv

# Create a virtual environment
python3 -m venv ~/doom-env
source ~/doom-env/bin/activate
```

---

## Software Installation

### Install DOOM MultiVec (inference only)

```bash
# Clone the repository
git clone https://github.com/david/doom-multivec.git
cd doom-multivec

# Install with raspi extras (minimal dependencies)
pip install -e ".[raspi]"
```

This installs only `numpy` and `onnxruntime` -- no PyTorch, no transformers.

### Install VizDoom on ARM

VizDoom must be built from source on ARM:

```bash
# Install build dependencies
sudo apt install -y cmake g++ libboost-all-dev libsdl2-dev \
    libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev \
    libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev \
    libgme-dev libopenal-dev zlib1g-dev timidity nasm

# Install VizDoom from source
pip install vizdoom
```

!!! tip "Build time"
    Building VizDoom from source on the Pi Zero 2W can take 30-60 minutes. Consider cross-compiling on a faster machine if possible.

If the pip install fails, build from source:

```bash
git clone https://github.com/Farama-Foundation/ViZDoom.git
cd ViZDoom
python setup.py build
pip install .
```

### Transfer Model Files

Copy the trained ONNX classifier model to the Pi:

```bash
# On your development machine:
scp doom_classifier_int8.onnx pi@raspberrypi:~/doom-multivec/models/
```

Alternatively, if you have the full trained model, export ONNX on the Pi (slower but avoids file transfer):

```bash
python scripts/export_onnx.py --model models/doom-multivec-trained --output models/doom_classifier_int8.onnx
```

---

## Running the Game Loop

### Basic Game Loop

```python
#!/usr/bin/env python3
"""Minimal DOOM game loop for Raspberry Pi."""

import time
import numpy as np
import onnxruntime as ort
from doom_multivec.ascii.converter import AsciiConverter

# --- Configuration ---
ONNX_MODEL = "models/doom_classifier_int8.onnx"
ACTION_NAMES = ["shoot", "move_forward", "turn_left", "turn_right"]
ASCII_WIDTH = 40
ASCII_HEIGHT = 25

# --- Setup ---
session = ort.InferenceSession(ONNX_MODEL)
converter = AsciiConverter(width=ASCII_WIDTH, height=ASCII_HEIGHT)

def classify_frame(ascii_text, depth_bins):
    """Classify an ASCII frame into action probabilities via ONNX."""
    # Character-level tokenization (simplified for deployment)
    # In production, use the saved tokenizer
    tokens = [2]  # [CLS]
    for ch in ascii_text:
        tokens.append(ord(ch) % 69)  # simplified mapping
    tokens.append(3)  # [SEP]

    input_ids = np.array([tokens], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    depth_input = np.array([depth_bins[:len(tokens)]], dtype=np.int64)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "depth_bins": depth_input,
    })

    # outputs[0] shape: (1, 4) -- raw logits
    logits = outputs[0][0]
    # Apply softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    return probs

def select_action(probs):
    """Select the best action from classifier probabilities."""
    best_idx = np.argmax(probs)
    return ACTION_NAMES[best_idx], dict(zip(ACTION_NAMES, probs.tolist()))

# --- Game Loop ---
# (Replace with actual VizDoom integration)
print("Starting DOOM MultiVec inference loop...")
print(f"Model: {ONNX_MODEL}")
print(f"Actions: {ACTION_NAMES}")
```

!!! note "Production tokenizer"
    The example above uses a simplified tokenization. For production, load the saved tokenizer with `PreTrainedTokenizerFast.from_pretrained()` and use its proper character-to-ID mapping.

---

## Performance Tuning

### CPU Governor

Set the CPU governor to `performance` for maximum clock speed:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Swap

Ensure swap is enabled as a safety net (though the model should fit in RAM):

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=256/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### ONNX Runtime Thread Configuration

Limit ONNX Runtime to the physical cores to avoid scheduling overhead:

```python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4  # Pi Zero 2W has 4 cores
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession("doom_classifier_int8.onnx", opts)
```

### Reduce ASCII Resolution

If latency exceeds 200ms, reduce the ASCII frame size:

```python
# 30x20 instead of 40x25 (40% fewer tokens)
converter = AsciiConverter(width=30, height=20)
```

This reduces the token sequence from ~1024 to ~620, cutting ONNX inference time by ~35%.

### Frame Skipping

Process every 2nd or 3rd VizDoom frame to increase effective FPS:

```python
frame_skip = 2  # Process every 2nd frame
# Previous action is held for skipped frames
```

---

## Fallback Options

If the model is too slow on the Pi Zero 2W, consider these fallbacks in order:

| Optimization | Impact | Tradeoff |
|---|---|---|
| Reduce layers (3 -> 2) | ~30% faster inference | Lower model capacity |
| Reduce ASCII resolution (40x25 -> 30x20) | ~35% fewer tokens to process | Less spatial detail |
| INT4 quantization | ~20% faster | Slightly lower accuracy |
| Skip every 2nd frame | 2x effective FPS | More reactive lag |
| Reduce document length (1100 -> 700) | Fewer tokens processed | Truncated frames |

---

## Monitoring

### Real-time Stats

Add performance monitoring to the game loop:

```python
import time

frame_times = []
while not game.is_episode_finished():
    t0 = time.perf_counter()

    # ... inference pipeline ...

    elapsed = time.perf_counter() - t0
    frame_times.append(elapsed)

    if len(frame_times) % 100 == 0:
        arr = np.array(frame_times[-100:])
        print(f"FPS: {1/arr.mean():.1f}  "
              f"Latency: {arr.mean()*1000:.0f}ms  "
              f"P95: {np.percentile(arr, 95)*1000:.0f}ms  "
              f"RAM: {get_ram_usage_mb():.0f}MB")
```

### Memory Monitoring

```python
import os

def get_ram_usage_mb():
    """Get current process RSS in MB."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0
```

### Target Metrics

| Metric | Target | Acceptable |
|---|---|---|
| Inference latency (P50) | < 150 ms | < 200 ms |
| Inference latency (P95) | < 180 ms | < 250 ms |
| FPS | > 6 | > 4 |
| Peak RAM | < 200 MB | < 256 MB |
| Stability | 10+ minutes | 5 minutes |
