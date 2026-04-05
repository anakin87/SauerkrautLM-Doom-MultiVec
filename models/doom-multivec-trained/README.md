---
tags:
- doom
- game-ai
- ascii
- ModernBERT
- hash-embeddings
- depth-aware
- attention-pooling
- classifier
- real-time
- edge-deployment
- tiny-model
pipeline_tag: text-classification
library_name: transformers
license: apache-2.0
datasets:
- VAGOsolutions/SauerkrautLM-Doom-MultiVec-31k
---

<img src="Logo.png" width="500" height="auto">

# SauerkrautLM-Doom-MultiVec-1.3M

**A tiny 1.3M parameter model that plays DOOM, outperforming LLMs up to 92,000x its size.**

<video controls width="100%" style="border-radius: 8px; margin: 16px 0;">
  <source src="https://vago-solutions.ai/wp-content/uploads/2026/04/1mio-parameter-plays-DOOM.mp4" type="video/mp4">
</video>

This model is a ModernBERT-Hash encoder with depth-aware token representations and an attention pooling classification head, trained on 31K human gameplay demonstrations to select actions from ASCII game frame representations in real time.

### Core Features and Innovations

- **178 frags in 10 episodes** (17.8 per episode) in VizDoom's `defend_the_center` scenario, more than all tested LLMs combined (13 frags total)
- **31ms inference on CPU**, enabling real-time gameplay at 35 FPS
- **Depth-aware ASCII encoding**: VizDoom depth buffer encoded as learned 16-bin token embeddings fused with character embeddings
- **ModernBERT-Hash architecture**: Hash embeddings + local/global attention + Flash Attention 2 support
- **Character-level tokenizer**: 75 tokens, no BPE, preserving full spatial granularity of ASCII frames

### David vs. Goliath: 1.3M Parameters vs. 120 Billion

With **1.3 million parameters** -- less than **1/92,000th the size** of Nemotron-120B -- SauerkrautLM-Doom-MultiVec achieves:
- **178 frags** vs 0 for GPT-4o-mini (proprietary)
- **178 frags** vs 3 for Nemotron-120B (120B, 92,000x larger)
- **178 frags** vs 2 for Qwen3.5-27B (27B, 20,000x larger)
- **178 frags** vs 8 for Gemini Flash Lite (proprietary)

All LLMs are **vision-capable multimodal models** evaluated on text (ASCII + depth), their strongest modality. Our tiny text-only model outperforms them on their home turf.

---

## Model Overview

**Model:** `VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M`\
**Architecture:** ModernBERT-Hash encoder + Attention Pooling + Linear Classifier\
**Task:** Real-time DOOM action classification from ASCII frames\
**Training Data:** 31,645 human gameplay demonstrations with depth annotations\
**License:** Apache 2.0\
**Model Size:** 1.3M parameters (~5MB FP32)

### Model Description
- **Model Type:** Multi-vector encoder with attention pooling classification head
- **Encoder:** 5-layer ModernBERT with hash embeddings (H=128, 4 heads)
- **Tokenizer:** Character-level, 75 tokens (no BPE)
- **Depth Bins:** 16 learned depth embeddings added to token representations
- **Actions:** 4 (shoot, move_forward, turn_left, turn_right)
- **Max Sequence Length:** 1,200 tokens
- **Training Loss:** KL-divergence on soft action scores
- **Inference Latency:** 31ms (CPU), 29ms (GPU)

### Architecture

```
DoomMultiVecClassifier(
  encoder: ModernBertHashModel(
    embeddings: HashEmbedding(75 vocab, 16 proj -> 128 dim)
    depth_embedding: DepthEmbedding(16 bins x 128 dim)
    layers: 5x TransformerLayer(H=128, heads=4, FFN=512)
  )
  attention_pool: Linear(128 -> 1)  # learned attention weights
  classifier: Linear(128 -> 4)      # action probabilities
)
```

### ModernBERT-Hash: Advancing Hash Embeddings to Modern Architectures

This model introduces **hash embeddings on the ModernBERT architecture**, a combination that has not been explored before. Previous work on hash embeddings for tiny language models ([NeuML's BERT-Hash](https://huggingface.co/NeuML/bert-hash-nano), Svenstrup et al. 2017) applied the technique to the original BERT architecture from 2018. We bring hash embeddings to **ModernBERT** (Warner et al. 2024), which provides several architectural advantages:

- **Rotary Position Embeddings (RoPE)** instead of learned absolute positions, enabling better generalization across sequence lengths
- **Alternating local + global attention**: Layers alternate between sliding-window local attention (w=128) and full global attention, matching the spatial structure of ASCII frames where local patterns (adjacent characters) and global context (arena layout) both matter
- **Flash Attention 2 support** for efficient GPU training with long sequences (~1,100 tokens per frame)
- **Pre-normalization** with RMSNorm for more stable training of tiny models

The hash embedding layer replaces the standard embedding table (`V x H`) with a two-stage projection: a compact lookup (`V x P`) followed by a linear projection (`P x H`), where P=16 is the projection dimension. For our 75-token vocabulary this reduces embedding parameters from 9,600 to 4,480 (53% reduction). While modest for a tiny vocabulary, the same architecture scales to standard vocabularies: at 30K tokens, hash embeddings reduce embedding parameters by **97%** (from 3.8M to 120K), which is the key enabler for sub-1M parameter language models.

Combined with **depth embeddings** (16 learned bins added to token representations), the model receives both spatial (what the character looks like) and distance (how far away it is) information at the token level, a novel input representation for game state encoding.

---

## Input Pipeline

<img src="pipeline.png" width="100%">

*From VizDoom game frame to model input: (a) RGB frame, (b) grayscale, (c) depth buffer, (d) ASCII brightness map (40x25), (e) depth bins with 16 quantization levels (red=near, green=far), (f) combined ASCII + depth representation as fed to the model.*

---

## Benchmark: DOOM defend_the_center

All agents receive identical input: ASCII frame (40x25) + depth map. Game settings match training conditions (640x480, HUD on, real-time pacing). Frags are tracked via VizDoom's per-step reward signal.

| Agent | Params | Episodes | Avg Survival | Max Survival | Total Frags | Latency |
|-------|--------|----------|-------------|-------------|-------------|---------|
| **SauerkrautLM-Doom-MultiVec-1.3M** | **1.3M** | **10** | **388** | **525** | **178** | **31ms** |
| GPT-4o-mini | proprietary | 10 | 104 | 228 | 0 | 646ms |
| Nemotron-120B | 120B | 5 | 88 | 104 | 3 | 8.9s |
| Qwen3.5-27B | 27B | 3 | 87 | 109 | 2 | 13.3s |
| Gemini Flash Lite | proprietary | 10 | 81 | 97 | 8 | 920ms |

**178 frags vs 13 for all LLMs combined.** GPT-4o-mini scores **zero frags** across 10 episodes (pure evasion). Our model actively engages enemies, turns to face them, and fires -- playing DOOM as intended.

---

## Quick Start

```python
import torch
from transformers import AutoTokenizer

# Load model
model_path = "VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

from doom_multivec.model.classifier import DoomMultiVecClassifier
state = torch.load(f"{model_path}/model.pt", map_location="cpu")
model = DoomMultiVecClassifier(model_path, pool_mode="attention", num_actions=4)
model.load_state_dict(state)
model.eval()

# Classify an ASCII frame
ascii_frame = "." * 1024  # 40x25 ASCII frame
encoded = tokenizer(ascii_frame, return_tensors="pt", max_length=1100,
                    padding="max_length", truncation=True)

with torch.no_grad():
    result = model(encoded["input_ids"], encoded["attention_mask"])
    probs = torch.softmax(result["logits"], dim=-1)[0]

actions = ["shoot", "move_forward", "turn_left", "turn_right"]
for action, prob in zip(actions, probs):
    print(f"  {action}: {prob:.3f}")
```

### Watch the Model Play DOOM

```bash
pip install vizdoom
python scripts/play_doom_visual.py --model models/doom-multivec-trained --scenario defend_the_center
```

### Run the LLM Benchmark

```bash
pip install openai
export OPENAI_API_KEY="your-key"
python scripts/benchmark.py --agent multivec --episodes 10 --realtime
python scripts/benchmark.py --agent gpt4mini --episodes 10 --realtime
```

---

## Parameter Budget

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Hash Embeddings (75 vocab, 16 proj) | 4,480 | 0.3% |
| Depth Embeddings (17 bins x 128) | 2,176 | 0.2% |
| Transformer Layers (x 5) | 1,312,000 | 99.1% |
| Attention Pool + Classifier | 644 | 0.05% |
| **Total** | **1,319,300** | **100%** |

---

## Paper

> **Playing DOOM with 1.3M Parameters: Specialized Small Models vs Large Language Models for Real-Time Game Control**
>
> David Golchinfar (VAGO Solutions, Germany), Daryoush Vaziri (University of Applied Sciences Bonn-Rhein-Sieg, Germany), Alexander Marquardt (CARE Laboratory, NAIST, Japan)

Available in the [project repository](https://github.com/VAGOsolutions/SauerkrautLM-Doom-MultiVec) under `paper/doom_multivec.pdf`.

---

## Acknowledgements

This work was developed using [VizDoom](https://vizdoom.cs.put.edu.pl/) as the game platform, [PyLate](https://github.com/lightonai/pylate) for initial multi-vector experiments, and the HuggingFace ecosystem for model development. The ModernBERT-Hash architecture builds on [NeuML's BERT-Hash models](https://huggingface.co/NeuML/bert-hash-nano). Training data includes human gameplay demonstrations and frames from the arnaudstiegler GameNGen reproduction datasets on HuggingFace.

DOOM is a registered trademark of id Software LLC. This project is not affiliated with or endorsed by id Software.

---

## Citation

```bibtex
@misc{SauerkrautLM-Doom-MultiVec,
  title={SauerkrautLM-Doom-MultiVec-1.3M: Playing DOOM with 1.3M Parameters},
  author={David Golchinfar and Daryoush Vaziri and Alexander Marquardt},
  url={https://huggingface.co/VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M},
  year={2026}
}
```
