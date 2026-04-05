# Quick Start

This guide walks through the minimal steps to create a model, collect data, train, and run inference. All commands are copy-paste-runnable from the project root.

---

## 1. Create the Model

```bash
python scripts/create_model.py
```

This creates a 5-layer ModernBERT-Hash model with:

- ~1.3M parameters
- Character-level ASCII tokenizer (69 tokens)
- Hash embeddings (16 projections)
- Depth embeddings (16 bins x 128)
- Attention pooling + Linear(128, 4) classification head

To verify:

```bash
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('models/doom-multivec-5L', trust_remote_code=True)
print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')
"
```

---

## 2. Record Training Data

The recommended approach is recording human gameplay demonstrations with real depth data:

```bash
python scripts/record_human.py \
    --scenario defend_the_center \
    --output data/human-demos \
    --frame-skip 4
```

This opens a VizDoom window where you play the game. The script captures ASCII frames, depth buffer bins, and action scores from your keyboard input. Play for 5-10 minutes to collect several thousand frames.

Alternatively, download pre-recorded frames from HuggingFace (no depth data):

```bash
python scripts/collect_data.py \
    --mode classifier \
    --dataset arnaudstiegler/vizdoom-50-episodes-skipframe-4 \
    --max-frames 10000 \
    --scan-limit 250000 \
    --stride 5 \
    --output data/doom-cls-10k
```

This streams JPEG frames, converts each to 40x25 ASCII art, computes soft action scores from the PPO agent's action labels, and saves a dataset with `text` (ASCII frames) and `scores` (4-dim action vectors). The `--stride 5` flag spreads frame selection across the entire dataset (all 50 episodes) instead of taking the first 10,000 frames sequentially.

!!! tip "Small test dataset"
    For a quick smoke test, use `--max-frames 500 --output data/doom-cls-test-500`.

---

## 3. Train the Classifier

Train the encoder + attention pool + classification head:

```bash
python scripts/train_classifier.py \
    --data data/human-demos \
    --output output/my-classifier \
    --epochs 5 \
    --batch-size 64
```

For a minimal quick test (a few minutes on CPU):

```bash
python scripts/train_classifier.py \
    --data data/doom-cls-test-500 \
    --output output/test \
    --epochs 2 \
    --batch-size 8 \
    --eval-steps 50
```

For full training on a GPU server:

```bash
python scripts/train_classifier.py \
    --data data/human-demos \
    --output output/my-classifier \
    --epochs 10 \
    --batch-size 128 \
    --bf16 \
    --lr 3e-4 \
    --wandb
```

The trainer reports accuracy across all 4 actions at each `--eval-steps` interval. Best accuracy: 57.7% on 4-action classification (random baseline: 25%).

---

## 4. Watch the Model Play DOOM

After training, watch the model play in a VizDoom window:

```bash
python scripts/play_doom_visual.py \
    --model models/doom-multivec-trained \
    --scenario basic \
    --episodes 3
```

This opens a DOOM window where the classifier drives all movement, rotation, and combat decisions in real time. See [Gameplay](../guide/gameplay.md) for scenario options and tuning.

To quickly verify the model on a sample frame without VizDoom:

```python
import torch
from doom_multivec.model.classifier import DoomMultiVecClassifier
from transformers import AutoTokenizer

model_path = "models/doom-multivec-trained"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DoomMultiVecClassifier(model_path, pool_mode='attention')
model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location='cpu'))
model.eval()

# Create a sample ASCII frame (40x25 characters)
sample_frame = "\n".join([
    "  .:-=+*#%@" * 3 + "  .:-=+*",
] * 12 + [
    "  ...  E  ...  ###  ...  H  ...",
] + [
    "  .:-=+*#%@" * 3 + "  .:-=+*",
] * 12)

encoded = tokenizer(sample_frame, return_tensors='pt', max_length=1100,
                     padding='max_length', truncation=True)
with torch.no_grad():
    result = model(encoded['input_ids'], encoded['attention_mask'])
    probs = torch.softmax(result['logits'], dim=-1)[0]

for i, name in enumerate(DoomMultiVecClassifier.ACTION_NAMES):
    print(f"  {name:15s}: {probs[i]:.3f}")
```

---

## Next Steps

- [Architecture](../guide/architecture.md) -- Understand the model design and classification head
- [Data Pipeline](../guide/data-pipeline.md) -- Learn about datasets and teacher scoring
- [Training](../guide/training.md) -- Full training guide with hyperparameter details
- [Inference](../guide/inference.md) -- Classifier inference and ONNX export
- [Gameplay](../guide/gameplay.md) -- Watch the model play DOOM
- [Deployment](../guide/deployment.md) -- Running on Raspberry Pi Zero 2W
