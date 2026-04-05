# Training

DOOM MultiVec is trained as a multi-vector classifier using `scripts/train_classifier.py`. The model learns to classify ASCII game frames (with depth data) into 4 actions using KL-divergence loss on soft action scores from human gameplay demonstrations.

---

## Prerequisites

Before training, you need:

1. **A base model** -- created with `scripts/create_model.py`
2. **Training data** -- either human gameplay recordings from `scripts/record_human.py` or HuggingFace datasets processed via `scripts/collect_data.py`

See the [Quick Start](../getting-started/quickstart.md) for the exact commands.

---

## Training Command

### Local Quick Test

Minimal training run for smoke testing (CPU, ~5 minutes):

```bash
python scripts/train_classifier.py \
    --data data/doom-cls-test-500 \
    --output output/test \
    --epochs 2 \
    --batch-size 8 \
    --eval-steps 50
```

### Full Training (GPU)

Full training on a CUDA GPU server with human demonstration data:

```bash
python scripts/train_classifier.py \
    --data data/human-demos \
    --output output/my-classifier \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4 \
    --bf16 \
    --wandb
```

### Apple Silicon (MPS)

On macOS with Apple Silicon, use `--fp16` instead of `--bf16`:

```bash
python scripts/train_classifier.py \
    --data data/human-demos \
    --output output/my-classifier \
    --epochs 10 \
    --batch-size 64 \
    --lr 3e-4 \
    --fp16
```

---

## Hyperparameters

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Epochs | `--epochs` | 5 | Full passes over the dataset |
| Batch size | `--batch-size` | 64 | Samples per training step |
| Learning rate | `--lr` | 3e-4 | Peak LR (cosine schedule with warmup) |
| Warmup steps | `--warmup-steps` | 500 | Linear warmup steps |
| Eval steps | `--eval-steps` | 500 | Evaluate every N steps |
| Pool mode | `--pool` | attention | Pooling: `attention`, `mean`, or `cls` |
| bf16 | `--bf16` | off | Mixed precision (CUDA) |
| fp16 | `--fp16` | off | Mixed precision (MPS) |
| W&B | `--wandb` | off | Report to Weights & Biases |

!!! tip "Learning rate"
    The default 3e-4 works well for training from scratch with the 1.3M parameter model. If the loss diverges early in training, try reducing to 1e-4.

!!! warning "Memory"
    Sequence length ~1100 with batch size 128 requires ~8GB GPU memory. On GPUs with less memory, reduce batch size to 64 or 32.

---

## Training Pipeline Details

### Loss Function

Training uses **KL-divergence loss** on soft action scores. For each training sample:

1. The model encodes the ASCII frame (with depth embeddings) into per-token embeddings
2. Attention pooling collapses the token embeddings into a single 128-dim vector
3. The linear classification head produces 4 action logits
4. KL divergence is minimized between the predicted action distribution (softmax over logits) and the teacher score distribution (soft labels from human gameplay)

The soft labels come from human gameplay demonstrations where actions are recorded with confidence scores, not hard one-hot labels. This provides richer supervision: a frame where the human is moving forward while slightly turning produces high scores for both `move_forward` and `turn_left`.

### Training Data Format

Each training sample contains:

- **ASCII frame**: 40x25 character string (~1024 characters)
- **Depth bins**: 16-bin quantized depth values per token position
- **Action scores**: 4-dim soft label vector (one score per action)

### Checkpoint Management

The `CopyModelingCallback` ensures that `modeling_doom_hash.py` is copied into every checkpoint directory. This is required because the model uses `trust_remote_code=True` loading, which expects the custom modeling file alongside `config.json`.

---

## Evaluation

The trainer evaluates top-1 action prediction accuracy on a held-out set at every `--eval-steps` interval:

1. For each evaluation frame, run the classifier to get 4 action probabilities
2. The predicted action is the one with the highest probability
3. The correct action is the one with the highest teacher score
4. Report accuracy = correct predictions / total

**Baseline**: Random guessing gives 1/4 = 25% accuracy.

**Best result**: 57.7% accuracy on 4-action classification with 31K human demonstration frames.

The trainer saves the best checkpoint based on evaluation accuracy (higher is better).

---

## Expected Training Curves

### Loss

- **Initial**: KL-divergence loss starts at ~1.0-1.5 (random model)
- **After 1 epoch**: Should drop below 0.8
- **Convergence**: Typically reaches 0.4-0.6 after 5-10 epochs

### Action Accuracy

- **Initial**: ~25% (random chance for 4 actions)
- **After 1 epoch**: ~35-45%
- **After 5 epochs**: ~50-55%
- **After 10 epochs**: ~55-60%

!!! note "Accuracy ceiling"
    Action accuracy above 70% may not be achievable because some frames are genuinely ambiguous -- multiple actions could be equally valid (e.g., a corridor with no enemies ahead could warrant moving forward or turning).

---

## Class Balancing

The DOOM gameplay data is heavily imbalanced -- `move_forward` dominates while `shoot` is rare. The training pipeline uses oversampling of minority actions (particularly `shoot`) to ensure every action class has adequate representation. Without balancing, the model learns to predict `move_forward` for every frame and ignores combat actions entirely.

---

## W&B Tracking

Pass `--wandb` to log training loss, learning rate, and evaluation accuracy to Weights & Biases:

```bash
wandb login
python scripts/train_classifier.py --data data/human-demos --epochs 10 --bf16 --wandb
```

Metrics logged: `loss`, `lr`, `epoch`, `eval_accuracy`.

---

## Legacy: MaxSim Knowledge Distillation

<details>
<summary>MaxSim KD training (deprecated)</summary>

The original training pipeline used PyLate's knowledge distillation with MaxSim scoring. A teacher signal derived from PPO agent actions was distilled into a ColBERT model using `scripts/train.py`. This approach was abandoned because MaxSim scores collapsed -- all action queries received similar scores due to shared character tokens in the 69-character vocabulary. See [Architecture: Architecture Evolution](architecture.md#architecture-evolution) for the full explanation.

The `scripts/train.py` script and PyLate KD format are preserved in the codebase for reference but are not recommended for new training runs.

</details>

---

## Training at Scale

For a 31K frame human demonstration dataset with 10 epochs on a single GPU:

| GPU | Batch Size | Time Estimate |
|---|---|---|
| NVIDIA A100 (40GB) | 128 | ~15 minutes |
| NVIDIA RTX 3090 (24GB) | 64 | ~30 minutes |
| Apple M2 Pro (MPS) | 64 | ~45 minutes |
| CPU only | 16 | ~3 hours |

---

## Troubleshooting

### Loss does not decrease

- **Check data**: Verify the dataset loaded correctly with the expected number of samples
- **Lower learning rate**: Try `--lr 1e-4` or `--lr 5e-5`
- **Check batch size**: Very small batches (< 4) can cause unstable gradients

### Out of memory

- **Reduce batch size**: `--batch-size 32` or `--batch-size 16`
- **Disable mixed precision**: Remove `--bf16`/`--fp16` to use FP32 (slower but uses less peak memory)

### `modeling_doom_hash.py` not found in checkpoint

The `CopyModelingCallback` should handle this automatically. If checkpoints fail to load, manually copy:

```bash
cp models/doom-multivec-*/modeling_doom_hash.py output/my-classifier/checkpoint-*/
```

### Evaluator reports 25% accuracy (no improvement)

- Ensure the dataset has frames with meaningful action variation. If all frames have the same dominant action, the model has nothing to learn.
- Check that depth data is included in the training samples.

### W&B not logging

Ensure you are logged in:

```bash
wandb login
```

And pass the `--wandb` flag to the training script.
