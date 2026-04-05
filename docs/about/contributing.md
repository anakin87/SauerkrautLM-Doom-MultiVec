# Contributing

Thank you for your interest in contributing to DOOM MultiVec.

---

## Development Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/david/doom-multivec.git
    cd doom-multivec
    ```

2. Install with development dependencies:

    ```bash
    pip install -e ".[dev]"
    ```

3. Verify the installation:

    ```bash
    python scripts/create_model.py --output /tmp/doom-test
    pytest tests/
    ```

---

## Project Structure

```
doom_multivec/
  src/doom_multivec/
    ascii/              # Frame-to-ASCII conversion
      converter.py      # AsciiConverter class
      charset.py        # Character set constants
    model/              # Model architecture
      modeling_doom_hash.py  # HashEmbedding + ModernBertHashModel
      tokenizer.py      # Character-level tokenizer
      classifier.py     # Attention pool + classifier head
      colbert_wrapper.py     # Legacy PyLate ColBERT loading
    doom/               # VizDoom integration
      engine.py         # DoomEngine + MockDoomEngine
      scenarios.py      # Scenario configs
    training/           # Training pipeline
      action_mapping.py # 18-action to 4-action decomposition
      teacher.py        # Action score computation
      dataset.py        # Classifier dataset builder
    inference/          # Real-time inference
  scripts/              # CLI entry points
    create_model.py     # Create model from scratch
    create_tokenizer.py # Create and save tokenizer
    record_human.py     # Record human gameplay demos
    collect_data.py     # Build training dataset from HuggingFace
    train_classifier.py # Classifier training
    export_onnx.py      # ONNX export + quantization
    play_doom_visual.py # Play DOOM with the model
  tests/                # Unit tests
  models/               # Saved model checkpoints
  data/                 # Training datasets
  docs/                 # Documentation (this site)
```

---

## Code Style

- Python 3.10+ type hints
- Docstrings in Google style (for mkdocstrings compatibility)
- No external linter config -- keep code readable and consistent with existing style

---

## Running Tests

```bash
pytest tests/
```

Tests that require VizDoom are skipped if the `vizdoom` package is not installed. The `MockDoomEngine` in `doom/engine.py` enables testing the ASCII conversion and inference pipelines without VizDoom.

---

## Building Documentation

```bash
pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

This starts a local preview server at `http://127.0.0.1:8000/`.

To build the static site:

```bash
mkdocs build
```

---

## Areas for Contribution

- **Model improvements**: Experiment with different layer counts, hidden sizes, or attention configurations
- **Action space**: Add support for `use` (door opening) and `move_backward` actions
- **Data augmentation**: Implement ASCII frame augmentations (noise, partial occlusion)
- **Evaluation**: Expand the `DoomActionEvaluator` with more metrics (per-action precision/recall)
- **ONNX export**: Improve the export pipeline to include the ColBERT projection in the ONNX graph
- **Raspberry Pi testing**: Performance profiling and optimization on actual hardware
- **New scenarios**: Support for additional VizDoom scenarios beyond deathmatch

---

## Reporting Issues

Please open a GitHub issue with:

- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- System information (OS, Python version, hardware)
- Relevant log output
