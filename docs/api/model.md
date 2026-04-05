# Model API

API reference for the DOOM MultiVec model components.

---

## Classifier

The classifier wraps the ModernBERT-Hash encoder with depth embeddings, attention pooling, and a linear classification head for 4-action prediction.

::: doom_multivec.model.classifier

---

## ColBERT Wrapper (Legacy)

The ColBERT wrapper provides an interface for loading the model as a PyLate ColBERT model with MaxSim scoring. This is preserved for reference but is not used in the current classifier pipeline.

::: doom_multivec.model.colbert_wrapper

---

## Tokenizer

The character-level tokenizer maps each ASCII character to exactly one token. No BPE merging -- every character retains its spatial meaning (brightness level, entity type, row boundary).

::: doom_multivec.model.tokenizer

---

## Hash Embedding Model

The custom ModernBERT model with hash embeddings. This module defines `HashEmbedding` (the compact embedding layer) and `ModernBertHashModel` (the full model class that replaces standard embeddings with hash embeddings).

::: doom_multivec.model.modeling_doom_hash
