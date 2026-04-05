"""
Custom character-level tokenizer for DOOM ASCII art.

Each ASCII character maps to exactly one token. No BPE merging -- every character
retains its spatial meaning (brightness level, entity type, row boundary).

Vocabulary (~100 tokens):
  0: [PAD], 1: [UNK], 2: [CLS], 3: [SEP], 4: [MASK]
  5-14:  brightness chars " .:-=+*#%@"
  15:    '\n' (row separator)
  16-22: entity chars "EHADWKX"
  23-48: lowercase a-z (for action query text)
  49-58: digits 0-9
  59-68: common punctuation/symbols

Uses HuggingFace `tokenizers` library with WordLevel model and a character-level
pre-tokenizer, wrapped in PreTrainedTokenizerFast.
"""

import json
import os
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


# ---------------------------------------------------------------------------
# Vocabulary definition
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Brightness chars (10), dark to bright -- index 5-14
BRIGHTNESS_CHARS = list(" .:-=+*#%@")

# Row separator -- index 15
ROW_SEPARATOR = ["\n"]

# Entity chars -- index 16-22
ENTITY_CHARS = list("EHADWKX")

# Lowercase a-z for action query text -- index 23-48
LOWERCASE = [chr(c) for c in range(ord("a"), ord("z") + 1)]

# Digits 0-9 -- index 49-58
DIGITS = [chr(c) for c in range(ord("0"), ord("9") + 1)]

# Common punctuation/symbols -- index 59-68
# Note: ':', '-', '+' are already in BRIGHTNESS_CHARS so they are excluded here.
PUNCTUATION = list("!?,;()'\"/_")

# Action-specific tokens -- index 69-74
# These give each action a unique, unambiguous signal in the embedding space.
# Without them, char-level queries like "shoot" vs "strafe" are too similar.
ACTION_TOKENS = [
    "[ACT_SHOOT]",
    "[ACT_MOVE_FWD]",
    "[ACT_TURN_LEFT]",
    "[ACT_TURN_RIGHT]",
    "[ACT_STRAFE_LEFT]",
    "[ACT_STRAFE_RIGHT]",
]

# Assemble full vocab in order
VOCAB_TOKENS = (
    SPECIAL_TOKENS      # 0-4
    + BRIGHTNESS_CHARS  # 5-14
    + ROW_SEPARATOR     # 15
    + ENTITY_CHARS      # 16-22
    + LOWERCASE         # 23-48
    + DIGITS            # 49-58
    + PUNCTUATION       # 59-68
    + ACTION_TOKENS     # 69-74
)


def _build_vocab() -> dict[str, int]:
    """Build the token-to-id mapping from :data:`VOCAB_TOKENS`.

    Returns:
        Dictionary mapping each vocabulary token string to its integer ID.
    """
    return {tok: idx for idx, tok in enumerate(VOCAB_TOKENS)}


# ---------------------------------------------------------------------------
# Tokenizer creation
# ---------------------------------------------------------------------------

def create_ascii_tokenizer() -> PreTrainedTokenizerFast:
    """Build and return a character-level tokenizer for DOOM ASCII frames.

    Every input character is split individually (no BPE merges), so each
    ASCII art character retains its spatial meaning (brightness level, entity
    type, row boundary).  The tokenizer is wrapped as a HuggingFace
    ``PreTrainedTokenizerFast`` and supports the standard interface.

    Returns:
        A ``PreTrainedTokenizerFast`` instance with a vocabulary of ~69
        tokens, ``model_max_length=1200``, and ``[CLS]``/``[SEP]`` post-
        processing.

    Example:
        >>> tok = create_ascii_tokenizer()
        >>> encoded = tok("  .:-=+*#%@\\nEHADWKX", return_tensors="pt")
        >>> tok.decode(encoded["input_ids"][0], skip_special_tokens=True)
        '  .:-=+*#%@\\nEHADWKX'
    """
    vocab = _build_vocab()

    # Core tokenizer using WordLevel (one token per character)
    tokenizer_core = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Pre-tokenizer: split every character individually.
    # We use Split with pattern="" (empty regex) in "isolated" mode to emit
    # each character as a separate token. However, the cleanest way for
    # character-level is to use the built-in `CharDelimiterSplit` trick or
    # just a custom `Split` with regex that matches each character.
    # The simplest: use pre_tokenizers.Split with a regex that matches every
    # single character and outputs each one individually.
    tokenizer_core.pre_tokenizer = pre_tokenizers.Split(
        pattern="",
        behavior="isolated",
    )

    # Decoder: concatenate tokens without spaces (char-level)
    tokenizer_core.decoder = decoders.Fuse()

    # Post-processor: add [CLS] at start and [SEP] at end
    tokenizer_core.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", vocab["[CLS]"]),
            ("[SEP]", vocab["[SEP]"]),
        ],
    )

    # Wrap into PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        additional_special_tokens=ACTION_TOKENS,
        model_max_length=1200,  # 1024 frame chars + special tokens + padding
    )

    return fast_tokenizer


def save_tokenizer(tokenizer: PreTrainedTokenizerFast, path: str | Path) -> None:
    """Save the tokenizer to disk for later loading via ``from_pretrained()``.

    In addition to the standard HuggingFace tokenizer files, a human-readable
    ``ascii_vocab.json`` is written for debugging.

    Args:
        tokenizer: The ``PreTrainedTokenizerFast`` to persist.
        path: Directory where the tokenizer files will be written.  Created
            (with parents) if it does not exist.

    Example:
        >>> tok = create_ascii_tokenizer()
        >>> save_tokenizer(tok, "/tmp/doom_tokenizer")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(path))

    # Also save a human-readable vocab mapping for debugging
    vocab = tokenizer.get_vocab()
    # Sort by id for readability
    sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
    with open(path / "ascii_vocab.json", "w") as f:
        json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok = create_ascii_tokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Model max length: {tok.model_max_length}")

    # Quick round-trip test
    test_frame = " .:-=+*#%@\nEHADWKX\nhello world"
    encoded = tok(test_frame, return_tensors="pt")
    decoded = tok.decode(encoded["input_ids"][0], skip_special_tokens=True)
    print(f"\nTest input:   {test_frame!r}")
    print(f"Token IDs:    {encoded['input_ids'][0].tolist()}")
    print(f"Decoded:      {decoded!r}")

    # Verify special token IDs
    print(f"\nSpecial tokens:")
    print(f"  [PAD] = {tok.pad_token_id}")
    print(f"  [UNK] = {tok.unk_token_id}")
    print(f"  [CLS] = {tok.cls_token_id}")
    print(f"  [SEP] = {tok.sep_token_id}")
    print(f"  [MASK] = {tok.mask_token_id}")
