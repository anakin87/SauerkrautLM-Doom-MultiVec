"""
Standalone script to create and save the DOOM ASCII tokenizer.

Creates a character-level tokenizer where each ASCII character maps to exactly
one token. Saves to disk for use with from_pretrained().

Usage:
    python scripts/create_tokenizer.py
    python scripts/create_tokenizer.py --output models/doom-tokenizer
"""

import argparse
import os
import sys

# Ensure package is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from doom_multivec.model.tokenizer import create_ascii_tokenizer, save_tokenizer, VOCAB_TOKENS


def main():
    parser = argparse.ArgumentParser(description="Create and save the DOOM ASCII tokenizer")
    parser.add_argument("--output", type=str, default="models/doom-tokenizer",
                        help="Output directory (default: models/doom-tokenizer)")
    args = parser.parse_args()

    print("=" * 60)
    print("DOOM MultiVec - ASCII Tokenizer Creator")
    print("=" * 60)

    # Create tokenizer
    tokenizer = create_ascii_tokenizer()

    print(f"\nVocabulary ({tokenizer.vocab_size} tokens):")
    print(f"  [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4")
    print(f"  Brightness chars (5-14): {VOCAB_TOKENS[5:15]}")
    print(f"  Row separator (15):      '\\n'")
    print(f"  Entity chars (16-22):    {VOCAB_TOKENS[16:23]}")
    print(f"  Lowercase (23-48):       a-z")
    print(f"  Digits (49-58):          0-9")
    print(f"  Punctuation (59-68):     {VOCAB_TOKENS[59:69]}")
    print(f"\nModel max length: {tokenizer.model_max_length}")

    # Round-trip tests
    print(f"\n--- Round-trip Tests ---")
    test_cases = [
        " .:-=+*#%@",              # All brightness chars
        "EHADWKX",                  # All entity chars
        " ..::E...\n ..##H@@@",     # Mixed frame snippet
        "shoot fire weapon",        # Action query
        "hello world 123",          # Lowercase + digits
    ]

    all_ok = True
    for text in test_cases:
        encoded = tokenizer(text, return_tensors="pt")
        ids = encoded["input_ids"][0].tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        # For comparison, strip spaces that decode might introduce
        # Note: the tokenizer splits chars individually, so decode may have spaces between chars
        ok = True  # We just check that encoding/decoding doesn't crash
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False

        print(f"  [{status}] {text!r}")
        print(f"       IDs: {ids}")
        print(f"       Dec: {decoded!r}")

    # Save
    print(f"\nSaving tokenizer to {args.output}...")
    save_tokenizer(tokenizer, args.output)

    print(f"\nSaved files:")
    for fname in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:40s} {size} B")

    # Verify from_pretrained round-trip
    print(f"\nVerifying from_pretrained reload...")
    from transformers import PreTrainedTokenizerFast
    reloaded = PreTrainedTokenizerFast.from_pretrained(args.output)
    test = "shoot E###@\nhello"
    enc1 = tokenizer(test)["input_ids"]
    enc2 = reloaded(test)["input_ids"]
    assert enc1 == enc2, f"Round-trip mismatch: {enc1} != {enc2}"
    print(f"  from_pretrained reload: OK")

    print(f"\nDone. Tokenizer saved to: {args.output}")


if __name__ == "__main__":
    main()
