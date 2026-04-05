"""Export the trained model to ONNX format for lightweight inference.

Designed for deployment on resource-constrained devices (e.g. Raspberry Pi).

Usage::

    # Basic export:
    python scripts/export_onnx.py \\
        --model models/doom-multivec-trained \\
        --output models/doom-encoder-onnx/model.onnx

    # Export with INT8 dynamic quantization:
    python scripts/export_onnx.py \\
        --model models/doom-multivec-trained \\
        --output models/doom-encoder-onnx/model.onnx \\
        --quantize

    # Custom opset version:
    python scripts/export_onnx.py \\
        --model models/doom-multivec-trained \\
        --output models/doom-encoder-onnx/model.onnx \\
        --opset 17
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class ColBERTEncoderWrapper:
    """Wraps the ColBERT encoder for clean ONNX export.

    Extracts just the transformer backbone and produces L2-normalised
    hidden states from ``(input_ids, attention_mask)`` inputs.

    Args:
        model: A PyLate ``ColBERT`` model instance.
    """

    def __init__(self, model):
        import torch
        import torch.nn as nn

        self.model = model

        # Extract the underlying transformer encoder
        # PyLate ColBERT stores the base model inside model[0].auto_model
        if hasattr(model, '_first_module'):
            first_module = model._first_module()
        else:
            # Fallback: try direct attribute access
            first_module = list(model.children())[0]

        if hasattr(first_module, 'auto_model'):
            self.encoder = first_module.auto_model
        elif hasattr(first_module, 'model'):
            self.encoder = first_module.model
        else:
            raise AttributeError(
                "Could not locate the transformer encoder inside the ColBERT model. "
                "Expected 'auto_model' or 'model' attribute on the first module."
            )

        # Build a simple wrapper as an nn.Module for export
        class _ExportableEncoder(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, input_ids, attention_mask):
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Get last hidden state
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs[0]

                # L2 normalise along the embedding dimension
                norms = torch.norm(hidden, p=2, dim=-1, keepdim=True)
                norms = torch.clamp(norms, min=1e-12)
                return hidden / norms

        self.exportable = _ExportableEncoder(self.encoder)
        self.exportable.eval()


def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes.

    Args:
        path: Path to a file.

    Returns:
        File size in MB, rounded to two decimal places.
    """
    return round(os.path.getsize(path) / (1024 * 1024), 2)


def export_onnx(model_path: str, output_path: str, opset: int = 14) -> str:
    """Export the ColBERT model to ONNX format.

    Args:
        model_path: Path to the trained model directory.
        output_path: Destination path for the ``.onnx`` file.
        opset: ONNX opset version.  Defaults to 14.

    Returns:
        The resolved output file path.
    """
    import torch
    from pylate import models

    print(f"Loading model from {model_path}...")
    model = models.ColBERT(
        model_name_or_path=model_path,
        trust_remote_code=True,
        query_length=48,
        document_length=1100,
    )

    print("Wrapping encoder for export...")
    wrapper = ColBERTEncoderWrapper(model)
    exportable = wrapper.exportable

    # Create dummy inputs
    batch_size = 1
    seq_len = 128
    dummy_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting to ONNX (opset {opset})...")
    torch.onnx.export(
        exportable,
        (dummy_ids, dummy_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['embeddings'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'embeddings': {0: 'batch_size', 1: 'seq_len'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    size_mb = get_file_size_mb(output_path)
    print(f"Exported ONNX model: {output_path} ({size_mb} MB)")
    return output_path


def quantize_onnx(input_path: str, output_path: str | None = None) -> str:
    """Apply INT8 dynamic quantization to an ONNX model.

    Args:
        input_path: Path to the original ``.onnx`` file.
        output_path: Destination for the quantized model.  Defaults to
            ``<input_stem>_quantized.onnx`` in the same directory.

    Returns:
        The resolved output file path.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if output_path is None:
        stem = Path(input_path).stem
        parent = Path(input_path).parent
        output_path = str(parent / f"{stem}_quantized.onnx")

    print(f"Quantizing to INT8: {output_path}...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    orig_mb = get_file_size_mb(input_path)
    quant_mb = get_file_size_mb(output_path)
    ratio = quant_mb / orig_mb * 100 if orig_mb > 0 else 0
    print(f"  Original:  {orig_mb} MB")
    print(f"  Quantized: {quant_mb} MB ({ratio:.0f}% of original)")
    return output_path


def copy_tokenizer(model_path: str, output_dir: str) -> None:
    """Copy tokenizer files to the ONNX output directory.

    This ensures that the ONNX model directory is self-contained and can
    be loaded with ``AutoTokenizer.from_pretrained(output_dir)``.

    Args:
        model_path: Source model directory containing tokenizer files.
        output_dir: Destination directory for the tokenizer files.
    """
    tokenizer_files = [
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.txt',
        'vocab.json',
        'merges.txt',
        'added_tokens.json',
        'ascii_vocab.json',
    ]

    os.makedirs(output_dir, exist_ok=True)
    copied = []
    for fname in tokenizer_files:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            copied.append(fname)

    if copied:
        print(f"Copied tokenizer files to {output_dir}: {', '.join(copied)}")
    else:
        print(
            f"Warning: no tokenizer files found in {model_path}. "
            "ONNX inference may require a separate tokenizer path."
        )


def main():
    """Parse arguments and run the ONNX export pipeline."""
    parser = argparse.ArgumentParser(
        description='Export DOOM MultiVec model to ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--model', required=True,
        help='Path to trained model directory',
    )
    parser.add_argument(
        '--output', required=True,
        help='Output path for the ONNX model file (e.g. models/doom-encoder.onnx)',
    )
    parser.add_argument(
        '--opset', type=int, default=14,
        help='ONNX opset version (default: 14)',
    )
    parser.add_argument(
        '--quantize', action='store_true',
        help='Apply INT8 dynamic quantization after export',
    )
    parser.add_argument(
        '--quantize-output', default=None,
        help='Output path for quantized model (default: auto-generated)',
    )
    args = parser.parse_args()

    # 1. Export to ONNX
    onnx_path = export_onnx(args.model, args.output, opset=args.opset)

    # 2. Optionally quantize
    if args.quantize:
        quantize_onnx(onnx_path, args.quantize_output)

    # 3. Copy tokenizer files alongside the ONNX model
    output_dir = os.path.dirname(args.output) or '.'
    copy_tokenizer(args.model, output_dir)

    print("\nExport complete.")


if __name__ == '__main__':
    main()
