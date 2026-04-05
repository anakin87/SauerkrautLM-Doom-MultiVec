"""
Collect training data from HuggingFace VizDoom datasets.

Converts frames to ASCII art and creates a PyLate KD-format dataset for
training the DOOM MultiVec ColBERT model.

Usage:
  python scripts/collect_data.py \\
      --dataset arnaudstiegler/vizdoom-50-episodes-skipframe-4 \\
      --max-frames 50000 \\
      --output data/doom-kd-50k

  python scripts/collect_data.py \\
      --dataset dokster/vizdoom-deathmatch-10k \\
      --max-frames 10000 \\
      --output data/doom-kd-10k
"""

import argparse
import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doom_multivec.training.dataset import DoomKDDatasetBuilder


def main():
    parser = argparse.ArgumentParser(
        description='Collect VizDoom data and build PyLate KD dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        default='arnaudstiegler/vizdoom-50-episodes-skipframe-4',
        help='HuggingFace dataset identifier',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=50000,
        help='Maximum number of frames to process',
    )
    parser.add_argument(
        '--n-ways',
        type=int,
        default=16,
        help='Number of documents per KD training sample',
    )
    parser.add_argument(
        '--ascii-width',
        type=int,
        default=40,
        help='ASCII frame width in characters',
    )
    parser.add_argument(
        '--ascii-height',
        type=int,
        default=25,
        help='ASCII frame height in characters',
    )
    parser.add_argument(
        '--noise-std',
        type=float,
        default=0.05,
        help='Gaussian noise std for teacher score robustness (0 = no noise)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split to load',
    )
    parser.add_argument(
        '--scan-limit',
        type=int,
        default=0,
        help='Total frames to scan in stream (0 = auto: max_frames * stride)',
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=0,
        help='Take every N-th frame for episode diversity (0 = auto from scan_limit)',
    )
    parser.add_argument(
        '--output',
        default='data/doom-kd-50k',
        help='Output directory for the saved DatasetDict',
    )
    parser.add_argument(
        '--mode',
        default='kd',
        choices=['kd', 'contrastive', 'classifier'],
        help='Dataset format: kd (Distillation), contrastive (triplets), classifier (frame+scores)',
    )
    args = parser.parse_args()

    builder = DoomKDDatasetBuilder(
        hf_dataset_name=args.dataset,
        ascii_width=args.ascii_width,
        ascii_height=args.ascii_height,
        max_frames=args.max_frames,
        n_ways=args.n_ways,
        split=args.split,
        noise_std=args.noise_std,
        seed=args.seed,
        scan_limit=args.scan_limit,
        stride=args.stride,
    )
    if args.mode == 'contrastive':
        builder.build_contrastive(args.output)
    elif args.mode == 'classifier':
        builder.build_classifier(args.output)
    else:
        builder.build(args.output)


if __name__ == '__main__':
    main()
