"""
DOOM KD Dataset Builder.

Builds a PyLate knowledge-distillation dataset from HuggingFace VizDoom datasets:
1. Streams JPEG frames from HuggingFace
2. Decodes and converts each frame to ASCII art
3. Computes teacher scores from the dataset's action labels
4. Groups frames into n_ways training samples (mix of high/low scores per action)
5. Saves everything in PyLate KD DatasetDict format

The PyLate KD format requires a DatasetDict with 3 splits:
  - train:     query_id (str), document_ids (list[str]), scores (list[float])
  - queries:   query_id (str), text (str)
  - documents: document_id (str), text (str)
"""

import io
import math

import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset, DatasetDict

from ..ascii.converter import AsciiConverter
from .action_mapping import BASE_ACTIONS, action_id_to_scores


# ---------------------------------------------------------------------------
# Action query texts (multi-word for richer embeddings)
# ---------------------------------------------------------------------------
ACTION_QUERY_TEXTS = {
    'shoot': '[ACT_SHOOT] shoot fire weapon attack enemy',
    'move_forward': '[ACT_MOVE_FWD] move forward advance walk ahead',
    'turn_left': '[ACT_TURN_LEFT] turn left look left rotate left',
    'turn_right': '[ACT_TURN_RIGHT] turn right look right rotate right',
    'strafe_left': '[ACT_STRAFE_LEFT] strafe left sidestep dodge left',
    'strafe_right': '[ACT_STRAFE_RIGHT] strafe right sidestep dodge right',
}


class DoomKDDatasetBuilder:
    """Builds a PyLate KD-format dataset from VizDoom gameplay recordings.

    Streams JPEG frames from a HuggingFace dataset, converts them to ASCII
    art, computes per-action teacher scores from the dataset's action labels,
    and packages everything into the ``DatasetDict`` format expected by
    PyLate's knowledge-distillation trainer.

    Attributes:
        hf_dataset_name: HuggingFace dataset identifier (e.g.
            ``'arnaudstiegler/vizdoom-50-episodes-skipframe-4'``).
        ascii_width: Width of the ASCII frame representation in characters.
        ascii_height: Height of the ASCII frame representation in characters.
        max_frames: Maximum number of frames to process from the dataset.
        n_ways: Number of documents per training sample.  Each sample
            contains a mix of high-scoring and low-scoring frames for a
            given action query.
        split: Dataset split to load (default ``'train'``).
        noise_std: Standard deviation of Gaussian noise added to teacher
            scores for training robustness.
        rng: NumPy random number generator seeded at construction.

    Example:
        >>> builder = DoomKDDatasetBuilder(
        ...     "arnaudstiegler/vizdoom-50-episodes-skipframe-4",
        ...     max_frames=1000,
        ... )
        >>> builder.build("/tmp/doom_kd_dataset")
    """

    def __init__(
        self,
        hf_dataset_name: str,
        ascii_width: int = 40,
        ascii_height: int = 25,
        max_frames: int = 50000,
        n_ways: int = 16,
        split: str = 'train',
        noise_std: float = 0.05,
        seed: int = 42,
        scan_limit: int = 0,
        stride: int = 0,
    ):
        """Initialise the dataset builder.

        Args:
            hf_dataset_name: HuggingFace dataset identifier.
            ascii_width: Width of the ASCII frame in characters.
                Defaults to 40.
            ascii_height: Height of the ASCII frame in characters.
                Defaults to 25.
            max_frames: Maximum number of frames to process.
                Defaults to 50000.
            n_ways: Number of documents per training sample.
                Defaults to 16.
            split: Dataset split to load. Defaults to ``'train'``.
            noise_std: Standard deviation of Gaussian noise added to teacher
                scores. Defaults to 0.05.
            seed: Random seed for reproducibility. Defaults to 42.
            scan_limit: Total number of frames to scan through in the stream.
                0 means auto-calculate from ``max_frames * stride``.
                Set to a large value to scan the full dataset.
            stride: Take every N-th frame to spread samples across
                episodes. 0 means auto-calculate as
                ``scan_limit // max_frames``. A stride of 1 takes every
                frame (no skipping).
        """
        self.hf_dataset_name = hf_dataset_name
        self.ascii_width = ascii_width
        self.ascii_height = ascii_height
        self.max_frames = max_frames
        self.n_ways = n_ways
        self.split = split
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.scan_limit = scan_limit
        self.stride = stride

    def build(self, output_path: str) -> None:
        """Build the complete PyLate KD dataset and save to disk.

        Executes the full pipeline: streaming frames, ASCII conversion,
        teacher-score computation, n-ways sampling, and serialisation.

        Args:
            output_path: Directory where the ``DatasetDict`` will be saved
                (created if it does not exist).
        """
        print(f"Building KD dataset from {self.hf_dataset_name}")
        print(f"  ASCII size: {self.ascii_width}x{self.ascii_height}")
        print(f"  Max frames: {self.max_frames}")
        print(f"  N-ways per sample: {self.n_ways}")
        print()

        # Phase 1: Process all frames
        documents, frame_scores = self._process_frames()

        print(f"\nPhase 1 complete: {len(documents)} frames processed")

        # Phase 2: Build KD training samples
        print("\nBuilding KD training samples...")
        train_samples = self._build_kd_samples(frame_scores)
        print(f"Phase 2 complete: {len(train_samples)} training samples created")

        # Phase 3: Save as DatasetDict
        print(f"\nSaving dataset to {output_path}...")
        self._save_dataset(documents, train_samples, output_path)
        print("Done!")

        # Print statistics
        self._print_statistics(documents, frame_scores, train_samples)

    def _process_frames(self) -> tuple[dict[str, str], list[tuple[str, dict[str, float]]]]:
        """Stream and process frames from the HuggingFace dataset.

        Uses stride-based sampling to spread frames across the entire
        dataset rather than taking only the first N frames.  This ensures
        diversity across episodes, map regions, and gameplay phases.

        Returns:
            A two-element tuple:

            * **documents** -- Mapping from document ID (e.g. ``'frame_000042'``)
              to the ASCII text representation of that frame.
            * **frame_scores** -- List of ``(doc_id, {action_name: score})``
              tuples containing teacher scores for every processed frame.
        """
        documents = {}
        frame_scores = []
        episodes_seen = set()

        converter = AsciiConverter(self.ascii_width, self.ascii_height)
        ds = load_dataset(self.hf_dataset_name, split=self.split, streaming=True)

        # Determine scan_limit and stride
        scan_limit = self.scan_limit
        stride = self.stride

        if scan_limit <= 0 and stride <= 0:
            # Default: scan 5x more than we need, stride accordingly
            scan_limit = self.max_frames * 5
            stride = 5
        elif scan_limit <= 0:
            scan_limit = self.max_frames * stride
        elif stride <= 0:
            stride = max(1, scan_limit // self.max_frames)

        print(f"Sampling strategy: scan {scan_limit:,} frames, take every {stride}-th")
        print("Processing frames...")

        collected = 0
        for i, sample in enumerate(ds):
            if i >= scan_limit:
                break
            if collected >= self.max_frames:
                break

            # Stride: only process every N-th frame
            if i % stride != 0:
                continue

            # Track episode diversity
            ep_id = sample.get('episode_id', None)
            if ep_id is not None:
                episodes_seen.add(ep_id)

            # Decode JPEG frame -- field name varies across datasets
            frame_bytes = self._extract_frame_bytes(sample)
            if frame_bytes is None:
                continue

            try:
                img = Image.open(io.BytesIO(frame_bytes)).convert('L')  # grayscale
            except Exception as e:
                print(f"  Warning: failed to decode frame {i}: {e}")
                continue

            frame_array = np.array(img)

            # Convert to ASCII
            ascii_text = converter.convert_simple(frame_array)

            doc_id = f"frame_{i:06d}"
            documents[doc_id] = ascii_text

            # Get teacher scores from the action label
            action_id = self._extract_action_id(sample)
            if action_id is None:
                continue

            # Add noise for robustness
            scores = action_id_to_scores(action_id)
            if self.noise_std > 0.0:
                noise = self.rng.normal(0.0, self.noise_std, size=len(BASE_ACTIONS))
                noisy_scores = {}
                for j, action_name in enumerate(BASE_ACTIONS):
                    noisy_scores[action_name] = float(
                        np.clip(scores[action_name] + noise[j], 0.0, 1.0)
                    )
                scores = noisy_scores

            frame_scores.append((doc_id, scores))
            collected += 1

            if collected % 5000 == 0:
                print(f"  Collected {collected} frames (scanned {i + 1:,}, "
                      f"{len(episodes_seen)} episodes)...")

        print(f"  Episodes covered: {len(episodes_seen)}")
        return documents, frame_scores

    @staticmethod
    def _extract_frame_bytes(sample: dict) -> bytes | None:
        """Extract JPEG frame bytes from a dataset sample.

        Handles different field names across HuggingFace VizDoom datasets
        (e.g. ``'frames'``, ``'compressed_image'``, ``'frame'``, ``'image'``).

        Args:
            sample: A single row from the HuggingFace dataset.

        Returns:
            Raw JPEG bytes, or ``None`` if no recognised image field is found.
        """
        for field in ('frames', 'compressed_image', 'frame', 'image'):
            if field in sample:
                val = sample[field]
                if isinstance(val, bytes):
                    return val
                # Some datasets wrap images as PIL or dicts
                if isinstance(val, dict) and 'bytes' in val:
                    return val['bytes']
        return None

    @staticmethod
    def _extract_action_id(sample: dict) -> int | None:
        """Extract the action ID from a dataset sample.

        Handles different field names (``'actions'`` and ``'action'``) and
        different value types (scalar int, list, or array).

        Args:
            sample: A single row from the HuggingFace dataset.

        Returns:
            Integer action ID, or ``None`` if no recognised action field is
            found.
        """
        for field in ('actions', 'action'):
            if field in sample:
                val = sample[field]
                if isinstance(val, (int, np.integer)):
                    return int(val)
                # Some datasets store action as a list or array
                if hasattr(val, '__len__') and len(val) > 0:
                    return int(val[0]) if isinstance(val[0], (int, np.integer)) else None
        return None

    def _build_kd_samples(
        self, frame_scores: list[tuple[str, dict[str, float]]]
    ) -> list[dict]:
        """Group frames per action query into n-ways training samples.

        For each base action the method:

        1. Sorts all frames by their score for that action (descending).
        2. Splits into a high-score tier (top half) and low-score tier.
        3. Shuffles within each tier.
        4. Creates training samples each containing ``n_ways`` frames --
           half drawn from the high tier and half from the low tier.
        5. Cycles through tiers if necessary to cover all frames.

        This ensures each training sample has a diverse mix of positive and
        negative examples, which is critical for effective KD training.

        Args:
            frame_scores: List of ``(doc_id, {action_name: score})`` tuples
                as returned by :meth:`_process_frames`.

        Returns:
            List of dicts, each with keys ``'query_id'`` (str),
            ``'document_ids'`` (list[str]), and ``'scores'`` (list[float]).
        """
        train_samples = []
        n_high = self.n_ways // 2
        n_low = self.n_ways - n_high  # handles odd n_ways

        for action_name in BASE_ACTIONS:
            # Sort frames by score for this action (descending)
            scored_frames = [
                (doc_id, scores[action_name])
                for doc_id, scores in frame_scores
            ]
            scored_frames.sort(key=lambda x: x[1], reverse=True)

            n_total = len(scored_frames)
            if n_total < self.n_ways:
                print(
                    f"  Warning: only {n_total} frames for action '{action_name}', "
                    f"need at least {self.n_ways}. Skipping."
                )
                continue

            # Split into high and low tiers at the median
            mid = n_total // 2
            high_tier = scored_frames[:mid]
            low_tier = scored_frames[mid:]

            # Shuffle within each tier
            high_tier_shuffled = list(high_tier)
            low_tier_shuffled = list(low_tier)
            self.rng.shuffle(high_tier_shuffled)
            self.rng.shuffle(low_tier_shuffled)

            # Determine number of samples to create:
            # enough to cycle through the larger tier at least once
            n_samples = max(
                math.ceil(len(high_tier_shuffled) / n_high),
                math.ceil(len(low_tier_shuffled) / n_low),
            )

            for sample_idx in range(n_samples):
                doc_ids = []
                sample_scores = []

                # Pick n_high from high tier (cycling)
                for k in range(n_high):
                    idx = (sample_idx * n_high + k) % len(high_tier_shuffled)
                    doc_id, score = high_tier_shuffled[idx]
                    doc_ids.append(doc_id)
                    sample_scores.append(score)

                # Pick n_low from low tier (cycling)
                for k in range(n_low):
                    idx = (sample_idx * n_low + k) % len(low_tier_shuffled)
                    doc_id, score = low_tier_shuffled[idx]
                    doc_ids.append(doc_id)
                    sample_scores.append(score)

                train_samples.append({
                    'query_id': action_name,
                    'document_ids': doc_ids,
                    'scores': sample_scores,
                })

        # Shuffle all training samples across actions
        self.rng.shuffle(train_samples)

        return train_samples

    def build_classifier(self, output_path: str) -> None:
        """Build a simple (frame_text, action_scores) dataset for classification.

        Each sample is one frame with a 6-dim soft score vector.
        No query/document distinction — just frames and labels.

        Args:
            output_path: Directory for the saved Dataset.
        """
        print(f"Building CLASSIFIER dataset from {self.hf_dataset_name}")
        print(f"  ASCII size: {self.ascii_width}x{self.ascii_height}")
        print(f"  Max frames: {self.max_frames}")
        print()

        documents, frame_scores = self._process_frames()
        print(f"\nPhase 1 complete: {len(documents)} frames processed")

        # Build flat dataset: text + scores vector
        texts = []
        scores_list = []
        action_names = list(BASE_ACTIONS)

        for doc_id, scores in frame_scores:
            texts.append(documents[doc_id])
            scores_list.append([scores[a] for a in action_names])

        # Oversample minority classes to balance primary-action distribution
        arr = np.array(scores_list)
        primary = arr.argmax(axis=1)
        counts = np.bincount(primary, minlength=len(action_names))
        # Use 25th percentile as target — moderate oversampling.
        # Median was too aggressive (shoot 1.9% → 15.2%, caused over-shooting).
        # 25th percentile brings shoot up to ~5-8% which is more natural.
        target_count = int(np.percentile(counts[counts > 0], 25))

        print(f"\n  Balancing classes (target ~{target_count:,} per primary action):")
        extra_texts = []
        extra_scores = []
        for action_idx, action_name in enumerate(action_names):
            current = counts[action_idx]
            if current >= target_count or current == 0:
                print(f"    {action_name:15s}: {current:>8,} — OK")
                continue
            # Oversample: repeat minority samples
            minority_indices = np.where(primary == action_idx)[0]
            repeats_needed = target_count - current
            oversample_indices = self.rng.choice(
                minority_indices, size=repeats_needed, replace=True
            )
            for idx in oversample_indices:
                extra_texts.append(texts[idx])
                extra_scores.append(scores_list[idx])
            print(f"    {action_name:15s}: {current:>8,} + {repeats_needed:,} oversampled → {target_count:,}")

        texts.extend(extra_texts)
        scores_list.extend(extra_scores)

        # Shuffle
        indices = list(range(len(texts)))
        self.rng.shuffle(indices)
        texts = [texts[i] for i in indices]
        scores_list = [scores_list[i] for i in indices]

        ds = Dataset.from_dict({
            'text': texts,
            'scores': scores_list,
        })
        ds.save_to_disk(output_path)

        # Stats
        arr = np.array(scores_list)
        primary = arr.argmax(axis=1)
        counts = np.bincount(primary, minlength=len(action_names))
        print(f"\n  Saved {len(ds):,} samples to {output_path}")
        print(f"  Final distribution (primary action):")
        for i, a in enumerate(action_names):
            print(f"    {a:15s}: {counts[i]:>8,} ({counts[i]/len(ds)*100:.1f}%)")

    def build_contrastive(self, output_path: str) -> None:
        """Build a contrastive (triplet) dataset for PyLate Contrastive loss.

        Format: each row has (query, positive_doc, negative_doc).
        The query is an action token, positive is a frame where that action
        was active, negative is a frame where it was NOT active.

        Args:
            output_path: Directory for the saved Dataset.
        """
        print(f"Building CONTRASTIVE dataset from {self.hf_dataset_name}")
        print(f"  ASCII size: {self.ascii_width}x{self.ascii_height}")
        print(f"  Max frames: {self.max_frames}")
        print()

        documents, frame_scores = self._process_frames()
        print(f"\nPhase 1 complete: {len(documents)} frames processed")

        # Group frames by action: positive (active) and negative (inactive)
        positive_frames = {a: [] for a in BASE_ACTIONS}
        negative_frames = {a: [] for a in BASE_ACTIONS}

        for doc_id, scores in frame_scores:
            for action in BASE_ACTIONS:
                if scores[action] > 0.5:
                    positive_frames[action].append(doc_id)
                else:
                    negative_frames[action].append(doc_id)

        print("\nPositive/Negative frame counts:")
        for action in BASE_ACTIONS:
            print(f"  {action:15s}: {len(positive_frames[action]):5d} pos, "
                  f"{len(negative_frames[action]):5d} neg")

        # Build triplets: (query_text, positive_frame, negative_frame)
        triplets = {'anchor': [], 'positive': [], 'negative': []}
        samples_per_action = self.max_frames // len(BASE_ACTIONS)

        for action in BASE_ACTIONS:
            pos = positive_frames[action]
            neg = negative_frames[action]
            if not pos or not neg:
                print(f"  Warning: skipping {action} (no pos or neg)")
                continue

            self.rng.shuffle(pos)
            self.rng.shuffle(neg)

            query_text = ACTION_QUERY_TEXTS[action]
            n = min(samples_per_action, len(pos), len(neg))

            for i in range(n):
                triplets['anchor'].append(query_text)
                triplets['positive'].append(documents[pos[i % len(pos)]])
                triplets['negative'].append(documents[neg[i % len(neg)]])

        # Shuffle
        indices = list(range(len(triplets['anchor'])))
        self.rng.shuffle(indices)
        triplets = {
            k: [v[i] for i in indices]
            for k, v in triplets.items()
        }

        ds = Dataset.from_dict(triplets)
        ds.save_to_disk(output_path)
        print(f"\n  Saved {len(ds)} contrastive triplets to {output_path}")
        print(f"  Columns: {ds.column_names}")

    def _save_dataset(
        self,
        documents: dict[str, str],
        train_samples: list[dict],
        output_path: str,
    ) -> None:
        """Serialise the processed data as a PyLate KD-format ``DatasetDict``.

        INVERTED setup: ASCII frames are QUERIES, actions are DOCUMENTS.
        This means MaxSim sums over ~1024 frame tokens (rich signal)
        instead of ~35 action tokens (dominated by shared chars).

        At inference this matches the real pipeline: the ASCII stream is
        encoded live (as query) and matched against pre-computed action
        embeddings (as documents).

        The train split maps each frame to 6 action documents with scores.

        Args:
            documents: Mapping from frame ID to ASCII text (these become queries).
            train_samples: List of training-sample dicts produced by
                :meth:`_build_kd_samples`.
            output_path: Directory to write the ``DatasetDict`` to.
        """
        # INVERTED: Queries = frames (ASCII text)
        queries_data = {
            'query_id': list(documents.keys()),
            'text': list(documents.values()),
        }

        # INVERTED: Documents = actions
        docs_data = {
            'document_id': list(ACTION_QUERY_TEXTS.keys()),
            'text': list(ACTION_QUERY_TEXTS.values()),
        }

        # Train split: each sample is one frame (query) with 6 action docs + scores
        # We need to restructure: instead of (action_query, [frames], [scores])
        # we need (frame_query, [actions], [scores])
        frame_to_scores = {}  # frame_id -> {action: score}
        for sample in train_samples:
            action = sample['query_id']
            for doc_id, score in zip(sample['document_ids'], sample['scores']):
                if doc_id not in frame_to_scores:
                    frame_to_scores[doc_id] = {}
                frame_to_scores[doc_id][action] = score

        # Build inverted train samples: each frame queries all 6 actions
        action_names = list(ACTION_QUERY_TEXTS.keys())
        train_data = {'query_id': [], 'document_ids': [], 'scores': []}

        for frame_id, action_scores in frame_to_scores.items():
            if len(action_scores) < len(action_names):
                continue  # skip frames that don't have all action scores
            train_data['query_id'].append(frame_id)
            train_data['document_ids'].append(action_names)
            train_data['scores'].append([action_scores[a] for a in action_names])

        dataset_dict = DatasetDict({
            'train': Dataset.from_dict(train_data),
            'queries': Dataset.from_dict(queries_data),
            'documents': Dataset.from_dict(docs_data),
        })
        dataset_dict.save_to_disk(output_path)
        print(f"  Saved INVERTED DatasetDict:")
        print(f"    {len(dataset_dict['train'])} train samples (1 per frame)")
        print(f"    {len(dataset_dict['queries'])} queries (= frames)")
        print(f"    {len(dataset_dict['documents'])} documents (= actions)")

    def _print_statistics(
        self,
        documents: dict[str, str],
        frame_scores: list[tuple[str, dict[str, float]]],
        train_samples: list[dict],
    ) -> None:
        """Print summary statistics about the built dataset.

        Args:
            documents: Mapping from document ID to ASCII text.
            frame_scores: List of ``(doc_id, scores)`` tuples.
            train_samples: List of training-sample dicts.
        """
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Total frames processed: {len(documents)}")
        print(f"Total training samples: {len(train_samples)}")
        print(f"Queries (base actions): {len(ACTION_QUERY_TEXTS)}")
        print()

        # Samples per action
        samples_per_action = {}
        for s in train_samples:
            action = s['query_id']
            samples_per_action[action] = samples_per_action.get(action, 0) + 1
        print("Samples per action:")
        for action in BASE_ACTIONS:
            count = samples_per_action.get(action, 0)
            print(f"  {action:20s}: {count}")

        # Score distribution per action
        print("\nScore distribution per action (across all frames):")
        for action in BASE_ACTIONS:
            all_scores = [scores[action] for _, scores in frame_scores]
            arr = np.array(all_scores)
            print(
                f"  {action:20s}: "
                f"mean={arr.mean():.3f}  "
                f"std={arr.std():.3f}  "
                f"min={arr.min():.3f}  "
                f"max={arr.max():.3f}  "
                f"high(>0.5)={np.sum(arr > 0.5)}"
            )

        # Document length stats
        doc_lengths = [len(text) for text in documents.values()]
        arr = np.array(doc_lengths)
        print(f"\nDocument lengths (chars): "
              f"mean={arr.mean():.0f}  min={arr.min()}  max={arr.max()}")
        print("=" * 60)
