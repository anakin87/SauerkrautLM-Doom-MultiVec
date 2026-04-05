"""
ModernBERT with Hash Embeddings for DOOM MultiVec.

Adapted from sauerkrautlm_reasonir/models/modernbert-hash-30k-2L/modeling_modernbert_hash.py.
This file must live alongside config.json so that AutoModel.from_pretrained()
with trust_remote_code=True can load the custom HashEmbedding layer.

The Hash Embedding replaces the standard vocab_size x hidden_size embedding table
with a much smaller vocab_size x projections lookup + projections x hidden_size
linear projection. For a 128-token ASCII vocabulary this saves almost nothing in
absolute terms, but it keeps the architecture consistent with the larger models
and allows seamless scaling.
"""

import torch
import torch.nn as nn
from transformers import ModernBertModel, ModernBertConfig


class HashEmbedding(nn.Module):
    """Hash-based token embeddings.

    Replaces a full ``vocab_size x hidden_size`` embedding table with a
    two-stage approach that is more parameter-efficient:

    1. Compact lookup: ``vocab_size x projections``
    2. Linear projection: ``projections -> hidden_size``
    3. LayerNorm normalisation

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Dimensionality of the output embeddings.
        projections: Dimensionality of the intermediate compact lookup.
        hash_embeddings: The compact ``nn.Embedding`` table.
        projection: Linear layer projecting from *projections* to
            *hidden_size*.
        norm: ``nn.LayerNorm`` applied after projection.
    """

    def __init__(self, vocab_size, hidden_size, projections=16, padding_idx=None):
        """Initialise the hash embedding layer.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            hidden_size: Output embedding dimensionality.
            projections: Size of the compact intermediate representation.
                Defaults to 16.
            padding_idx: Token ID that should be zero-padded.  Passed through
                to the underlying ``nn.Embedding``.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projections = projections

        # Compact lookup table: vocab_size x projections (instead of vocab_size x hidden_size)
        self.hash_embeddings = nn.Embedding(vocab_size, projections, padding_idx=padding_idx)

        # Projection: projections -> hidden_size
        self.projection = nn.Linear(projections, hidden_size, bias=False)

        # LayerNorm after projection
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids):
        """Compute hash embeddings for the given token IDs.

        Args:
            input_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Float tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        # 1. Compact lookup: [batch, seq_len] -> [batch, seq_len, projections]
        hash_emb = self.hash_embeddings(input_ids)

        # 2. Project up: [batch, seq_len, projections] -> [batch, seq_len, hidden_size]
        projected = self.projection(hash_emb)

        # 3. Normalize
        return self.norm(projected)

    @property
    def weight(self):
        """Compatibility with HuggingFace (for tie_word_embeddings etc.)."""
        return self.projection.weight

    @property
    def embedding_dim(self):
        """Return the output embedding dimensionality."""
        return self.hidden_size

    @property
    def num_embeddings(self):
        """Return the vocabulary size."""
        return self.vocab_size


class DepthEmbedding(nn.Module):
    """Quantized depth embedding that gets added to token embeddings.

    The VizDoom depth buffer is quantized into discrete bins (e.g. 16 levels)
    and each bin has a learned embedding vector. This embedding is ADDED to the
    token embedding at each position, giving the model depth/distance info
    without extra tokens.

    Attributes:
        num_bins: Number of depth quantization bins (default 16).
        depth_emb: The learned embedding table.
    """

    def __init__(self, num_bins, hidden_size):
        super().__init__()
        self.num_bins = num_bins
        # +1 for a "no depth" bin (for CLS, SEP, PAD, newline tokens)
        self.depth_emb = nn.Embedding(num_bins + 1, hidden_size, padding_idx=num_bins)
        self.no_depth_id = num_bins

    def forward(self, depth_ids):
        """Look up depth embeddings.

        Args:
            depth_ids: Integer tensor of shape (batch, seq_len) with values
                in [0, num_bins-1] for depth, or num_bins for "no depth".

        Returns:
            Float tensor of shape (batch, seq_len, hidden_size).
        """
        return self.depth_emb(depth_ids)


class ModernBertHashModel(ModernBertModel):
    """ModernBERT with Hash Embeddings and optional Depth Embeddings.

    Inherits from ``ModernBertModel`` and replaces only the token embedding
    layer with a :class:`HashEmbedding`. Optionally adds :class:`DepthEmbedding`
    that gets summed into the token embeddings at each position.

    Attributes:
        hash_projections: Number of projections used in the
            :class:`HashEmbedding` layer.
        depth_embedding: Optional :class:`DepthEmbedding` layer. Enabled when
            ``depth_bins`` is set in the config.
    """

    def __init__(self, config: ModernBertConfig):
        """Initialise ModernBERT and replace the embedding layer.

        Args:
            config: A ``ModernBertConfig`` instance. Supports:
                - ``hash_projections``: compact lookup size (default 16)
                - ``depth_bins``: number of depth quantization bins (default 0 = disabled)
        """
        projections = getattr(config, "hash_projections", 16)
        depth_bins = getattr(config, "depth_bins", 0)

        super().__init__(config)

        # Replace standard embeddings with hash embeddings
        self.hash_projections = projections
        hash_emb = HashEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            projections=projections,
            padding_idx=config.pad_token_id,
        )

        if hasattr(self, "embeddings") and hasattr(self.embeddings, "tok_embeddings"):
            self.embeddings.tok_embeddings = hash_emb

        # Optional depth embedding
        self.depth_bins = depth_bins
        if depth_bins > 0:
            self.depth_embedding = DepthEmbedding(depth_bins, config.hidden_size)
        else:
            self.depth_embedding = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                depth_ids=None, **kwargs):
        """Forward pass with optional depth embedding injection.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.
            token_type_ids: Ignored (BERT compat).
            depth_ids: Optional (batch, seq_len) depth bin IDs. If provided
                and depth_embedding is enabled, depth embeddings are added
                to the token embeddings before the transformer layers.
        """
        if depth_ids is not None and self.depth_embedding is not None:
            # Get token embeddings manually, add depth, then run transformer
            tok_emb = self.embeddings.tok_embeddings(input_ids)
            depth_emb = self.depth_embedding(depth_ids)
            # Add depth info to token embeddings
            combined = tok_emb + depth_emb

            # We need to bypass the normal embedding layer and inject directly.
            # ModernBERT's forward expects input_ids, so we use inputs_embeds.
            return super().forward(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

    def get_input_embeddings(self):
        """Return the hash embedding layer, falling back to the parent implementation."""
        if hasattr(self, "embeddings") and hasattr(self.embeddings, "tok_embeddings"):
            return self.embeddings.tok_embeddings
        return super().get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        """Resize the hash embedding table to match a new vocabulary size.

        PyLate calls this method to ensure the embedding table matches the
        tokenizer vocabulary size.  For :class:`HashEmbedding` layers, only
        the compact lookup table (``hash_embeddings``) is resized; the
        projection layer is left unchanged.

        Args:
            new_num_tokens: Target vocabulary size.  If ``None``, returns the
                current embedding layer unchanged.
            pad_to_multiple_of: Unused; kept for API compatibility.
            mean_resizing: Unused; kept for API compatibility.

        Returns:
            The (possibly resized) input embedding module.
        """
        if new_num_tokens is None:
            return self.get_input_embeddings()

        old_emb = self.get_input_embeddings()
        if not isinstance(old_emb, HashEmbedding):
            return super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

        old_num = old_emb.vocab_size
        if new_num_tokens == old_num:
            return old_emb

        # Resize the hash_embeddings (the compact lookup table)
        new_hash = nn.Embedding(
            new_num_tokens,
            old_emb.projections,
            padding_idx=old_emb.hash_embeddings.padding_idx,
        )

        # Copy old weights
        num_to_copy = min(old_num, new_num_tokens)
        new_hash.weight.data[:num_to_copy] = old_emb.hash_embeddings.weight.data[:num_to_copy]

        old_emb.hash_embeddings = new_hash
        old_emb.vocab_size = new_num_tokens

        # Update config
        self.config.vocab_size = new_num_tokens

        return old_emb
