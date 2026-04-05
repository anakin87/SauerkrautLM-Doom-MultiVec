"""
Multi-vector encoder with classification head for DOOM action prediction.

Uses the ModernBERT-Hash encoder to produce token-level embeddings,
then pools and classifies into 6 DOOM actions. This avoids the MaxSim
collapse problem while keeping the rich token-level representations.

Architecture:
    ASCII Frame → ModernBERT-Hash → token embeddings (seq_len, 128)
    → attention-weighted pool → (128,)
    → Linear(128, 6) → action logits
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DoomMultiVecClassifier(nn.Module):
    """Multi-vector encoder + classification head for action prediction.

    Args:
        encoder_path: Path to the pretrained ModernBERT-Hash model.
        num_actions: Number of action classes. Defaults to 6.
        pool_mode: Pooling strategy ('attention', 'mean', 'cls').
            'attention' learns a weighted combination of token embeddings.
            'mean' averages all token embeddings.
            'cls' uses only the [CLS] token embedding.
    """

    ACTION_NAMES = [
        'shoot', 'move_forward', 'turn_left',
        'turn_right', 'strafe_left', 'strafe_right',
    ]

    def __init__(self, encoder_path, num_actions=6, pool_mode='attention',
                 use_flash_attn=False, ascii_rows=25, ascii_cols=40,
                 protos_per_action=4):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        # Load encoder on CPU first; flash attention is enabled after
        # moving to CUDA via enable_flash_attn().
        self.encoder = AutoModel.from_pretrained(
            encoder_path, trust_remote_code=True,
        )
        hidden_size = self.encoder.config.hidden_size  # 128

        self.pool_mode = pool_mode
        self.ascii_rows = ascii_rows
        self.ascii_cols = ascii_cols

        if pool_mode == 'attention':
            self.attn_weight = nn.Linear(hidden_size, 1, bias=False)
            self.classifier = nn.Linear(hidden_size, num_actions)
        elif pool_mode == 'spatial':
            self.row_proj = nn.Linear(hidden_size, hidden_size // 2)
            self.classifier = nn.Linear(ascii_rows * (hidden_size // 2), num_actions)
        elif pool_mode == 'token_vote':
            # Per-Token Voting: each token independently predicts all actions.
            # No pooling — every token keeps its own vote.
            # LogSumExp aggregation combines votes differentiably.
            # Gradient flows to ALL tokens proportionally (not winner-take-all).
            self.token_classifier = nn.Linear(hidden_size, num_actions)
            self.lse_temperature = nn.Parameter(torch.tensor(1.0))  # learnable temperature
            self.classifier = None
        elif pool_mode == 'maxsim':
            self.protos_per_action = protos_per_action
            self.prototypes = nn.Parameter(
                torch.randn(num_actions, self.protos_per_action, hidden_size) * 0.02
            )
            self.classifier = None
        elif pool_mode == 'multi_proto_attn':
            # Multi-Prototype Attention: each action has K learnable prototype
            # vectors. Each prototype computes soft attention over frame tokens
            # to extract an action-specific representation. Then an MLP scores
            # each action from its K attended representations.
            #
            # This is Late Interaction: each prototype interacts independently
            # with ALL frame token embeddings via attention, preserving the
            # multi-vector structure instead of collapsing to a single vector.
            self.protos_per_action = protos_per_action
            # Learnable prototypes: (num_actions, K, hidden)
            self.prototypes = nn.Parameter(
                torch.randn(num_actions, protos_per_action, hidden_size) * 0.02
            )
            # Per-action MLP: takes K attended vectors → 1 score
            # Input: K * hidden, Output: 1
            self.action_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(protos_per_action * hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(num_actions)
            ])
            self.classifier = None
        else:
            self.classifier = nn.Linear(hidden_size, num_actions)

        self.num_actions = num_actions

    def enable_flash_attn(self):
        """Enable Flash Attention 2 after model is on CUDA.

        Must be called after `.to('cuda')`. Converts encoder to bf16
        and sets the attention implementation config flag. The actual
        flash attention dispatch happens automatically when running
        under `torch.autocast('cuda', dtype=torch.bfloat16)`.
        """
        self.encoder = self.encoder.to(dtype=torch.bfloat16)
        self.encoder.config._attn_implementation = "flash_attention_2"
        self.encoder.config._attn_implementation_internal = "flash_attention_2"

    def forward(self, input_ids, attention_mask=None, labels=None, depth_ids=None):
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) mask.
            labels: (batch, num_actions) soft targets or (batch,) hard targets.
            depth_ids: Optional (batch, seq_len) depth bin IDs.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            depth_ids=depth_ids,
        )
        token_embs = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if self.pool_mode == 'token_vote':
            # Per-Token Voting: each token independently classifies
            per_token_logits = self.token_classifier(token_embs)  # (B, S, num_actions)

            # Mask padding tokens
            if attention_mask is not None:
                per_token_logits = per_token_logits.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )

            # LogSumExp aggregation: smooth differentiable alternative to max
            # r controls sharpness: r=1 ≈ mean, r→∞ ≈ max
            # Each token's vote contributes proportionally via softmax-like weighting
            r = torch.clamp(self.lse_temperature, min=0.1, max=20.0)
            logits = (1.0 / r) * torch.logsumexp(r * per_token_logits, dim=1)  # (B, num_actions)

        elif self.pool_mode == 'maxsim':
            B, S, H = token_embs.shape
            A = self.num_actions
            P = self.protos_per_action

            tokens_norm = nn.functional.normalize(token_embs, dim=-1)
            protos_flat = self.prototypes.reshape(A * P, H).to(dtype=tokens_norm.dtype)
            protos_norm = nn.functional.normalize(protos_flat, dim=-1)
            sim = torch.matmul(tokens_norm, protos_norm.T)
            if attention_mask is not None:
                sim = sim.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
            sim = sim.reshape(B, S, A, P)
            max_per_proto = sim.max(dim=1).values
            logits = max_per_proto.sum(dim=-1) * 5.0

        elif self.pool_mode == 'multi_proto_attn':
            # Multi-Prototype Attention (Late Interaction)
            # Each prototype attends softly over frame tokens to extract
            # an action-specific representation. No max() — full gradient flow.
            B, S, H = token_embs.shape
            A = self.num_actions
            P = self.protos_per_action

            # Cast prototypes to match encoder dtype (bf16 with flash attn)
            protos = self.prototypes.to(dtype=token_embs.dtype)  # (A, P, H)

            # Compute attention: each prototype attends over frame tokens
            # protos reshaped: (A*P, H), token_embs: (B, S, H)
            # attention scores: (B, A*P, S)
            protos_flat = protos.reshape(A * P, H)  # (A*P, H)
            attn_scores = torch.matmul(protos_flat, token_embs.transpose(1, 2))  # (A*P, S) broadcast → (B, A*P, S)
            # Need to expand for batch: (B, A*P, S)
            attn_scores = torch.einsum('ph,bsh->bps', protos_flat, token_embs)

            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    attention_mask.unsqueeze(1) == 0, float('-inf')
                )

            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, A*P, S)

            # Weighted sum of token embeddings per prototype
            # (B, A*P, S) @ (B, S, H) → (B, A*P, H)
            attended = torch.bmm(attn_weights, token_embs)

            # Reshape to (B, A, P*H)
            attended = attended.reshape(B, A, P * H)

            # Per-action MLP: (B, A, P*H) → (B, A, 1) → (B, A)
            logits_list = []
            for a in range(A):
                score = self.action_mlps[a](attended[:, a, :])  # (B, 1)
                logits_list.append(score)
            logits = torch.cat(logits_list, dim=-1)  # (B, A)

        else:
            pooled = self._pool(token_embs, attention_mask)
            logits = self.classifier(pooled)  # (batch, num_actions)

        result = {'logits': logits}

        if labels is not None:
            if labels.dim() == 1:
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                # Weighted soft-label loss: combines KL-div with per-action
                # weighting to prevent minority actions (shoot) from being ignored.
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                targets = nn.functional.softmax(labels, dim=-1)

                if hasattr(self, 'class_weights') and self.class_weights is not None:
                    # Weight the KL-div per action
                    w = self.class_weights.to(logits.device)
                    # Per-sample, per-action weighted KL
                    kl_per = targets * (targets.log() - log_probs)  # (batch, actions)
                    loss = (kl_per * w.unsqueeze(0)).sum(dim=-1).mean()
                else:
                    loss = nn.functional.kl_div(log_probs, targets, reduction='batchmean')
            result['loss'] = loss

        return result

    def _pool(self, token_embs, attention_mask):
        """Pool token embeddings into a single vector."""
        if self.pool_mode == 'cls':
            return token_embs[:, 0, :]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        else:
            mask = torch.ones_like(token_embs[:, :, :1])

        if self.pool_mode == 'mean':
            return (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if self.pool_mode == 'attention':
            scores = self.attn_weight(token_embs).squeeze(-1)  # (batch, seq)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
            return (token_embs * weights).sum(dim=1)

        if self.pool_mode == 'spatial':
            # Spatial pooling: mean-pool per ASCII row to preserve vertical structure.
            # Token sequence: [CLS] row0_chars \n row1_chars \n ... [SEP] [PAD...]
            # We skip CLS (idx 0), then take chunks of (ascii_cols + 1) for each row
            # (+1 for the \n separator token).
            batch_size = token_embs.shape[0]
            hidden = token_embs.shape[2]
            stride = self.ascii_cols + 1  # 40 chars + 1 newline per row

            row_vectors = []
            for r in range(self.ascii_rows):
                start = 1 + r * stride  # skip CLS
                end = start + self.ascii_cols
                if end > token_embs.shape[1]:
                    # Pad with zeros if sequence is shorter
                    row_vec = torch.zeros(batch_size, hidden, device=token_embs.device)
                else:
                    row_vec = token_embs[:, start:end, :].mean(dim=1)  # (batch, hidden)
                row_vectors.append(torch.relu(self.row_proj(row_vec)))

            # Stack: (batch, n_rows, hidden//2) → flatten → (batch, n_rows * hidden//2)
            spatial = torch.stack(row_vectors, dim=1)  # (batch, 25, 64)
            return spatial.reshape(batch_size, -1)  # (batch, 1600)

        raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

    def predict(self, input_ids, attention_mask=None):
        """Predict action probabilities.

        Returns:
            Dict mapping action names to probabilities.
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            probs = torch.softmax(result['logits'], dim=-1)

        # Return per-sample predictions
        predictions = []
        for i in range(probs.shape[0]):
            predictions.append({
                name: probs[i, j].item()
                for j, name in enumerate(self.ACTION_NAMES)
            })
        return predictions
