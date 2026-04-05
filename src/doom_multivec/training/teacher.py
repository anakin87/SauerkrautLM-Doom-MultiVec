"""
Teacher scoring for knowledge distillation.

Converts dataset action labels into per-base-action teacher scores used to
train the ColBERT model via PyLate's knowledge distillation pipeline.

The PPO agent in the original GameNGen dataset chose a single composite action
per frame. We decompose that into scores for each of our 6 base actions,
optionally adding Gaussian noise for training robustness.
"""

import numpy as np

from .action_mapping import action_id_to_scores, BASE_ACTIONS


def compute_teacher_scores(
    action_id: int,
    noise_std: float = 0.05,
    active_score: float = 0.85,
    inactive_score: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute teacher scores for a frame based on the PPO agent's chosen action.

    Decomposes the GameNGen composite action into per-base-action scores and
    optionally adds Gaussian noise for training robustness.  Scores are
    clipped to ``[0, 1]`` after noise is applied.

    Args:
        action_id: GameNGen discrete action ID (0--17).
        noise_std: Standard deviation of Gaussian noise added to each score.
            Set to ``0.0`` for deterministic output.  Defaults to 0.05.
        active_score: Base score for active action components.
            Defaults to 0.85.
        inactive_score: Base score for inactive action components.
            Defaults to 0.05.
        rng: NumPy random number generator for reproducibility.  If
            ``None``, ``numpy.random.default_rng()`` is used.

    Returns:
        Dictionary mapping each of the 6 base action names to a float score
        in ``[0, 1]``.

    Example:
        >>> import numpy as np
        >>> scores = compute_teacher_scores(8, noise_std=0.0)
        >>> scores["move_forward"]
        0.85
    """
    base_scores = action_id_to_scores(
        action_id,
        active_score=active_score,
        inactive_score=inactive_score,
    )

    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0.0, noise_std, size=len(BASE_ACTIONS))
        scores = {}
        for i, action_name in enumerate(BASE_ACTIONS):
            noisy_score = base_scores[action_name] + noise[i]
            scores[action_name] = float(np.clip(noisy_score, 0.0, 1.0))
    else:
        scores = base_scores

    return scores
