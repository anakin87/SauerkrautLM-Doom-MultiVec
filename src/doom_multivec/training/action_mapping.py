"""
Maps the 18 GameNGen composite actions to our 6 base actions with teacher scores.

The GameNGen/arnaudstiegler/dokster VizDoom datasets use a discrete action space
of 18 composite actions built from button vector:
  [ATTACK, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT, TURN_RIGHT, TURN_LEFT]

We decompose each composite action into its constituent base actions and assign
teacher scores: high for active components, low for inactive ones.
"""

# ---------------------------------------------------------------------------
# The exact 18-action mapping from GameNGen/arnaudstiegler datasets
# ---------------------------------------------------------------------------
# Each entry maps a discrete action ID to the set of base actions that are
# simultaneously active when that composite action is taken.

GAMANGEN_ACTIONS = {
    0:  {'turn_left': True},
    1:  {'turn_right': True},
    2:  {'strafe_right': True},
    3:  {'strafe_right': True, 'turn_left': True},
    4:  {'strafe_right': True, 'turn_right': True},
    5:  {'strafe_left': True},
    6:  {'strafe_left': True, 'turn_left': True},
    7:  {'strafe_left': True, 'turn_right': True},
    8:  {'move_forward': True},
    9:  {'move_forward': True, 'turn_left': True},
    10: {'move_forward': True, 'turn_right': True},
    11: {'move_forward': True, 'strafe_right': True},
    12: {'move_forward': True, 'strafe_right': True, 'turn_left': True},
    13: {'move_forward': True, 'strafe_right': True, 'turn_right': True},
    14: {'move_forward': True, 'strafe_left': True},
    15: {'move_forward': True, 'strafe_left': True, 'turn_left': True},
    16: {'move_forward': True, 'strafe_left': True, 'turn_right': True},
    17: {'shoot': True},
}

# Our 6 base actions (no move_backward or use -- not present in deathmatch data)
BASE_ACTIONS = [
    'shoot',
    'move_forward',
    'turn_left',
    'turn_right',
    'strafe_left',
    'strafe_right',
]


# Semantic similarity between actions: related actions get partial scores
# instead of binary 0/1. This gives the KD loss a smoother gradient landscape.
# Values represent "if action X is active, how relevant is action Y?"
ACTION_AFFINITY = {
    'shoot':        {'shoot': 1.0, 'move_forward': 0.0, 'turn_left': 0.0, 'turn_right': 0.0, 'strafe_left': 0.15, 'strafe_right': 0.15},
    'move_forward': {'shoot': 0.0, 'move_forward': 1.0, 'turn_left': 0.2, 'turn_right': 0.2, 'strafe_left': 0.3, 'strafe_right': 0.3},
    'turn_left':    {'shoot': 0.0, 'move_forward': 0.2, 'turn_left': 1.0, 'turn_right': 0.0, 'strafe_left': 0.35, 'strafe_right': 0.0},
    'turn_right':   {'shoot': 0.0, 'move_forward': 0.2, 'turn_left': 0.0, 'turn_right': 1.0, 'strafe_left': 0.0, 'strafe_right': 0.35},
    'strafe_left':  {'shoot': 0.15, 'move_forward': 0.3, 'turn_left': 0.35, 'turn_right': 0.0, 'strafe_left': 1.0, 'strafe_right': 0.0},
    'strafe_right': {'shoot': 0.15, 'move_forward': 0.3, 'turn_left': 0.0, 'turn_right': 0.35, 'strafe_left': 0.0, 'strafe_right': 1.0},
}


def action_id_to_scores(
    action_id: int,
    mode: str = 'soft',
) -> dict[str, float]:
    """Convert a GameNGen composite action ID to teacher scores.

    Args:
        action_id: Discrete action ID (0--17) from the GameNGen action space.
        mode: Scoring mode.

            - ``'binary'``: Active components get 0.85, inactive get 0.05.
              Simple but causes embedding collapse with KD loss.
            - ``'soft'``: Uses :data:`ACTION_AFFINITY` to compute graded
              scores. Related actions get partial credit (e.g., strafe_left
              gets 0.35 when turn_left is active). Produces a smoother score
              distribution that gives the KD loss more gradient signal.

    Returns:
        Dictionary mapping each base action name to its teacher score.

    Raises:
        KeyError: If *action_id* is not in the valid range 0--17.

    Example:
        >>> scores = action_id_to_scores(9, mode='soft')  # fwd + turn_left
        >>> scores['move_forward']  # active
        0.8
        >>> scores['strafe_left']  # related to turn_left
        0.35
        >>> scores['shoot']  # unrelated
        0.0
    """
    if action_id not in GAMANGEN_ACTIONS:
        raise KeyError(
            f"Unknown action_id {action_id}. Must be in range 0-17."
        )

    active_actions = GAMANGEN_ACTIONS[action_id]

    if mode == 'binary':
        return {
            a: 0.85 if a in active_actions else 0.05
            for a in BASE_ACTIONS
        }

    # mode == 'soft': accumulate affinity from all active actions
    scores = {a: 0.0 for a in BASE_ACTIONS}
    for active in active_actions:
        for target, affinity in ACTION_AFFINITY[active].items():
            scores[target] = max(scores[target], affinity)

    # Scale: active=0.8, partially related=0.2-0.5, unrelated stays 0.0
    for a in BASE_ACTIONS:
        if a in active_actions:
            scores[a] = 0.8
        elif scores[a] > 0:
            scores[a] = scores[a] * 0.6  # scale partial affinities down

    return scores
