# Training API

API reference for the DOOM MultiVec training pipeline components.

---

## Action Mapping

Maps the 18 GameNGen composite actions to 4 base actions with teacher scores. This module defines the action space decomposition used for classifier training.

::: doom_multivec.training.action_mapping

---

## Teacher Scoring

Computes action scores from game action labels. Decomposes PPO agent composite actions into per-base-action scores (4 actions) with optional Gaussian noise.

::: doom_multivec.training.teacher

---

## Dataset Builder

Builds classifier training datasets from HuggingFace VizDoom gameplay recordings and human demonstrations. Handles frame streaming, ASCII conversion, depth binning, and action score construction.

::: doom_multivec.training.dataset
