"""
Collect DOOM frames with depth buffer using VizDoom.
Runs headless with a random/heuristic agent and saves frames + depth + actions.

Usage:
  python scripts/collect_vizdoom_depth.py --episodes 500 --output data/doom-depth-500ep
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import vizdoom
from doom_multivec.ascii.converter import AsciiConverter
from doom_multivec.training.action_mapping import BASE_ACTIONS, action_id_to_scores


# The 18 GameNGen-compatible actions
# Button order: [ATTACK, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT, TURN_RIGHT, TURN_LEFT]
ACTIONS_18 = []
import itertools
for combo in itertools.product([0, 1], repeat=6):
    if sum(combo) == 0:
        continue  # skip no-op
    if combo[0] == 1 and sum(combo) > 1:
        continue  # ATTACK is exclusive
    if combo[2] == 1 and combo[3] == 1:
        continue  # MOVE_LEFT + MOVE_RIGHT mutually exclusive
    if combo[4] == 1 and combo[5] == 1:
        continue  # TURN_LEFT + TURN_RIGHT mutually exclusive
    ACTIONS_18.append(list(combo))


def setup_game(scenario='deathmatch'):
    """Set up VizDoom for data collection with depth buffer."""
    game = vizdoom.DoomGame()

    if scenario == 'deathmatch':
        game.set_doom_scenario_path(vizdoom.scenarios_path + '/deathmatch.wad')
        game.set_doom_map("map01")
    else:
        scenarios = {
            'basic': '/basic.cfg',
            'defend_the_center': '/defend_the_center.cfg',
            'health_gathering': '/health_gathering.cfg',
            'deadly_corridor': '/deadly_corridor.cfg',
        }
        game.load_config(vizdoom.scenarios_path + scenarios.get(scenario, scenarios['basic']))

    game.set_screen_resolution(vizdoom.ScreenResolution.RES_160X120)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_render_hud(False)

    # Buttons: ATTACK, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT, TURN_RIGHT, TURN_LEFT
    game.add_available_button(vizdoom.Button.ATTACK)
    game.add_available_button(vizdoom.Button.MOVE_FORWARD)
    game.add_available_button(vizdoom.Button.MOVE_LEFT)
    game.add_available_button(vizdoom.Button.MOVE_RIGHT)
    game.add_available_button(vizdoom.Button.TURN_RIGHT)
    game.add_available_button(vizdoom.Button.TURN_LEFT)

    game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)

    game.set_episode_timeout(4200)  # ~2 minutes
    game.set_mode(vizdoom.Mode.PLAYER)

    game.init()
    return game


def heuristic_action(state, game, rng):
    """Simple heuristic agent: mostly random but biased toward useful actions."""
    # 70% random, 30% move forward (exploration bias)
    if rng.random() < 0.3:
        return 8  # MOVE_FORWARD (action index in ACTIONS_18)
    return rng.integers(0, len(ACTIONS_18))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--scenario', default='deathmatch')
    parser.add_argument('--output', default='data/doom-depth')
    parser.add_argument('--ascii-width', type=int, default=40)
    parser.add_argument('--ascii-height', type=int, default=25)
    parser.add_argument('--depth-bins', type=int, default=16)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    converter = AsciiConverter(width=args.ascii_width, height=args.ascii_height)

    print(f"Collecting VizDoom data with depth buffer")
    print(f"  Scenario: {args.scenario}")
    print(f"  Episodes: {args.episodes}")
    print(f"  ASCII: {args.ascii_width}x{args.ascii_height}")
    print(f"  Depth bins: {args.depth_bins}")

    game = setup_game(args.scenario)
    print(f"  Actions: {len(ACTIONS_18)}")

    all_texts = []
    all_depth_bins = []
    all_scores = []

    t0 = time.time()

    for ep in range(args.episodes):
        game.new_episode()
        step = 0

        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break

            screen = state.screen_buffer
            depth = state.depth_buffer

            # Pick action
            action_idx = heuristic_action(state, game, rng)
            action_vec = ACTIONS_18[action_idx]

            # Convert to ASCII + depth bins
            if depth is not None:
                ascii_text, depth_bins = converter.convert_with_depth(
                    screen, depth.astype(np.float32), num_bins=args.depth_bins
                )
            else:
                ascii_text = converter.convert_simple(screen)
                depth_bins = [args.depth_bins] * len(ascii_text)  # all "no depth"

            # Get teacher scores from action
            scores = action_id_to_scores(action_idx, mode='soft')
            score_vec = [scores[a] for a in BASE_ACTIONS]

            all_texts.append(ascii_text)
            all_depth_bins.append(depth_bins)
            all_scores.append(score_vec)

            # Execute action
            game.make_action(action_vec, args.frame_skip)
            step += 1

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps = len(all_texts) / elapsed
            print(f"  Episode {ep+1}/{args.episodes}: {len(all_texts):,} frames ({fps:.0f} fps)")

    game.close()

    print(f"\nCollected {len(all_texts):,} frames with depth")

    # Oversample minority classes (same as classifier dataset)
    scores_arr = np.array(all_scores)
    primary = scores_arr.argmax(axis=1)
    counts = np.bincount(primary, minlength=6)
    target = int(np.percentile(counts[counts > 0], 25))

    print(f"\nBalancing (target p25 = {target:,}):")
    extra_idx = []
    for i, a in enumerate(BASE_ACTIONS):
        if counts[i] < target and counts[i] > 0:
            idx = np.where(primary == i)[0]
            needed = target - counts[i]
            extra = rng.choice(idx, size=needed, replace=True)
            extra_idx.extend(extra.tolist())
            print(f"  {a}: {counts[i]:,} + {needed:,}")
        else:
            print(f"  {a}: {counts[i]:,} OK")

    if extra_idx:
        for idx in extra_idx:
            all_texts.append(all_texts[idx])
            all_depth_bins.append(all_depth_bins[idx])
            all_scores.append(all_scores[idx])

    # Save raw first, then tokenize via Dataset.map (memory efficient)
    from datasets import Dataset

    print(f"\nSaving {len(all_texts):,} raw samples...")
    ds_raw = Dataset.from_dict({
        'text': all_texts,
        'depth_bins': all_depth_bins,
        'scores': all_scores,
    })
    ds_raw = ds_raw.shuffle(seed=args.seed)

    # Pre-tokenize using Dataset.map (streams, no OOM)
    print(f"Pre-tokenizing...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('models/doom-multivec-5L')
    no_depth = args.depth_bins

    def tokenize_with_depth(batch):
        encoded = tok(batch['text'], max_length=1100, padding='max_length', truncation=True)
        all_depth_ids = []
        for j in range(len(batch['text'])):
            ids = encoded['input_ids'][j]
            depth_data = batch['depth_bins'][j]
            # Align: [CLS]=no_depth, chars=depth_data, [SEP]+[PAD]=no_depth
            depth_ids = [no_depth]  # CLS
            for k in range(min(len(depth_data), len(ids) - 2)):
                depth_ids.append(depth_data[k])
            while len(depth_ids) < len(ids):
                depth_ids.append(no_depth)
            all_depth_ids.append(depth_ids[:len(ids)])
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'depth_ids': all_depth_ids,
            'scores': batch['scores'],
        }

    ds_tok = ds_raw.map(tokenize_with_depth, batched=True, batch_size=1000,
                        remove_columns=['text', 'depth_bins'])
    ds_tok.save_to_disk(args.output)

    # Stats
    scores_final = np.array(ds_tok['scores'])
    primary_final = scores_final.argmax(axis=1)
    counts_final = np.bincount(primary_final, minlength=6)
    print(f"\nSaved {len(ds_tok):,} pre-tokenized samples to {args.output}")
    print(f"Columns: {ds_tok.column_names}")
    print(f"Distribution:")
    for i, a in enumerate(BASE_ACTIONS):
        print(f"  {a:15s}: {counts_final[i]:>8,} ({counts_final[i]/len(ds_tok)*100:.1f}%)")


if __name__ == '__main__':
    main()
