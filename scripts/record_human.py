"""
Record human DOOM gameplay with depth buffer.
SPECTATOR mode — you play with native DOOM controls, we record everything.

Controls (native DOOM):
  Up Arrow    = move forward
  Down Arrow  = move backward
  Left Arrow  = turn left
  Right Arrow = turn right
  Ctrl        = shoot

Usage:
  python scripts/record_human.py --scenario defend_the_center --episodes 20
  python scripts/record_human.py --scenario deadly_corridor --episodes 20
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import vizdoom
from doom_multivec.ascii.converter import AsciiConverter

# 4 actions (no strafe)
ACTIONS = ['shoot', 'move_forward', 'turn_left', 'turn_right']


def setup_game(scenario='defend_the_center'):
    """Set up VizDoom in SPECTATOR mode."""
    game = vizdoom.DoomGame()

    scenarios = {
        'basic': vizdoom.scenarios_path + '/basic.cfg',
        'defend_the_center': vizdoom.scenarios_path + '/defend_the_center.cfg',
        'health_gathering': vizdoom.scenarios_path + '/health_gathering.cfg',
        'deadly_corridor': vizdoom.scenarios_path + '/deadly_corridor.cfg',
        'my_way_home': vizdoom.scenarios_path + '/my_way_home.cfg',
    }
    cfg = scenarios.get(scenario)
    if cfg:
        game.load_config(cfg)
    else:
        game.load_config(scenario)

    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(True)
    game.set_render_hud(True)

    game.set_mode(vizdoom.Mode.SPECTATOR)

    # 4 buttons matching our actions
    game.clear_available_buttons()
    game.add_available_button(vizdoom.Button.ATTACK)        # 0
    game.add_available_button(vizdoom.Button.MOVE_FORWARD)  # 1
    game.add_available_button(vizdoom.Button.TURN_LEFT)     # 2
    game.add_available_button(vizdoom.Button.TURN_RIGHT)    # 3

    game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)

    game.set_episode_timeout(4200)
    game.init()
    return game


def buttons_to_scores(buttons):
    """Convert 4-button vector to soft score vector."""
    scores = {a: 0.0 for a in ACTIONS}

    if len(buttons) >= 4:
        if buttons[0]:  scores['shoot'] = 0.8
        if buttons[1]:  scores['move_forward'] = 0.8
        if buttons[2]:  scores['turn_left'] = 0.8
        if buttons[3]:  scores['turn_right'] = 0.8

    # Soft affinities
    if scores['turn_left'] > 0.5:
        scores['move_forward'] = max(scores['move_forward'], 0.2)
    if scores['turn_right'] > 0.5:
        scores['move_forward'] = max(scores['move_forward'], 0.2)
    if scores['move_forward'] > 0.5:
        scores['turn_left'] = max(scores['turn_left'], 0.15)
        scores['turn_right'] = max(scores['turn_right'], 0.15)

    # Idle frame
    if sum(1 for v in scores.values() if v > 0.5) == 0:
        scores['move_forward'] = 0.2

    return [scores[a] for a in ACTIONS]


def main():
    parser = argparse.ArgumentParser(description='Record human DOOM gameplay')
    parser.add_argument('--scenario', default='defend_the_center')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output', default='data/doom-human')
    parser.add_argument('--ascii-width', type=int, default=40)
    parser.add_argument('--ascii-height', type=int, default=25)
    parser.add_argument('--depth-bins', type=int, default=16)
    parser.add_argument('--frame-skip', type=int, default=4)
    args = parser.parse_args()

    converter = AsciiConverter(width=args.ascii_width, height=args.ascii_height)
    game = setup_game(args.scenario)

    all_texts = []
    all_depth_bins = []
    all_scores = []

    print(f"\n{'='*60}")
    print(f"  DOOM Recorder (SPECTATOR mode)")
    print(f"  Scenario: {args.scenario}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Actions: {ACTIONS}")
    print(f"{'='*60}")
    print(f"  Controls (click DOOM window to focus):")
    print(f"    Up Arrow    = move forward")
    print(f"    Down Arrow  = move backward")
    print(f"    Left Arrow  = turn left")
    print(f"    Right Arrow = turn right")
    print(f"    Ctrl        = shoot")
    print(f"{'='*60}\n")

    try:
        for ep in range(args.episodes):
            game.new_episode()
            step = 0
            ep_actions = {a: 0 for a in ACTIONS}

            print(f"Episode {ep+1}/{args.episodes} — Play!")

            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    break

                screen = state.screen_buffer
                depth = state.depth_buffer

                # Advance (human plays in spectator mode)
                game.advance_action(args.frame_skip)
                action = game.get_last_action()

                if screen is None:
                    continue

                gray = np.mean(screen, axis=2).astype(np.uint8)

                # ASCII + depth
                if depth is not None:
                    ascii_text, depth_bins_list = converter.convert_with_depth(
                        gray, depth.astype(np.float32), num_bins=args.depth_bins
                    )
                else:
                    ascii_text = converter.convert_simple(gray)
                    depth_bins_list = [args.depth_bins] * len(ascii_text)

                # Scores from human action
                score_vec = buttons_to_scores(action)

                all_texts.append(ascii_text)
                all_depth_bins.append(depth_bins_list)
                all_scores.append(score_vec)

                for i, a in enumerate(ACTIONS):
                    if score_vec[i] > 0.5:
                        ep_actions[a] += 1
                step += 1

            try:
                kills = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
            except:
                kills = 0
            print(f"  Steps: {step}, Kills: {kills:.0f}")
            action_str = ", ".join(f"{a}={c}" for a, c in ep_actions.items() if c > 0)
            print(f"  Actions: {action_str}")
            print(f"  Total frames: {len(all_texts):,}\n")

    except Exception as e:
        print(f"\n  Crash: {e}")
        print(f"  Saving {len(all_texts):,} frames collected so far...")
    finally:
        try:
            game.close()
        except:
            pass

    print(f"\nRecorded {len(all_texts):,} frames total")
    if not all_texts:
        print("No frames recorded!")
        return

    # Save + pre-tokenize
    print("Saving and pre-tokenizing...")
    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained('models/doom-multivec-5L')
    no_depth = args.depth_bins

    ds = Dataset.from_dict({
        'text': all_texts,
        'depth_bins': all_depth_bins,
        'scores': all_scores,
    })

    def tokenize_with_depth(batch):
        encoded = tok(batch['text'], max_length=1100, padding='max_length', truncation=True)
        all_depth_ids = []
        for j in range(len(batch['text'])):
            ids = encoded['input_ids'][j]
            depth_data = batch['depth_bins'][j]
            depth_ids = [no_depth]
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

    ds_tok = ds.map(tokenize_with_depth, batched=True, batch_size=1000,
                    remove_columns=['text', 'depth_bins'])
    os.makedirs(args.output, exist_ok=True)
    ds_tok.save_to_disk(args.output)

    scores_arr = np.array(all_scores)
    primary = scores_arr.argmax(axis=1)
    counts = np.bincount(primary, minlength=len(ACTIONS))
    print(f"\nSaved {len(ds_tok):,} frames to {args.output}")
    print(f"Action distribution:")
    for i, a in enumerate(ACTIONS):
        print(f"  {a:15s}: {counts[i]:>6,} ({counts[i]/len(ds_tok)*100:.1f}%)")


if __name__ == '__main__':
    main()
