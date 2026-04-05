"""
Generate paper figures showing the VizDoom -> ASCII + Depth pipeline.

Creates:
  figures/pipeline.pdf - Full pipeline: RGB -> Grayscale -> ASCII -> Depth Buffer -> Depth Bins
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import vizdoom
from doom_multivec.ascii.converter import AsciiConverter


def capture_interesting_frame():
    """Play a few steps to get a frame with an enemy visible."""
    game = vizdoom.DoomGame()
    game.load_config(vizdoom.scenarios_path + '/defend_the_center.cfg')
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_render_hud(True)
    game.set_mode(vizdoom.Mode.PLAYER)

    game.clear_available_buttons()
    game.add_available_button(vizdoom.Button.ATTACK)
    game.add_available_button(vizdoom.Button.MOVE_FORWARD)
    game.add_available_button(vizdoom.Button.TURN_LEFT)
    game.add_available_button(vizdoom.Button.TURN_RIGHT)

    game.set_episode_timeout(2100)
    game.init()
    game.new_episode()

    converter = AsciiConverter(width=40, height=25)

    # Advance a few frames to get enemies on screen
    best_frame = None
    best_depth = None
    best_score = 0

    # Run multiple episodes to find a frame with good character diversity
    for episode in range(5):
        game.new_episode()
        for step in range(150):
            if game.is_episode_finished():
                break

            state = game.get_state()
            if state is None:
                continue

            screen = state.screen_buffer
            depth = state.depth_buffer

            gray = np.mean(screen, axis=2).astype(np.uint8)
            ascii_tmp = converter.convert_simple(gray)
            chars_no_nl = ascii_tmp.replace('\n', '')

            # Count character variety
            from collections import Counter
            char_counts = Counter(chars_no_nl)
            unique_chars = len(char_counts)

            # Prefer: many unique chars, NOT dominated by single char
            top_char_pct = char_counts.most_common(1)[0][1] / len(chars_no_nl)
            # Want top char < 50% and at least 6 unique chars
            diversity_score = unique_chars * (1.0 - top_char_pct)

            # Also want some bright chars (enemy/wall) but not too many
            bright_count = sum(1 for c in chars_no_nl if c in '#%@')
            bright_pct = bright_count / len(chars_no_nl)
            # Sweet spot: 10-40% bright chars
            bright_score = 1.0 - abs(bright_pct - 0.25) * 4

            score = diversity_score * 10 + bright_score * 5

            if score > best_score:
                best_score = score
                best_frame = screen.copy()
                best_depth = depth.copy()
                print(f"  ep{episode} step{step}: unique={unique_chars}, top_pct={top_char_pct:.2f}, bright={bright_pct:.2f}, score={score:.1f}")

            # Alternate actions to explore
            actions = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1]]
            game.make_action(actions[step % len(actions)], 4)

    game.close()
    return best_frame, best_depth


def render_ascii_as_image(grayscale_img, converter, ax, title="ASCII Frame (40x25)"):
    """Render the downscaled grayscale as the ASCII panel -- shows what the converter sees."""
    # Use the actual downscaled grayscale that the ASCII converter works with
    downscaled = converter._downscale(grayscale_img.astype(np.float32), converter.height, converter.width)
    ax.imshow(downscaled, cmap='gray', aspect='auto', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    for spine in ax.spines.values():
        spine.set_color('#333333')


def render_depth_bins_as_image(depth_bins, ax, title="Depth Bins (16 bins)"):
    """Render depth bins as a colored heatmap matching ASCII layout."""
    # Reshape depth bins to 25x40 grid (skip newlines)
    grid = np.zeros((25, 40), dtype=int)
    idx = 0
    for y in range(25):
        for x in range(40):
            if idx < len(depth_bins):
                grid[y, x] = depth_bins[idx]
                idx += 1
        idx += 1  # skip newline token

    cmap = plt.cm.RdYlGn_r  # Red=near, Green=far
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=15, aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    return im


def main():
    print("Capturing VizDoom frame...")
    rgb_frame, depth_buffer = capture_interesting_frame()

    converter = AsciiConverter(width=40, height=25)

    # Convert to grayscale
    grayscale = np.mean(rgb_frame, axis=2).astype(np.uint8)

    # Convert to ASCII + depth
    ascii_text, depth_bins = converter.convert_with_depth(
        grayscale, depth_buffer.astype(np.float32), num_bins=16
    )

    print(f"Frame shape: {rgb_frame.shape}")
    print(f"ASCII length: {len(ascii_text)}")
    print(f"Depth bins: {len(depth_bins)}")
    print(f"ASCII preview:\n{ascii_text[:200]}...")

    # === Figure 1: Full pipeline ===
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.04, right=0.96, top=0.93, bottom=0.04)

    # (a) RGB Frame
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_frame)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('(a) VizDoom RGB Frame\n640 x 480 x 3', fontsize=10, fontweight='bold')

    # (b) Grayscale
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(grayscale, cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('(b) Grayscale\n640 x 480', fontsize=10, fontweight='bold')

    # (c) Depth Buffer
    ax3 = fig.add_subplot(gs[0, 2])
    # Clip extreme depth values for better visualization
    depth_viz = depth_buffer.copy()
    depth_viz = np.clip(depth_viz, 0, np.percentile(depth_viz, 95))
    im3 = ax3.imshow(depth_viz, cmap='plasma')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('(c) VizDoom Depth Buffer\n640 x 480', fontsize=10, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Distance')

    # (d) ASCII Frame
    ax4 = fig.add_subplot(gs[1, 0])
    render_ascii_as_image(grayscale, converter, ax4, '(d) ASCII Frame\n40 x 25 characters')

    # (e) Depth Bins heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = render_depth_bins_as_image(depth_bins, ax5, '(e) Depth Bins\n40 x 25, 16 bins')
    cbar = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar.set_label('Bin (0=near, 15=far)', fontsize=8)

    # (f) Combined: ASCII brightness * depth as RGBA image
    ax6 = fig.add_subplot(gs[1, 2])
    lines = ascii_text.split('\n')
    h = len(lines)
    w = max(len(l) for l in lines) if lines else 40
    chars = ' .:-=+*#%@'
    cmap_depth = plt.cm.RdYlGn_r

    combined = np.zeros((h, w, 4))
    idx = 0
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            brightness = chars.index(ch) / (len(chars) - 1) if ch in chars else 0.5
            if idx < len(depth_bins):
                depth_color = np.array(cmap_depth(depth_bins[idx] / 15.0))
                # Modulate depth color by brightness
                depth_color[:3] *= (0.3 + 0.7 * brightness)
                combined[y, x] = depth_color
                idx += 1
            else:
                combined[y, x] = [brightness, brightness, brightness, 1.0]
        idx += 1  # newline

    ax6.imshow(combined, aspect='auto', interpolation='nearest')
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title('(f) Model Input: ASCII + Depth\nBrightness x Depth Color', fontsize=11, fontweight='bold')
    for spine in ax6.spines.values():
        spine.set_color('#333333')

    # Main title
    fig.suptitle('DOOM MultiVec Input Pipeline: Game Frame to ASCII + Depth Representation',
                 fontsize=13, fontweight='bold', y=0.98)

    # Arrow annotations between panels
    fig.text(0.34, 0.72, r'$\rightarrow$', fontsize=20, ha='center', va='center', color='#555')
    fig.text(0.66, 0.72, r'$\rightarrow$', fontsize=20, ha='center', va='center', color='#555')
    fig.text(0.20, 0.50, r'$\downarrow$', fontsize=20, ha='center', va='center', color='#555')
    fig.text(0.52, 0.50, r'$\downarrow$', fontsize=20, ha='center', va='center', color='#555')
    fig.text(0.67, 0.32, r'$\rightarrow$', fontsize=20, ha='center', va='center', color='#555')

    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'pipeline.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Also save PNG for preview
    out_png = out_path.replace('.pdf', '.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")

    plt.close()


if __name__ == '__main__':
    main()
