# Gameplay

Watch the trained DOOM MultiVec classifier play DOOM in real time using `scripts/play_doom_visual.py`. The script opens a VizDoom window, feeds each frame through the ASCII converter and classifier, and executes the predicted actions.

---

## Running the Player

```bash
python scripts/play_doom_visual.py \
    --model models/doom-multivec-trained \
    --scenario basic \
    --episodes 3
```

The window shows the DOOM game running with the model making all decisions. Per-step logging prints the chosen action, health, frags, and inference latency every 50 steps.

---

## Available Scenarios

VizDoom ships with several built-in scenarios, each testing different skills:

| Scenario | Flag | Description | Actions Tested |
|---|---|---|---|
| Basic | `--scenario basic` | Shoot a single enemy in a room | shoot, turn |
| Defend the Center | `--scenario defend_the_center` | Enemies approach from all sides, survive as long as possible | shoot, turn_left, turn_right |
| Deadly Corridor | `--scenario deadly_corridor` | Navigate a corridor full of enemies to reach a vest | move_forward, shoot, turn |
| My Way Home | `--scenario my_way_home` | Navigate a maze to find the goal | move_forward, turn_left, turn_right |
| Health Gathering | `--scenario health_gathering` | Collect health packs on toxic floor before dying | move_forward, turn_left, turn_right |

You can also pass a path to any custom `.cfg` file instead of a scenario name.

---

## FPS Control

The `--frame-skip` flag controls how many DOOM tics pass between model decisions. DOOM runs at 35 tics per second internally.

```bash
# Default: decide every 4 tics (~8.75 decisions/sec, real-time feel)
python scripts/play_doom_visual.py --model models/doom-multivec-trained --frame-skip 4

# Faster gameplay, less responsive (every 8 tics)
python scripts/play_doom_visual.py --model models/doom-multivec-trained --frame-skip 8

# More responsive, slower overall (every 2 tics)
python scripts/play_doom_visual.py --model models/doom-multivec-trained --frame-skip 2
```

The `--fps` flag controls the visual playback speed:

| Flag | Effect |
|---|---|
| `--fps 30` | Real-time (default) |
| `--fps 60` | Fast playback |
| `--fps 10` | Slow motion -- useful for debugging action choices |

---

## Composite Actions

By default, the player combines the top-2 predicted actions when they come from different categories (movement + rotation, movement + combat, etc.). This produces natural-looking gameplay where the model can move and turn simultaneously.

Disable composite actions with `--no-composite` to see single-action-per-step behavior:

```bash
python scripts/play_doom_visual.py --model models/doom-multivec-trained --no-composite
```

---

## Observations by Scenario

### my_way_home (best performance)

The maze navigation scenario works best with the current model. The model learns to explore by combining `move_forward` with turning actions, and reliably navigates toward the goal. The ASCII representation captures corridor geometry well, giving the model clear spatial signals.

### basic

The model handles the basic scenario adequately -- it turns to face the enemy and fires. Performance depends on how quickly the model locks onto the enemy character (`E`) in the ASCII frame.

### defend_the_center

Works reasonably well. The model turns to face approaching enemies and shoots. It can struggle when enemies approach from multiple directions simultaneously, as the single-frame classifier has no memory of off-screen threats.

### deadly_corridor

Mixed results. The model tries to advance through the corridor but takes heavy damage. Successful runs require good composite actions (move_forward + shoot) that the model sometimes produces.

### health_gathering

The model moves around and collects some health packs, but lacks the strategic planning to optimize collection routes. Edge-running behavior is common -- the model tends to follow walls rather than cutting across open space, likely because wall patterns in the ASCII frame provide strong directional signals.

!!! note "Arena edge-running"
    In open arena scenarios, the model often runs along the edges of the map. This happens because wall characters (`#`, `W`) at screen edges create strong spatial gradients in the ASCII representation that bias the model toward turning along walls rather than moving into open space. This is a known artifact of the ASCII encoding approach.
