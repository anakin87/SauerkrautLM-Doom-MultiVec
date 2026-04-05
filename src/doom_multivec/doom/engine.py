"""VizDoom engine wrapper for frame capture and action execution.

Provides :class:`DoomEngine`, a thin wrapper around VizDoom that configures
the game for ASCII-art conversion (grayscale screen, depth buffer, label
buffer), and :class:`MockDoomEngine`, a drop-in replacement that generates
synthetic frames for testing without VizDoom installed.
"""

import numpy as np

try:
    import vizdoom
    HAS_VIZDOOM = True
except ImportError:
    HAS_VIZDOOM = False


DOOM_ACTIONS: dict[str, str] = {
    'move_forward':  'MOVE_FORWARD',
    'move_backward': 'MOVE_BACKWARD',
    'turn_left':     'TURN_LEFT',
    'turn_right':    'TURN_RIGHT',
    'strafe_left':   'MOVE_LEFT',
    'strafe_right':  'MOVE_RIGHT',
    'shoot':         'ATTACK',
    'use':           'USE',
}
"""Mapping from human-readable action names to VizDoom ``Button`` names."""


class DoomEngine:
    """Wrapper around VizDoom for frame capture and action execution.

    Configures VizDoom with grayscale rendering, depth buffer, and label
    buffer so that the captured state can be converted to ASCII art by
    :class:`~doom_multivec.ascii.converter.AsciiConverter`.

    Attributes:
        game: The underlying ``vizdoom.DoomGame`` instance.
        button_names: Ordered list of action name strings matching the
            configured available buttons.
    """

    def __init__(self, scenario='basic', screen_res=(160, 120), frame_skip=4):
        """Initialise and configure VizDoom.

        Args:
            scenario: Scenario name (``'basic'``, ``'defend_the_center'``,
                etc.) or path to a ``.cfg`` file.  Defaults to ``'basic'``.
            screen_res: ``(width, height)`` tuple for the screen resolution.
                Defaults to ``(160, 120)``.
            frame_skip: Number of game tics per step (unused directly here
                but accepted for API consistency).  Defaults to 4.

        Raises:
            ImportError: If the ``vizdoom`` package is not installed.
        """
        if not HAS_VIZDOOM:
            raise ImportError("vizdoom is required. Install with: pip install vizdoom")

        self.game = vizdoom.DoomGame()
        self._setup(scenario, screen_res, frame_skip)
        self.button_names = []

    def _setup(self, scenario, screen_res, frame_skip):
        """Configure VizDoom game settings, buttons, and variables.

        Args:
            scenario: Scenario name or path to ``.cfg`` file.
            screen_res: ``(width, height)`` screen resolution tuple.
            frame_skip: Unused (kept for future use).
        """
        # Scenario
        scenarios = {
            'basic': 'basic.cfg',
            'defend_the_center': 'defend_the_center.cfg',
            'health_gathering': 'health_gathering.cfg',
            'deadly_corridor': 'deadly_corridor.cfg',
            'my_way_home': 'my_way_home.cfg',
        }
        cfg = scenarios.get(scenario, scenario)
        self.game.load_config(cfg)

        # Screen
        res_map = {
            (160, 120): vizdoom.ScreenResolution.RES_160X120,
            (320, 200): vizdoom.ScreenResolution.RES_320X200,
            (320, 240): vizdoom.ScreenResolution.RES_320X240,
        }
        self.game.set_screen_resolution(res_map.get(screen_res, vizdoom.ScreenResolution.RES_160X120))
        self.game.set_screen_format(vizdoom.ScreenFormat.GRAY8)

        # Buffers
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)

        # Display
        self.game.set_window_visible(False)
        self.game.set_render_hud(False)

        # Gameplay
        self.game.set_episode_timeout(2100)  # ~60 seconds at 35 FPS
        self.game.set_episode_start_time(10)

        # Available buttons - ensure all our actions are available
        self.game.add_available_button(vizdoom.Button.MOVE_FORWARD)
        self.game.add_available_button(vizdoom.Button.MOVE_BACKWARD)
        self.game.add_available_button(vizdoom.Button.TURN_LEFT)
        self.game.add_available_button(vizdoom.Button.TURN_RIGHT)
        self.game.add_available_button(vizdoom.Button.MOVE_LEFT)
        self.game.add_available_button(vizdoom.Button.MOVE_RIGHT)
        self.game.add_available_button(vizdoom.Button.ATTACK)
        self.game.add_available_button(vizdoom.Button.USE)

        # Game variables for teacher scoring
        self.game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
        self.game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
        self.game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)

        self.game.set_mode(vizdoom.Mode.PLAYER)

    def init(self):
        """Initialise the VizDoom game and populate :attr:`button_names`."""
        self.game.init()
        n_buttons = self.game.get_available_buttons_size()
        self.button_names = list(DOOM_ACTIONS.keys())[:n_buttons]

    def new_episode(self):
        """Start a new game episode, resetting the environment."""
        self.game.new_episode()

    def is_episode_finished(self):
        """Check whether the current episode has ended.

        Returns:
            ``True`` if the episode is finished, ``False`` otherwise.
        """
        return self.game.is_episode_finished()

    def get_state(self):
        """Retrieve the current game state buffers and variables.

        Returns:
            A 5-tuple ``(screen_buffer, depth_buffer, labels_buffer, labels,
            game_vars)`` where each buffer is a ``numpy.ndarray`` or ``None``
            if unavailable, and *game_vars* is a dict with keys
            ``'health'``, ``'ammo'``, and ``'killcount'``.
        """
        state = self.game.get_state()
        if state is None:
            return None, None, None, None, None

        screen = state.screen_buffer
        depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None
        labels_buf = state.labels_buffer if hasattr(state, 'labels_buffer') else None
        labels = state.labels if hasattr(state, 'labels') else None

        game_vars = {
            'health': self.game.get_game_variable(vizdoom.GameVariable.HEALTH),
            'ammo': self.game.get_game_variable(vizdoom.GameVariable.AMMO2),
            'killcount': self.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT),
        }

        return screen, depth, labels_buf, labels, game_vars

    def make_action(self, action_name, frame_repeat=4):
        """Execute a single DOOM action.

        Args:
            action_name: One of the keys in :data:`DOOM_ACTIONS` (e.g.
                ``'shoot'``, ``'move_forward'``).
            frame_repeat: Number of game tics to hold the action.
                Defaults to 4.

        Returns:
            The reward received from VizDoom for the action.
        """
        action_vector = self._name_to_vector(action_name)
        return self.game.make_action(action_vector, frame_repeat)

    def make_composite_action(self, action_names, frame_repeat=4):
        """Execute multiple DOOM actions simultaneously.

        Args:
            action_names: List of action name strings (keys in
                :data:`DOOM_ACTIONS`).
            frame_repeat: Number of game tics to hold the actions.
                Defaults to 4.

        Returns:
            The reward received from VizDoom for the composite action.
        """
        vector = [0] * self.game.get_available_buttons_size()
        for name in action_names:
            idx = self.button_names.index(name) if name in self.button_names else -1
            if idx >= 0:
                vector[idx] = 1
        return self.game.make_action(vector, frame_repeat)

    def _name_to_vector(self, action_name):
        """Convert an action name to a VizDoom button vector.

        Args:
            action_name: One of the keys in :data:`DOOM_ACTIONS`.

        Returns:
            List of 0/1 integers matching the available-buttons ordering.
        """
        n = self.game.get_available_buttons_size()
        vector = [0] * n
        if action_name in self.button_names:
            idx = self.button_names.index(action_name)
            vector[idx] = 1
        return vector

    def close(self):
        """Shut down VizDoom and release resources."""
        self.game.close()


class MockDoomEngine:
    """Mock DOOM engine for testing without VizDoom installed.

    Generates random frames with synthetic corridor-like patterns so that
    downstream code (ASCII conversion, tokenization, model inference) can be
    exercised without the ``vizdoom`` system dependency.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        episode_step: Current step within the episode.
        max_steps: Maximum steps before the episode ends.
    """

    def __init__(self, width=160, height=120):
        """Initialise the mock engine.

        Args:
            width: Frame width in pixels.  Defaults to 160.
            height: Frame height in pixels.  Defaults to 120.
        """
        self.width = width
        self.height = height
        self.episode_step = 0
        self.max_steps = 500
        self._finished = False

    def init(self):
        """No-op initialisation (mock has no external dependencies)."""
        pass

    def new_episode(self):
        """Reset the episode step counter."""
        self.episode_step = 0
        self._finished = False

    def is_episode_finished(self):
        """Check whether the mock episode has ended.

        Returns:
            ``True`` if the step limit has been reached.
        """
        return self._finished or self.episode_step >= self.max_steps

    def get_state(self):
        """Generate a synthetic game state with corridor-like patterns.

        Returns:
            A 5-tuple ``(screen_buffer, depth_buffer, labels_buffer, labels,
            game_vars)`` matching the :class:`DoomEngine` interface.
            ``labels_buffer`` and ``labels`` are always ``None``.
        """
        if self.is_episode_finished():
            return None, None, None, None, None

        # Generate synthetic frame with some structure
        screen = np.random.randint(0, 256, (self.height, self.width), dtype=np.uint8)

        # Add a "corridor" pattern (darker at edges, brighter in center)
        center_x = self.width // 2
        for x in range(self.width):
            dist = abs(x - center_x) / center_x
            screen[:, x] = (screen[:, x] * (1 - 0.5 * dist)).astype(np.uint8)

        # Synthetic depth (closer in center)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                depth[y, x] = 100 + 200 * (abs(x - center_x) / center_x)

        # No labels in mock
        game_vars = {'health': 100, 'ammo': 50, 'killcount': 0}

        self.episode_step += 1
        return screen, depth, None, None, game_vars

    def make_action(self, action_name, frame_repeat=4):
        """Advance the mock episode by one step, ignoring the action.

        Args:
            action_name: Action name (ignored).
            frame_repeat: Frame repeat count (ignored).

        Returns:
            Always ``0.0``.
        """
        self.episode_step += 1
        return 0.0

    def make_composite_action(self, action_names, frame_repeat=4):
        """Advance the mock episode by one step, ignoring the actions.

        Args:
            action_names: List of action names (ignored).
            frame_repeat: Frame repeat count (ignored).

        Returns:
            Always ``0.0``.
        """
        self.episode_step += 1
        return 0.0

    def close(self):
        """No-op cleanup."""
        pass
