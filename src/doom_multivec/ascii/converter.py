"""ASCII art conversion of VizDoom game frames.

This module provides the :class:`AsciiConverter` class, which turns raw
VizDoom screen, depth, and label buffers into compact ASCII text
representations suitable for tokenization by the DOOM MultiVec model.

The conversion uses a hybrid brightness/entity approach: background pixels
are mapped to brightness characters (dark to bright), while recognised game
entities (enemies, items, doors, etc.) are represented by dedicated letter
markers that override brightness.
"""

import numpy as np
from .charset import BRIGHTNESS_CHARS, ENTITY_CHARS, LABEL_TO_ENTITY, DEFAULT_WIDTH, DEFAULT_HEIGHT


class AsciiConverter:
    """Converts VizDoom frames to ASCII text representation.

    Uses a hybrid approach where entity labels from VizDoom's labels buffer
    override brightness-based characters, and depth or screen brightness is
    mapped to ASCII characters for background regions.

    Attributes:
        width: Target width of the ASCII frame in characters.
        height: Target height of the ASCII frame in characters.
        brightness_chars: String of ASCII characters ordered from dark to bright.
        n_levels: Number of distinct brightness levels (length of
            ``brightness_chars``).

    Example:
        >>> converter = AsciiConverter(width=40, height=25)
        >>> ascii_frame = converter.convert_simple(screen_buffer)
        >>> print(ascii_frame)
    """

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        """Initialise the converter with target ASCII resolution.

        Args:
            width: Target width in characters. Defaults to ``DEFAULT_WIDTH`` (40).
            height: Target height in characters. Defaults to ``DEFAULT_HEIGHT`` (25).
        """
        self.width = width
        self.height = height
        self.brightness_chars = BRIGHTNESS_CHARS
        self.n_levels = len(self.brightness_chars)

    def convert(self, screen_buffer, depth_buffer=None, labels_buffer=None, labels=None):
        """Convert VizDoom buffers to an ASCII string.

        This is the full-featured conversion path.  When depth and label
        buffers are provided the output uses inverse depth for brightness
        and overlays entity markers for recognised game objects.

        Args:
            screen_buffer: ``(H, W)`` grayscale or ``(H, W, 3)`` RGB uint8
                array captured from VizDoom.
            depth_buffer: Optional ``(H, W)`` float array with per-pixel
                depth values.  When provided, depth is used instead of screen
                brightness for the background characters.
            labels_buffer: Optional ``(H, W)`` uint8 array of VizDoom label
                IDs.  Each pixel value corresponds to a label entry.
            labels: List of VizDoom ``Label`` objects that map label IDs to
                object names (required together with *labels_buffer*).

        Returns:
            A multi-line string of size ``height`` rows by ``width`` columns
            with ``'\\n'`` as the row separator.

        Example:
            >>> converter = AsciiConverter()
            >>> ascii_text = converter.convert(
            ...     screen_buffer, depth_buffer, labels_buffer, labels
            ... )
            >>> print(ascii_text)
        """
        # Convert to grayscale if RGB
        if screen_buffer.ndim == 3:
            gray = np.mean(screen_buffer, axis=2).astype(np.uint8)
        else:
            gray = screen_buffer

        # Downscale to target resolution using block averaging
        gray_resized = self._downscale(gray, self.height, self.width)

        # Build entity map if labels available
        entity_map = None
        if labels_buffer is not None and labels is not None:
            entity_map = self._build_entity_map(labels_buffer, labels)

        # Use depth for brightness if available (more informative than screen brightness)
        if depth_buffer is not None:
            brightness = self._depth_to_brightness(depth_buffer)
        else:
            brightness = gray_resized

        # Convert to ASCII
        return self._to_ascii(brightness, entity_map)

    def _downscale(self, img, target_h, target_w):
        """Downscale an image to the target resolution via block averaging.

        Args:
            img: 2-D ``numpy.ndarray`` of shape ``(H, W)``.
            target_h: Desired output height in pixels.
            target_w: Desired output width in pixels.

        Returns:
            ``numpy.ndarray`` of shape ``(target_h, target_w)`` with dtype
            ``uint8``.
        """
        h, w = img.shape
        # Calculate block size
        bh = h // target_h
        bw = w // target_w
        # Crop to exact multiple
        cropped = img[:bh * target_h, :bw * target_w]
        # Reshape and average
        return cropped.reshape(target_h, bh, target_w, bw).mean(axis=(1, 3)).astype(np.uint8)

    def _depth_to_brightness(self, depth_buffer):
        """Convert a depth buffer to brightness values (near=bright, far=dark).

        Args:
            depth_buffer: ``(H, W)`` depth array from VizDoom.

        Returns:
            ``numpy.ndarray`` of shape ``(height, width)`` with uint8
            brightness values where 255 is closest and 0 is farthest.
        """
        depth_resized = self._downscale(
            depth_buffer.astype(np.uint8) if depth_buffer.dtype != np.uint8 else depth_buffer,
            self.height, self.width
        )
        # Invert: near objects are bright (high char density), far are dark (sparse)
        max_depth = depth_resized.max()
        if max_depth > 0:
            inverted = 255 - (depth_resized * 255 / max_depth).astype(np.uint8)
        else:
            inverted = np.zeros_like(depth_resized)
        return inverted

    def _build_entity_map(self, labels_buffer, labels):
        """Build a downscaled entity category map from VizDoom labels.

        Args:
            labels_buffer: ``(H, W)`` uint8 array of VizDoom label IDs.
            labels: List of VizDoom ``Label`` objects providing the mapping
                from label IDs to object names.

        Returns:
            ``numpy.ndarray`` of shape ``(height, width)`` where each cell
            contains an entity category ID (0 = background).
        """
        # Map VizDoom label IDs to our entity categories
        label_id_to_entity = {}
        for label in labels:
            name = label.object_name if hasattr(label, 'object_name') else str(label)
            entity_cat = LABEL_TO_ENTITY.get(name, 0)
            if hasattr(label, 'value'):
                label_id_to_entity[label.value] = entity_cat

        # Create category map at original resolution
        h, w = labels_buffer.shape
        category_map = np.zeros_like(labels_buffer, dtype=np.uint8)
        for label_id, entity_cat in label_id_to_entity.items():
            category_map[labels_buffer == label_id] = entity_cat

        # Downscale with max (entity presence wins over background)
        bh = h // self.height
        bw = w // self.width
        cropped = category_map[:bh * self.height, :bw * self.width]
        entity_resized = cropped.reshape(self.height, bh, self.width, bw).max(axis=(1, 3))
        return entity_resized

    def _to_ascii(self, brightness, entity_map=None):
        """Convert a brightness array and optional entity map to an ASCII string.

        Args:
            brightness: ``(height, width)`` uint8 array of brightness values.
            entity_map: Optional ``(height, width)`` array of entity category
                IDs.  Non-zero values override the brightness character.

        Returns:
            Multi-line ASCII string with ``'\\n'`` row separators.
        """
        rows = []
        for y in range(self.height):
            row_chars = []
            for x in range(self.width):
                # Entity labels override brightness
                if entity_map is not None and entity_map[y, x] > 0:
                    row_chars.append(ENTITY_CHARS.get(entity_map[y, x], '?'))
                else:
                    # Map brightness (0-255) to ASCII character
                    level = int(brightness[y, x] / 256 * self.n_levels)
                    level = min(level, self.n_levels - 1)
                    row_chars.append(self.brightness_chars[level])
            rows.append(''.join(row_chars))
        return '\n'.join(rows)

    def convert_simple(self, screen_buffer):
        """Convert a screen buffer to ASCII using brightness only.

        This is a simplified conversion path that does not require depth or
        label buffers, making it useful for testing without a full VizDoom
        setup or when processing pre-recorded JPEG frames.

        Args:
            screen_buffer: ``(H, W)`` grayscale or ``(H, W, 3)`` RGB uint8
                array.

        Returns:
            A multi-line ASCII string of size ``height`` rows by ``width``
            columns with ``'\\n'`` as the row separator.

        Example:
            >>> import numpy as np
            >>> converter = AsciiConverter(width=10, height=5)
            >>> fake_screen = np.random.randint(0, 256, (120, 160), dtype=np.uint8)
            >>> print(converter.convert_simple(fake_screen))
        """
        if screen_buffer.ndim == 3:
            gray = np.mean(screen_buffer, axis=2).astype(np.uint8)
        else:
            gray = screen_buffer
        gray_resized = self._downscale(gray, self.height, self.width)
        return self._to_ascii(gray_resized)

    def convert_with_depth(self, screen_buffer, depth_buffer, num_bins=16,
                           labels_buffer=None, labels=None):
        """Convert frame to ASCII text + quantized depth bin sequence.

        Returns both the ASCII string and a list of depth bin IDs (one per
        character in the ASCII string, including newlines). Depth bins range
        from 0 (very near) to num_bins-1 (very far). Special positions
        (newlines) get the "no depth" bin ID = num_bins.

        Args:
            screen_buffer: (H, W) or (H, W, 3) screen frame.
            depth_buffer: (H, W) float depth buffer from VizDoom.
            num_bins: Number of depth quantization levels.
            labels_buffer: Optional (H, W) labels from VizDoom.
            labels: Optional list of VizDoom Label objects.

        Returns:
            Tuple of (ascii_string, depth_bins_list) where depth_bins_list
            has the same length as ascii_string.
        """
        # Get ASCII
        if screen_buffer.ndim == 3:
            gray = np.mean(screen_buffer, axis=2).astype(np.uint8)
        else:
            gray = screen_buffer

        gray_resized = self._downscale(gray, self.height, self.width)

        entity_map = None
        if labels_buffer is not None and labels is not None:
            entity_map = self._build_entity_map(labels_buffer, labels)

        if depth_buffer is not None:
            brightness = self._depth_to_brightness(depth_buffer)
        else:
            brightness = gray_resized

        ascii_text = self._to_ascii(brightness, entity_map)

        # Quantize depth to bins
        depth_resized = self._downscale(
            depth_buffer.astype(np.float32) if depth_buffer.dtype != np.float32 else depth_buffer,
            self.height, self.width
        ).astype(np.float32)

        # Normalize depth to [0, 1] then quantize
        d_min = depth_resized.min()
        d_max = depth_resized.max()
        if d_max > d_min:
            depth_norm = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_resized)

        depth_quantized = np.clip(
            (depth_norm * num_bins).astype(int), 0, num_bins - 1
        )

        # Build depth_bins list matching the ASCII string
        # ASCII: row0_chars \n row1_chars \n ... (no trailing newline)
        no_depth = num_bins  # special "no depth" bin for newlines
        depth_bins = []
        for y in range(self.height):
            for x in range(self.width):
                depth_bins.append(int(depth_quantized[y, x]))
            if y < self.height - 1:
                depth_bins.append(no_depth)  # newline separator

        return ascii_text, depth_bins
