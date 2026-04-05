"""Character set definitions for DOOM ASCII art.

This module defines the character palettes used to convert VizDoom frames into
ASCII text:

* **Brightness characters** -- A 10-character ramp from space (darkest) to
  ``@`` (brightest), used to represent background geometry based on pixel
  brightness or inverse depth.
* **Entity characters** -- Single-letter markers for game objects detected
  via VizDoom's labels buffer (e.g. ``E`` for enemies, ``H`` for health
  items).  These override brightness characters in the final ASCII output.
* **Label-to-entity mapping** -- Maps VizDoom object names (e.g.
  ``'Zombieman'``, ``'Medikit'``) to entity category IDs used as keys in
  :data:`ENTITY_CHARS`.

Design rationale: the character set is deliberately small so that a
character-level tokenizer can cover every possible ASCII frame token with
fewer than 100 vocabulary entries, keeping the embedding table tiny.
"""

# Brightness levels (10 chars, dark to bright)
BRIGHTNESS_CHARS = " .:-=+*#%@"

# Entity markers (from VizDoom labels buffer)
ENTITY_CHARS = {
    0: ' ',    # nothing/background
    1: 'E',    # enemy (DoomPlayer, Zombieman, ShotgunGuy, etc.)
    2: 'H',    # health items
    3: 'A',    # ammo items
    4: 'D',    # door
    5: 'W',    # wall/obstacle
    6: 'K',    # key items
    7: 'X',    # explosion/projectile
}

# VizDoom label name to entity category mapping
LABEL_TO_ENTITY = {
    'DoomPlayer': 1, 'Zombieman': 1, 'ShotgunGuy': 1, 'ChaingunGuy': 1,
    'DoomImp': 1, 'Demon': 1, 'Spectre': 1, 'LostSoul': 1,
    'Cacodemon': 1, 'BaronOfHell': 1, 'HellKnight': 1, 'Cyberdemon': 1,
    'Revenant': 1, 'Arachnotron': 1, 'PainElemental': 1, 'Archvile': 1,
    'Medikit': 2, 'Stimpack': 2, 'HealthBonus': 2, 'Soulsphere': 2,
    'Clip': 3, 'Shell': 3, 'Cell': 3, 'RocketAmmo': 3, 'Backpack': 3,
    'BlueCard': 6, 'RedCard': 6, 'YellowCard': 6,
    'BlueSkull': 6, 'RedSkull': 6, 'YellowSkull': 6,
    'BFG9000': 3, 'Shotgun': 3, 'Chaingun': 3, 'RocketLauncher': 3,
    'PlasmaRifle': 3, 'Chainsaw': 3, 'SuperShotgun': 3,
}

# Default ASCII resolution
DEFAULT_WIDTH = 40
DEFAULT_HEIGHT = 25
