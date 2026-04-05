"""Pre-configured VizDoom scenarios for data collection and testing."""

SCENARIOS = {
    'basic': {
        'description': 'Single stationary enemy, shoot to score',
        'actions': ['shoot', 'turn_left', 'turn_right'],
        'difficulty': 'easy',
        'good_for': 'Initial model testing, basic shoot recognition',
    },
    'defend_the_center': {
        'description': 'Enemies approach from all directions, 360° defense',
        'actions': ['shoot', 'turn_left', 'turn_right'],
        'difficulty': 'medium',
        'good_for': 'Turn + shoot coordination',
    },
    'health_gathering': {
        'description': 'Navigate maze to collect health packs',
        'actions': ['move_forward', 'turn_left', 'turn_right'],
        'difficulty': 'medium',
        'good_for': 'Navigation, movement decisions',
    },
    'deadly_corridor': {
        'description': 'Navigate corridor with enemies, reach goal',
        'actions': ['move_forward', 'move_backward', 'turn_left', 'turn_right',
                    'strafe_left', 'strafe_right', 'shoot'],
        'difficulty': 'hard',
        'good_for': 'Full action space, navigation + combat',
    },
}
