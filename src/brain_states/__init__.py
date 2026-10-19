"""Exploratory overlapping brain-state research.

Array conventions:
    signals X: (channels, samples)
    channel_activity: (channels, samples), binary
    memberships W: (channels, states), binary
    state_activations H: (states, samples), binary
    adjacency: (channels, channels), weighted connectivity

Ground truth belongs to simulation/reference evaluation, never detector input.
"""
__version__ = "0.1.0"
