"""Backward-compatible entry point; implementation lives in brain_states.simulation."""
from _compat import expose

expose(__name__, 'brain_states.simulation')
