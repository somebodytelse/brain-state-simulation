"""Legacy clustering experiment; not a regression test. Requires the legacy extra."""
from _compat import run_legacy

if __name__ == "__main__":
    run_legacy("brain_states.legacy.three_state_clustering")
