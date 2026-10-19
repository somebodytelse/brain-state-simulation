"""Source-checkout compatibility only; scientific implementations live in src/."""
from importlib import import_module
from pathlib import Path
import runpy
import sys


def _source_path() -> None:
    source = str(Path(__file__).resolve().parent / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)


def expose(name: str, implementation: str) -> None:
    """Alias imports to the real module so monkeypatching and legacy imports agree."""
    _source_path()
    module = import_module(implementation)
    if name == '__main__':
        entry = getattr(module, 'main', None)
        if entry is not None:
            entry()
    else:
        sys.modules[name] = module


def run_legacy(module: str) -> None:
    """Execute the historical exploratory script only when explicitly requested."""
    _source_path()
    runpy.run_module(module, run_name='__main__')
