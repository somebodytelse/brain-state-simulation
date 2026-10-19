"""Portable project paths for source checkouts and installed command-line tools."""
from pathlib import Path

_checkout = Path(__file__).resolve().parents[2]
# Source checkouts use the repository; installed wheels use the working directory.
PROJECT_ROOT = _checkout if (_checkout / 'pyproject.toml').exists() else Path.cwd()
