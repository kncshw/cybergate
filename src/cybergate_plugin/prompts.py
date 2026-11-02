"""Compatibility module for `from cybergate_plugin.prompts import ...`."""
from __future__ import annotations

from cybergate.serving.prompts import *  # noqa: F403 - re-export the whole public surface
from cybergate.serving.prompts import __all__ as __all__  # noqa: PLC0414
