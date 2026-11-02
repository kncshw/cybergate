"""`cybergate`: selective cyber-domain unlocking as a vLLM plugin (`K = 0.0`, no weight edits).

Deliberately lazy. The plugin entry point runs in **both** the API-server process and the model-worker
process, so importing this package must not pull in torch -- the API server does not need it and pays
for it on every start. Everything except `register_cybergate_plugin` and `__version__` is imported
inside the function that needs it.

See `docs/PROCESS_TOPOLOGY.md` for which module runs in which process.
"""
from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__", "register_cybergate_plugin"]


def register_cybergate_plugin() -> None:
    """Entry point for `vllm.general_plugins`. Dispatches on which process invoked it."""
    from ._entrypoint import register_cybergate_plugin as _register

    _register()

