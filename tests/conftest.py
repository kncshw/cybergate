"""Test configuration.

Both the new package and the vendored baseline read `CYBERGATE_IPC_DIR`, `CYBERGATE_EVENTS_LOG_PATH`
and `CYBERGATE_OPS_DIR` at **import** time, so they are set here, before any test module is imported.
Without this the vendored baseline would write to the real `/dev/shm/cybergate_ipc` and to the real
plugin event log while the suite runs.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_BASELINE = _ROOT / "tests" / "equivalence" / "baseline"

# Point both implementations at a throwaway IPC directory and log before anything imports them.
_SANDBOX = Path(tempfile.mkdtemp(prefix="cybergate-tests-"))
os.environ["CYBERGATE_IPC_DIR"] = str(_SANDBOX / "ipc")
os.environ["CYBERGATE_EVENTS_LOG_PATH"] = str(_SANDBOX / "events.log")
os.environ["CYBERGATE_OPS_DIR"] = str(_ROOT / "tests" / "fixtures")
os.environ["CYBERGATE_OP_PATH"] = str(_ROOT / "tests" / "fixtures" / "gemma4_L10_M1.json")
os.environ["CYBERGATE_PROXY_PORT"] = "0"
# The fixture operators are deliberately synthetic: 4-dimensional vectors at L10/L15, chosen so the
# policy and loader can be tested without shipping a 2816-dim direction into the repo. They
# therefore contradict their families' kit declarations on purpose, and the binding check has to
# stand down for the suite to run at all. `tests/unit/test_kit_binding.py` clears this flag with
# monkeypatch and asserts the enforcing path, so the check itself is still covered.
os.environ["CYBERGATE_SKIP_KIT_CHECK"] = "1"
# Never let a test reach an external gate endpoint.
os.environ.pop("CYBERGATE_GATE_URL", None)

for path in (_SRC, _BASELINE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

IPC_DIR = Path(os.environ["CYBERGATE_IPC_DIR"])
FIXTURES = _ROOT / "tests" / "fixtures"
GEMMA_OP = FIXTURES / "gemma4_L10_M1.json"
QWEN_OP = FIXTURES / "qwen36_L15_M1.json"


@pytest.fixture
def ipc_dir() -> Path:
    """The sandbox IPC directory, emptied before each test."""
    IPC_DIR.mkdir(parents=True, exist_ok=True)
    for child in IPC_DIR.iterdir():
        if child.is_file():
            child.unlink()
    return IPC_DIR


@pytest.fixture
def clean_process_state():
    """Clear every in-process cache both implementations keep, before and after a test.

    Without this, a decision cached under one test's turn text leaks into the next test: an earlier 25-turn live
    test reported a run-to-run sd of exactly 0.000000 at 22 of 25 turn indices, which looked like
    stability and was actually result reuse.
    """
    import baseline.gate_probe as legacy

    from cybergate.state import request_state

    def _clear() -> None:
        legacy.LIVE_GPU_PROMPT_CACHE.clear()
        legacy.REQUEST_GATE_STATES.clear()
        request_state.LIVE_GPU_PROMPT_CACHE.clear()
        request_state.REQUEST_GATE_STATES.clear()

    _clear()
    yield
    _clear()
