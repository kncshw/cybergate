"""The vendored baseline must be byte-identical to what was pinned.

If this fails, someone re-copied or edited `tests/equivalence/baseline/`, and every equivalence
result in the suite is now measured against a different reference than the one the refactor was
verified against.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "baseline"
PIN = BASELINE / "BASELINE_PIN.sha256"

#: Expected line counts for the canonical files.
EXPECTED_LINES = {"gate_probe.py": 1499, "middleware.py": 1048}


def test_pinned_hashes_match() -> None:
    expected = {}
    for line in PIN.read_text().splitlines():
        digest, name = line.split(None, 1)
        expected[name.strip()] = digest

    assert expected, "BASELINE_PIN.sha256 is empty"
    for name, digest in expected.items():
        actual = hashlib.sha256((BASELINE / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} changed since it was pinned"


def test_line_counts_match_the_plan() -> None:
    for name, count in EXPECTED_LINES.items():
        actual = len((BASELINE / name).read_text().splitlines())
        assert actual == count, f"{name}: expected {count} lines, found {actual}"
