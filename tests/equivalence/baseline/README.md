# Vendored baseline — DO NOT EDIT

These five files are a copy of the **canonical** pre-refactor `cybergate_plugin` package taken on
2026-10-27. A separate local copy of the same package was verified sha256-identical to
the canonical one for all 5 modules at copy time.

## Sanitised for public release

In this public copy, two of the five files differ from the canonical original by **exactly two
string literals**, replaced to keep an internal host out of a public repository:

| file | line | canonical | here |
|---|---|---|---|
| `gate_probe.py` | 425 | a `/home/<user>/gemma4_restore` tokenizer fallback path | placeholder path in `gate_probe.py` |
| `middleware.py` | 57 | the default `base_url`, an internal address | `https://192.0.2.39/v1` |

Both are inert for the equivalence suite: the tokenizer fallback is never reached (the tests do not
load a tokenizer) and `base_url` is overridden by every test that constructs a client. No logic,
no control flow and no line count changed -- `gate_probe.py` is still 1499 lines and
`middleware.py` still 1048, and the other three files are untouched, with their original hashes.
`BASELINE_PIN.sha256` was regenerated over the sanitised content.

`BASELINE_PIN.sha256` pins the exact content. `tests/equivalence/test_baseline_pin.py` re-verifies it
on every run, so if anyone re-copies or edits these files the suite fails loudly rather than silently
re-parenting the equivalence proof.

Line counts: `gate_probe.py` 1499, `middleware.py` 1048.

## Why these are here

The refactor's correctness claim is *behaviour preservation*. These files are the reference
implementation that `tests/equivalence/test_differential.py` drives in lockstep with the new
`cybergate.decision.policy.decide()`, asserting exact equality on every decision field.

They are imported by tests only. Nothing under `src/cybergate/` imports them, and
`tests/unit/test_purity.py` enforces that.

## Known non-reproducible behaviour

`gate_probe.py` reads module-level env vars at import time (`CYBERGATE_IPC_DIR`,
`CYBERGATE_EVENTS_LOG_PATH`). The differential test sets both to a tmp dir **before** importing, so
the vendored baseline never touches `/dev/shm` or the real log file.
