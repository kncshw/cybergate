"""Every cache-key constructor, with the identity each key asserts.

This module exists because three separate defects in the audit came from key choice, and in the
canonical package the keys are built inline at six different call sites (`gate_probe.py:1287`,
`:1289`, `:1322`, `:1381`, `:676-684`, `:1009-1017`). Nothing outside this module may build a key.

Read the identity line before using a key. "Identity" means: two inputs that produce the same key are
being asserted to deserve the same gate decision.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

__all__ = [
    "CONV_TURN_SEPARATOR",
    "bare_turn_key",
    "conversation_key",
    "legacy_in_context_key_from_sha",
    "standalone_turn_key",
    "standalone_turn_key_from_sha",
    "text_sha256",
]

#: Separator joining user turns before hashing a conversation. Chosen so that a single-turn
#: conversation hashes identically to that turn's text alone (`"x".join([a]) == a`), which is what
#: makes `conversation_key([t]) == text_sha256(t)` hold.
CONV_TURN_SEPARATOR = "\n<|turn_sep|>\n"


def text_sha256(text: str) -> str:
    """SHA-256 of `text` exactly as given. No stripping, no normalisation.

    The canonical package hashes the *stripped* text when building a conversation key but the *raw*
    text when building a per-turn key (`hashlib.sha256(latest_user.encode())` at `gate_probe.py:1281`,
    where `latest_user` was already stripped during turn extraction). Both call sites pass
    already-stripped text, so the distinction is invisible today -- but it is why this function does
    not strip on your behalf.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def conversation_key(user_turns: Sequence[str]) -> str:
    """Identity: **the whole conversation**, as the ordered list of its user turns.

    Empty and whitespace-only turns are dropped, and each kept turn is stripped, before joining with
    `CONV_TURN_SEPARATOR`. An empty conversation hashes the empty string.

    This is the key the in-context record is stored under. Before the R11 fix it was only the
    *fallback*, behind a text-only `ctx_<sha(final turn)>` key that collided across conversations;
    see `legacy_in_context_key_from_sha`.
    """
    cleaned = [str(u).strip() for u in user_turns if str(u).strip()]
    if not cleaned:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(CONV_TURN_SEPARATOR.join(cleaned).encode("utf-8")).hexdigest()


def standalone_turn_key_from_sha(sha: str) -> str:
    """`standalone_turn_key` for a hash that has already been computed.

    The worker hook decodes token spans and hashes the decoded text itself, so it holds the hash and
    never the original request text. Same identity as `standalone_turn_key`.
    """
    return f"st_{sha}"


def standalone_turn_key(text: str) -> str:
    """Identity: **one turn's text, scored alone**, with no conversation around it.

    Written by the worker hook when it observes a 1-turn prefill, and read whenever the policy needs
    a standalone score for a turn (the Step 2A revocation test, and the prior-unlock scan).
    """
    return standalone_turn_key_from_sha(text_sha256(text))


def bare_turn_key(text: str) -> str:
    """Identity: same as `standalone_turn_key` -- a duplicate written under the unprefixed hash.

    The hook writes every standalone decision twice, once as `st_<sha>` and once as `<sha>`
    (`gate_probe.py:1013-1017`), and readers try the prefixed key first and fall back to this one.
    For a single-turn conversation `conversation_key([t]) == text_sha256(t) == this key`, so the
    unprefixed slot is also where a 1-turn conversation record lands -- the two writes collide by
    design and carry the same decision.

    Nothing requires this key that could not use `standalone_turn_key`. Phase 6 deletes it.
    """
    return text_sha256(text)


def legacy_in_context_key_from_sha(sha: str) -> str:
    """The canonical package's in-context key, `ctx_<sha(final user turn)>`. **R11. Do not use.**

    Kept for exactly one caller: the equivalence fixture, which must feed the vendored canonical
    implementation the keys *it* reads. Production code in this package no longer writes or reads
    this key -- `conversation_key` holds the in-context record instead.

    Why it had to go. The key carries *no conversation component*, yet it held the decision for a
    multi-turn in-context prefill, so two different conversations that happen to end in the same
    user text shared one decision and whichever was sent first won. It was also read *before* the
    correctly-keyed conversation record (`read(ctx_<latest>) or read(conv_sha)` at
    `gate_probe.py:1287`), so the collision took precedence over the correct lookup rather than
    acting as a fallback.

    Demonstrated live twice: an unlock transferred into a conversation that scored below both
    maintenance thresholds on its own merits, and during the 2026-10-28 producer review a 3-turn
    trajectory read its in-context score out of a record written by a *different* 2-turn request.
    """
    return f"ctx_{sha}"
