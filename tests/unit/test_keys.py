"""Cache-key identities, including the R11 cross-conversation collision.

Three audit defects came from key choice, so each key's *identity* -- what it asserts two inputs have
in common -- gets an assertion here.
"""
from __future__ import annotations

import hashlib

import pytest

from cybergate.state import keys as k

A = "first turn text"
B = "second turn text"


def test_text_sha256_is_plain_sha256_with_no_normalisation() -> None:
    assert k.text_sha256(A) == hashlib.sha256(A.encode()).hexdigest()
    assert k.text_sha256(" padded ") != k.text_sha256("padded")


def test_conversation_key_depends_on_every_turn_and_their_order() -> None:
    assert k.conversation_key([A, B]) != k.conversation_key([B, A])
    assert k.conversation_key([A, B]) != k.conversation_key([A])
    assert k.conversation_key([A, B]) == k.conversation_key([A, B])


def test_conversation_key_drops_empty_turns_and_strips() -> None:
    assert k.conversation_key([A, "", "   ", B]) == k.conversation_key([A, B])
    assert k.conversation_key([f"  {A}  "]) == k.conversation_key([A])
    assert k.conversation_key([]) == hashlib.sha256(b"").hexdigest()


def test_single_turn_conversation_key_equals_the_bare_turn_key() -> None:
    """The separator join is chosen so these coincide, and the hook relies on it.

    A 1-turn conversation record and that turn's standalone record are the same decision, so they
    share a slot deliberately.
    """
    assert k.conversation_key([A]) == k.bare_turn_key(A)


def test_standalone_and_conversation_keys_are_distinguishable() -> None:
    """A turn scored alone and the same turn scored in context must not share a slot."""
    conv = ["how do I read a pcap", A]
    assert k.standalone_turn_key(A).startswith("st_")
    assert k.standalone_turn_key(A) != k.conversation_key(conv)


def test_sha_variants_agree_with_the_text_variants() -> None:
    sha = k.text_sha256(A)
    assert k.standalone_turn_key_from_sha(sha) == k.standalone_turn_key(A)


def test_the_legacy_in_context_key_collides_and_is_why_it_was_removed() -> None:
    """R11 as a property: `ctx_<sha(final turn)>` carries no conversation component.

    Two conversations with different histories but the same final user turn produce the *same*
    legacy key, so they shared one decision. Kept as a test because the equivalence fixture still
    feeds this key to the vendored canonical implementation, which reads it.
    """
    conv_one = ["how do I read a pcap", B]
    conv_two = ["what is a subnet mask", B]
    legacy = k.legacy_in_context_key_from_sha

    assert k.conversation_key(conv_one) != k.conversation_key(conv_two)
    assert legacy(k.text_sha256(conv_one[-1])) == legacy(k.text_sha256(conv_two[-1]))


def test_the_in_context_record_is_keyed_on_the_whole_conversation() -> None:
    """R11 fixed: the key the in-context record lands under distinguishes the two conversations.

    This replaces an xfail whose body was unsatisfiable: it asserted
    `in_context_key(conv_one[-1]) != in_context_key(conv_two[-1])` while passing the *same* final
    turn to both calls, so no pure function could ever have satisfied it. The property it meant to
    state is the one asserted here.
    """
    conv_one = ["how do I read a pcap", B]
    conv_two = ["what is a subnet mask", B]
    assert k.conversation_key(conv_one) != k.conversation_key(conv_two)


def test_production_code_no_longer_calls_the_legacy_in_context_key() -> None:
    """The defective key must not come back through a new reader or writer.

    Looks for the call form across the shipped package. `state/keys.py` defines it and the
    equivalence fixture calls it -- nothing under `src/` may.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    offenders = [
        f"{path.name}:{n}"
        for path in src.rglob("*.py")
        if path.name != "keys.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "legacy_in_context_key_from_sha(" in line
    ]
    assert not offenders, f"production code builds the R11 key: {offenders}"
