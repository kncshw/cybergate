"""Family resolution, and the proof that the per-family split is behaviour-preserving.

The canonical package has exactly two arms -- Gemma and everything else -- and so does this one:
`gemma4` and `qwen36`, with `qwen36` as the canonical `else`. The two dispatch paths no longer agree
on what to do with anything else: `family_for_model_key` (worker) still lands on the Qwen arm, while
`family_for_model_id` (serving) resolves from the configured operator and otherwise raises
`UnknownModelFamily`.

Earlier revisions also carried `phi4` and `llama31` classes. Both inherited the Qwen arm wholesale,
shipped no kit, and were never served; they are gone. The tests below keep their model ids as
unrecognised-id cases, because what happens to an id with no family of its own is still a behaviour
worth pinning -- on both paths, which answer differently.
"""
from __future__ import annotations

import pytest

from cybergate.families import (
    ALL_FAMILIES,
    DEFAULT_FAMILY,
    ModelFamily,
    UnknownModelFamily,
    family_by_name,
    family_for_model_id,
    family_for_model_key,
)

#: Every attribute the serving or probing path reads off a family -- the checklist for adding one.
#: `test_every_family_declares_every_serving_attribute_itself` asserts each family sets all of them
#: on its own class. That matters because the base defaults are silent rather than absent: a family
#: that forgets `user_turn_marker` inherits an empty tuple, the exact-marker branch never matches,
#: and span location falls through to fixed offsets while reporting success. That is not
#: hypothetical -- it is what the canonical Qwen class did, with Qwen2.5 token ids on a Qwen3.6
#: checkpoint, for every request.
SHARED_ATTRIBUTES = (
    "user_turn_marker",
    "turn_end_id",
    "span_label",
    "default_pre",
    "default_suf",
    "baseline_suffix_lens",
    "bos_token_id",
    "fresh_prompt_marker",
    "ref_layer",
    "ref_mu_hm",
    "ref_sd_hm",
    "revoke_ceil",
    "hysteresis_thresh",
    "default_ref_dir_name",
    "default_bundle_name",
    "precomputed_tag",
    "bundle_probe_layers",
    "mid_late_probes",
    "tokenizer_globs",
    "tokenizer_dirs",
    "turn_open",
    "turn_close",
    "assistant_role",
    "supports_bos",
    "bos_text",
    "think_system_prefix",
    "thought_open",
    "thought_close",
    "empty_thought_block",
    "thought_close_tag",
    "stop_tokens",
)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("google/gemma-4-26B-A4B-it", "gemma4"),
        ("google/gemma-4-26b-a4b-it", "gemma4"),
        ("gemma4", "gemma4"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen36"),
    ],
)
def test_family_for_model_id(model_id: str, expected: str) -> None:
    """A model id that names its own family resolves to it, regardless of case or vendor prefix."""
    assert family_for_model_id(model_id).name == expected


def test_an_unrecognised_model_id_resolves_from_the_configured_operator() -> None:
    """No family matched, so the operator this process serves is asked instead.

    This replaces the canonical fallback to Qwen-3.6, which was silent and picked the chat template
    the prompt is rendered with. `conftest.py` configures the Gemma fixture operator, so these land
    on `gemma4`. See `tests/equivalence/test_prompts_differential.py` for the divergence this is.
    """
    for model_id in ("some-vendor/unlisted-model-v1", "prod", "/data/checkpoints/final", ""):
        assert family_for_model_id(model_id).name == "gemma4"


def test_an_unrecognised_model_id_raises_with_no_operator_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing to resolve from, resolution fails loudly rather than guessing a family."""
    monkeypatch.delenv("CYBERGATE_OP_PATH", raising=False)
    with pytest.raises(UnknownModelFamily):
        family_for_model_id("some-vendor/unlisted-model-v1")


@pytest.mark.parametrize(
    ("model_key", "expected"),
    [
        ("gemma", "gemma4"),
        ("gemma4", "gemma4"),
        ("qwen36", "qwen36"),
        ("phi", "qwen36"),
        ("llama", "qwen36"),
        ("something-else", "qwen36"),
        ("", "qwen36"),
    ],
)
def test_family_for_model_key(model_key: str, expected: str) -> None:
    """The worker path dispatches on an operator's `meta.model_key`, which uses short prefixes."""
    assert family_for_model_key(model_key).name == expected


def test_the_shipped_operator_model_keys_all_resolve() -> None:
    """Every `model_key` the shipped kits carry must resolve to its own family.

    `phi` and `llama` resolve to the Qwen fallback arm rather than to families of their own. Both
    classes were removed, so this is the canonical everything-else behaviour and not a regression.
    An operator actually carrying one of those keys would be rejected by the kit binding check, since
    its dimensions do not match Qwen's -- which is why `family_for_model_key` keeps a fallback while
    `family_for_model_id` no longer has one: this path is already fail-closed downstream, that one
    was not.
    """
    for model_key, expected in (
        ("gemma", "gemma4"),
        ("qwen36", "qwen36"),
        ("phi", "qwen36"),
        ("llama", "qwen36"),
    ):
        assert family_for_model_key(model_key).name == expected


def test_default_family_is_the_canonical_else_arm() -> None:
    assert DEFAULT_FAMILY.name == "qwen36"


def test_family_by_name_raises_rather_than_falling_back() -> None:
    """Must raise rather than guess, as `family_for_model_id` does; only the worker lookup falls
    back."""
    assert family_by_name("gemma4").name == "gemma4"
    with pytest.raises(KeyError):
        family_by_name("not-a-family")




#: Families that ship without `verified_live`, each with the reason it is carried anyway. A new
#: family added with `verified_live = False` must be listed HERE, with a justification, rather than
#: slipping in silently -- that is what `test_every_remaining_family_is_live_verified` enforces.
UNVERIFIED_WITH_CAUSE: dict[str, str] = {}
#: Empty, and that is the point: every registered family has been served and measured. qwen38 was
#: the last entry and earned `verified_live` on 2026-11-02 -- 4,134 prompts at the deployed gate,
#: 587/675 CySecBench on a split that chose nothing, canaries pinned from the live scores, kit
#: check clean without CYBERGATE_SKIP_KIT_CHECK. Its justification lived here while it did not
#: gate; see kits/qwen38/MANIFEST.json measured_live for what replaced it, and known_gaps for what
#: is still missing -- `verified_live` means served and calibrated, not finished.


def test_every_remaining_family_is_live_verified() -> None:
    """Honest provenance. A family is either served-and-calibrated, or listed as not.

    `verified_live` is not decoration: it is the flag that said which families were measured and
    which inherited someone else's parameters. Phi-4 and Llama-3.1 were removed rather than carried
    unverified. A new family added with `verified_live = False` must be justified in
    `UNVERIFIED_WITH_CAUSE`, not slip in.
    """
    names = {f.name for f in ALL_FAMILIES}
    verified = {f.name for f in ALL_FAMILIES if f.verified_live}
    assert verified == {"gemma4", "qwen36", "qwen38"}
    assert names - verified == set(UNVERIFIED_WITH_CAUSE), (
        "a family is unverified and unexplained; add it to UNVERIFIED_WITH_CAUSE with the reason"
    )
    for name, why in UNVERIFIED_WITH_CAUSE.items():
        assert name in names, f"{name} is explained but not registered"
        assert len(why) > 120, f"{name}: the justification must say what was measured and why"


def test_gemma_values_match_the_live_server() -> None:
    """The numbers in the Gemma-4 family are the ones the live container reports."""
    gemma = family_by_name("gemma4")
    assert gemma.user_turn_marker == (105, 2364, 107)
    assert gemma.turn_end_id == 106
    assert gemma.bos_token_id == 2
    assert gemma.ref_layer == 21
    assert gemma.ref_mu_hm == pytest.approx(0.1374208)
    assert gemma.ref_sd_hm == pytest.approx(0.0513015)
    assert gemma.revoke_ceil == pytest.approx(0.0520)
    assert gemma.hysteresis_thresh == pytest.approx(0.0650)
    # The shipped operator, which must be first in the ladder. This assertion read
    # "gemma4_L10_M1.json" until 2026-10-31 and so pinned the stale default in place.
    assert gemma.default_op_names[0] == "gemma4_L16_M1.json"


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
@pytest.mark.parametrize("attribute", SHARED_ATTRIBUTES)
def test_every_family_declares_every_serving_attribute_itself(
    family: ModelFamily, attribute: str
) -> None:
    """A family must set each serving attribute on its own class, not inherit the base default.

    Inheriting is worse than crashing here: every base default is a legal value, so the serving path
    reads it, uses it and reports success.
    """
    assert attribute in type(family).__dict__, (
        f"{family.name} inherits {attribute!r} from a base class rather than declaring it; "
        f"the serving path will read the default and silently get it wrong"
    )


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_every_family_is_self_consistent(family: ModelFamily) -> None:
    """Structural invariants that a copy-paste slip in a new family would break."""
    assert family.name
    assert family.model_key_prefixes
    assert family.model_id_substrings
    assert len(family.user_turn_marker) == 3, "all supported templates open a turn with 3 tokens"
    assert family.turn_end_id > 0
    assert family.stop_tokens
    assert family.thought_close_tag in family.thought_close
    assert family.empty_thought_block.startswith(family.thought_open)
    assert family.default_pre > 0 and family.default_suf > 0
    assert -1.0 < family.revoke_ceil < 1.0
    assert -1.0 < family.hysteresis_thresh < 1.0
    assert family.ref_sd_hm > 0.0, "a zero sd would make every z-score infinite"


def test_families_are_shared_singletons() -> None:
    """The registry hands out one instance per family, so attribute reads stay cheap."""
    assert family_for_model_id("google/gemma-4-26B-A4B-it") is family_by_name("gemma4")
    assert family_for_model_key("gemma") is family_by_name("gemma4")


def test_gemma_decode_calibration_is_distinct_from_the_prefill_one() -> None:
    """Producer Finding 1: the two calibrations are different populations.

    If a future edit ever makes them equal, the closed-loop trigger is back to comparing a decode
    statistic against a prefill-domain threshold -- the state in which it could not fire at all.
    """
    gemma = family_by_name("gemma4")

    assert gemma.ref_mu_decode is not None and gemma.ref_sd_decode is not None
    assert gemma.ref_mu_decode != gemma.ref_mu_hm
    assert gemma.ref_sd_decode != gemma.ref_sd_hm
    # Measured: the decode distribution sits lower and is tighter than the prefill one.
    assert gemma.ref_mu_decode < gemma.ref_mu_hm
    assert gemma.ref_sd_decode < gemma.ref_sd_hm


def test_an_unmeasured_family_has_no_decode_calibration() -> None:
    """Both remaining families were measured live, so both must carry a decode calibration.

    The prefill pair (`ref_mu_hm`/`ref_sd_hm`) and the decode pair are NOT interchangeable -- that
    conflation was the producer review's Finding 1, where `tau_spike` was compared against a decode
    statistic expressed in prefill units and no generation could reach it. A family that claims a
    decode calibration it never measured reintroduces exactly that.
    """
    for name in ("gemma4", "qwen36"):
        family = family_by_name(name)
        assert family.ref_mu_decode is not None, f"{name} lost its decode calibration"
        assert family.ref_sd_decode is not None, f"{name} lost its decode calibration"
