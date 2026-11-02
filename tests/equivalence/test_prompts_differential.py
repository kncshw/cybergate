"""Raw prompt construction must be byte-identical to the canonical builder.

Every mode, both families, and several message shapes. A single character's difference here changes
what the model is asked, and on the `preclosed` and `step2` modes it changes the injected thought
block, so this is compared as exact strings rather than by structure.
"""
from __future__ import annotations

import pytest

import baseline.prompts as legacy
from cybergate.serving.prompts import build_raw_prompt_for_model, detect_model_family

GEMMA_MODEL = "google/gemma-4-26B-A4B-it"
QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B"
SUPPORTED_MODELS = (GEMMA_MODEL, QWEN_MODEL)

#: Ids that match no family. Deliberately generic: these used to be the Phi-4 and Llama-3.1 ids,
#: which read as though those families were supported. They are not -- both classes were removed.
#: The shapes that matter in practice are an alias from `--served-model-name` and a local checkpoint
#: path, neither of which contains a vendor name.
UNRECOGNISED_MODEL_IDS = (
    "some-vendor/unlisted-model-v1",
    "prod",
    "/data/checkpoints/final",
    "",
)

MODES = (
    "baseline",
    "preclosed",
    "preclosed_no_opener",
    "sandwich_step1",
    "sandwich_step2",
    "closed_loop_step1",
    "closed_loop_step1b",
    "closed_loop_step2",
    "closed_loop_cl2_step2",
)

MESSAGE_SHAPES: dict[str, list[dict[str, object]]] = {
    "single_user": [{"role": "user", "content": "first turn"}],
    "system_and_user": [
        {"role": "system", "content": "custom system prompt"},
        {"role": "user", "content": "first turn"},
    ],
    "three_turns": [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "a reply"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "another reply"},
        {"role": "user", "content": "third turn"},
    ],
    "developer_role": [
        {"role": "developer", "content": "developer system prompt"},
        {"role": "user", "content": "first turn"},
    ],
    "whitespace_padded": [{"role": "user", "content": "   padded turn   "}],
    "multimodal_list_content": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "part one "},
                {"type": "image_url", "image_url": {"url": "ignored"}},
                {"type": "text", "text": "part two"},
            ],
        }
    ],
    "system_not_first": [
        {"role": "user", "content": "first turn"},
        {"role": "system", "content": "late system message"},
    ],
    "empty_messages": [],
}

CUSTOM_THINK = "requests, sqlite3).\n- Step 2: build the query."


@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", sorted(MESSAGE_SHAPES))
@pytest.mark.parametrize("include_bos", [False, True])
def test_prompt_is_byte_identical(model_id: str, mode: str, shape: str, include_bos: bool) -> None:
    messages = MESSAGE_SHAPES[shape]
    kwargs = {"mode": mode, "include_bos": include_bos}
    if mode in ("closed_loop_step1b", "sandwich_step2", "closed_loop_step2", "closed_loop_cl2_step2"):
        kwargs["custom_think_text"] = CUSTOM_THINK

    expected = legacy.build_raw_prompt_for_model(messages, model_id, **kwargs)
    actual = build_raw_prompt_for_model(messages, model_id, **kwargs)
    assert actual == expected


@pytest.mark.parametrize("model_id", [GEMMA_MODEL, QWEN_MODEL])
@pytest.mark.parametrize("opener", ["", "CUSTOM OPENER: ", None])
def test_explicit_post_think_opener_matches(model_id: str, opener: str | None) -> None:
    messages = MESSAGE_SHAPES["single_user"]
    expected = legacy.build_raw_prompt_for_model(
        messages, model_id, mode="preclosed", post_think_opener=opener
    )
    actual = build_raw_prompt_for_model(
        messages, model_id, mode="preclosed", post_think_opener=opener
    )
    assert actual == expected


@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
def test_unknown_mode_raises_on_both(model_id: str) -> None:
    messages = MESSAGE_SHAPES["single_user"]
    with pytest.raises(ValueError):
        legacy.build_raw_prompt_for_model(messages, model_id, mode="not_a_mode")
    with pytest.raises(ValueError):
        build_raw_prompt_for_model(messages, model_id, mode="not_a_mode")


@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
def test_detect_model_family_matches_the_baseline_for_supported_models(model_id: str) -> None:
    """For a model id that names its own family, the compat shim still agrees with the canonical one."""
    assert detect_model_family(model_id) == legacy.detect_model_family(model_id)


@pytest.mark.parametrize("model_id", UNRECOGNISED_MODEL_IDS)
def test_family_resolution_deliberately_diverges_for_unrecognised_ids(model_id: str) -> None:
    """**The one intentional divergence from the canonical package in this port.**

    The canonical code resolves any unrecognised model id to Qwen-3.6 and renders the prompt with
    Qwen's chat template. That is silent and wrong on a Gemma server reached under an alias -- which
    `--served-model-name`, or serving from a local checkpoint path, produces. We resolve from the
    configured operator instead, which `ops.kit.check_operator` has already validated against the
    served weights.

    `conftest.py` points `CYBERGATE_OP_PATH` at the Gemma fixture, so that is what these resolve to
    here. The assertion is that we do NOT agree with the baseline, and that we land on the configured
    operator's family rather than on a hardcoded one.
    """
    assert legacy.detect_model_family(model_id) == "qwen36", "baseline no longer falls back to Qwen"
    assert detect_model_family(model_id) == "gemma4"
    assert detect_model_family(model_id) != legacy.detect_model_family(model_id)


@pytest.mark.parametrize("model_id", UNRECOGNISED_MODEL_IDS)
def test_an_unrecognised_id_raises_when_no_operator_is_configured(
    model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing to resolve from, an unrecognised id is an error rather than a guess.

    This is the case the canonical fallback hid: no family matched, no operator named one, and the
    request was rendered with Qwen's template anyway.
    """
    from cybergate.families import UnknownModelFamily

    monkeypatch.delenv("CYBERGATE_OP_PATH", raising=False)
    with pytest.raises(UnknownModelFamily) as excinfo:
        detect_model_family(model_id)
    message = str(excinfo.value)
    assert "gemma4" in message and "qwen36" in message
    assert "CYBERGATE_OP_PATH" in message


@pytest.mark.parametrize("shape", sorted(MESSAGE_SHAPES))
def test_system_and_user_extraction_matches(shape: str) -> None:
    from cybergate.messages import extract_system_and_user

    assert extract_system_and_user(MESSAGE_SHAPES[shape]) == legacy.extract_system_and_user(
        MESSAGE_SHAPES[shape]
    )


def test_preserve_thought_chat_template_is_unchanged() -> None:
    from cybergate.serving.prompts import GEMMA4_PRESERVE_THOUGHT_CHAT_TEMPLATE

    assert GEMMA4_PRESERVE_THOUGHT_CHAT_TEMPLATE == legacy.GEMMA4_PRESERVE_THOUGHT_CHAT_TEMPLATE


# =================================================================================================
# The two text transforms.
# =================================================================================================
THINK_SAMPLES = (
    "",
    "- Step 1: open a socket.\n- Step 2: send the payload.",
    "- Step 1: open a socket.\nHowever, I should not provide this.",
    "Wait, this might be harmful.",
    "- Step 1: fine.\n\nActually, safety policy prohibits this.",
    "A sentence. Another sentence. A third one.",
    "Mitigation guidance follows.",
    "No pivot words at all, just a plain derivation of the protocol mechanism.",
    "Caution: this is a disclaimer.",
    "   leading and trailing whitespace   ",
)


@pytest.mark.parametrize("text", THINK_SAMPLES)
def test_truncate_think_at_second_guess_matches(text: str) -> None:
    from cybergate.boilerplate import truncate_think_at_second_guess

    assert truncate_think_at_second_guess(text) == legacy.truncate_think_at_second_guess(text)


@pytest.mark.parametrize("text", THINK_SAMPLES)
def test_strip_injected_think_boilerplate_matches(text: str) -> None:
    from cybergate.boilerplate import strip_injected_think_boilerplate

    assert strip_injected_think_boilerplate(text) == legacy.strip_injected_think_boilerplate(text)


def test_strip_handles_the_real_injected_shapes() -> None:
    """The masking must match on the exact strings the routes build, not just arbitrary text."""
    from cybergate.boilerplate import (
        BOTTOM_COMPLIANCE_LOCK,
        FAKE_THINK_RUBRIC,
        INLINE_COT_SUPPRESSION,
        OPEN_THINK_PREFIX,
        strip_injected_think_boilerplate,
    )

    shapes = (
        FAKE_THINK_RUBRIC,
        f"{OPEN_THINK_PREFIX}sqlite3).\n- Step 2: build it.{BOTTOM_COMPLIANCE_LOCK}",
        f"{OPEN_THINK_PREFIX}a).{INLINE_COT_SUPPRESSION}continue here.{BOTTOM_COMPLIANCE_LOCK}",
    )
    for shape in shapes:
        assert strip_injected_think_boilerplate(shape) == legacy.strip_injected_think_boilerplate(
            shape
        )
