"""Prompt building, as the canonical module's public surface.

The family-independent boilerplate lives in `cybergate.boilerplate`; the per-architecture templates
live on each `ModelFamily`. This module is the compatibility facade that keeps
`build_raw_prompt_for_model(messages, model_id, ...)` and `detect_model_family(model_id)` working for
`cli.py`, the example scripts and anything importing from `cybergate_plugin.prompts`.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..boilerplate import (
    BOTTOM_COMPLIANCE_LOCK,
    DEFAULT_SYSTEM_PROMPT,
    FAKE_THINK_RUBRIC,
    FAKE_THINK_SHORT,
    INLINE_COT_SUPPRESSION,
    INLINE_FINAL_STEP_CLEARANCE,
    OPEN_THINK_PREFIX,
    OPEN_THINK_STEP1_STEM,
    POST_THINK_OPENER,
    SECOND_GUESS_RE,
    strip_injected_think_boilerplate,
    truncate_think_at_second_guess,
)
from ..families import family_for_model_id
from ..messages import extract_system_and_user

__all__ = [
    "BOTTOM_COMPLIANCE_LOCK",
    "DEFAULT_SYSTEM_PROMPT",
    "FAKE_THINK_RUBRIC",
    "FAKE_THINK_SHORT",
    "GEMMA4_PRESERVE_THOUGHT_CHAT_TEMPLATE",
    "INLINE_COT_SUPPRESSION",
    "INLINE_FINAL_STEP_CLEARANCE",
    "OPEN_THINK_PREFIX",
    "OPEN_THINK_STEP1_STEM",
    "POST_THINK_OPENER",
    "SECOND_GUESS_RE",
    "build_raw_prompt_for_model",
    "detect_model_family",
    "extract_system_and_user",
    "strip_injected_think_boilerplate",
    "truncate_think_at_second_guess",
]


def detect_model_family(model_id: str) -> str:
    """Return the resolved family's own `name`.

    **Deliberate divergence from the canonical package.** The canonical version was two-valued --
    `"gemma4" if "gemma" in model_id else "qwen36"` -- so a third family would report `"qwen36"` and
    be rendered with Qwen's chat template. This returns the real family name instead, and raises
    `UnknownModelFamily` for an id that resolves to nothing rather than guessing.

    Nothing inside this package branches on the returned string; it exists because the canonical
    surface exported it. New code should use `cybergate.families.family_for_model_id` and read
    attributes off the family object.
    """
    return family_for_model_id(model_id).name


def build_raw_prompt_for_model(
    messages: Sequence[dict[str, Any]],
    model_id: str,
    mode: str = "baseline",
    custom_think_text: str | None = None,
    post_think_opener: str | None = None,
    include_bos: bool = False,
) -> str:
    """Construct the exact raw prompt string for `/v1/completions`.

    Dispatches to the family's own template. Raises `ValueError` on an unknown `mode`.
    """
    return family_for_model_id(model_id).build_raw_prompt(
        messages,
        mode=mode,
        custom_think_text=custom_think_text,
        post_think_opener=post_think_opener,
        include_bos=include_bos,
    )


# Non-stripping Jinja2 chat template for Gemma-4 on `/v1/chat/completions` with
# `continue_final_message=True, add_generation_prompt=False`.
#
# Gemma-4's shipped `chat_template.jinja` defines a `strip_thinking(text)` macro that removes
# `<|channel>thought...<channel|>` from every assistant message, which would delete an injected
# thought block before the model ever sees it. This override omits that macro. It is only needed on
# the chat endpoint; the routes in `think_routes` use `/v1/completions` with a raw prompt and so are
# unaffected.
GEMMA4_PRESERVE_THOUGHT_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{%- if (enable_thinking is defined and enable_thinking) or messages[0]['role'] in ['system', 'developer'] -%}"
    "{{- '<|turn>system\\n' -}}"
    "{%- if enable_thinking is defined and enable_thinking -%}"
    "{{- '<|think|>\\n' -}}"
    "{%- endif -%}"
    "{%- if messages[0]['role'] in ['system', 'developer'] -%}"
    "{{- messages[0]['content'] | trim -}}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{{- '<turn|>\\n' -}}"
    "{%- else -%}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{- '<|turn>' + role + '\\n' + message['content'] -}}"
    "{%- if not (loop.last and (continue_final_message is defined and continue_final_message)) -%}"
    "{{- '<turn|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|turn>model\\n<|channel>thought\\n<channel|>' -}}"
    "{%- endif -%}"
)
