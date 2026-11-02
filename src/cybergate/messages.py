"""OpenAI-format message handling shared by the policy, the prompt builders and the middleware.

`extract_user_turns` is written out three times in the canonical package (`gate_probe.py:1241-1251`,
`middleware.py:616-626`, and inside `extract_system_and_user`) with slightly different list-content
joining each time. This module keeps the two distinct behaviours as two named functions rather than
unifying them, because the difference is observable:

  * `extract_user_turns` joins list parts with `" "` and keeps every dict part (`gate_probe` /
    `middleware` behaviour),
  * `extract_system_and_user` joins with `""` and keeps only parts whose `type == "text"`.

Unifying them would change what text gets hashed into a cache key, which is a behaviour change.
Phase 6 can reconcile them deliberately; this port does not.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from .boilerplate import DEFAULT_SYSTEM_PROMPT

__all__ = ["extract_system_and_user", "extract_user_turns"]

_PRIOR_THINK_RE = re.compile(
    r"^\s*(?:<think>.*?</think>|<\|channel>thought.*?<channel\|>)\s*",
    re.DOTALL,
)


def extract_user_turns(messages: Sequence[dict[str, Any]]) -> list[str]:
    """Return every non-empty user turn's text, in order.

    Multimodal list content is flattened by joining every dict part's `text` with a single space,
    including parts that are not `type == "text"`.
    """
    turns: list[str] = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            text_s = str(content).strip()
            if text_s:
                turns.append(text_s)
    return turns


def extract_system_and_user(
    messages: Sequence[dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]]]:
    """Extract `(sys_prompt, last_user_text, prior_messages)` from OpenAI-format `messages`.

    Only a `system`/`developer` message at index 0 is treated as the system prompt. Multimodal list
    content is flattened by concatenating the `text` of parts whose `type == "text"`.
    """
    sys_prompt = DEFAULT_SYSTEM_PROMPT
    user_text = ""
    prior: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if i == 0 and role in ("system", "developer"):
            sys_prompt = str(content).strip()
        else:
            text_val = str(content)
            if role == "assistant":
                text_val = _PRIOR_THINK_RE.sub("", text_val).strip()
            prior.append({"role": role, "content": text_val})
    for msg in reversed(prior):
        if msg["role"] == "user":
            user_text = msg["content"]
            break
    return sys_prompt, user_text, prior
