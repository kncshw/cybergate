"""Unit tests for the shipped kit `chat_template.jinja` files and native `/v1/chat/completions` fixes.

Covers:
1. Exact Jinja2 rendering parity between `kits/{gemma4,qwen38}/chat_template.jinja` and
   `family.build_raw_prompt(..., mode="closed_loop_step1")`, plus suffix adoption by
   `RefusalSteeredCoTProcessor`.
2. Qwen3.8 `reasoning_effort` (`minimal`/`low`/`medium`/`high`/`xhigh`/`max`) and active-tool-loop
   `preserve_thinking` behavior.
3. Gemma-4 `format_function_declaration` brace balance on tools with `response` schemas and
   multi-turn tool-loop closure.
4. Unclosed-thought safety (`content == ""` + sanitized `reasoning_content`).
5. `/tokenize` verification & fallback when `--chat-template` is missing (HTTP 500 on tool turns,
   `/v1/completions` fallback on non-tool turns).
6. Stripping client-supplied `cybergate_*` keys from `chat_template_kwargs` and `vllm_xargs`.
7. Native `tool_calls` deduplication (streaming & non-streaming, including all-dropped `finish_reason`
   reset and `_OPENER_PREFIX_RE` cleanup).
8. Schema-aware Qwen XML `<function=...><parameter=...>` fallback parameter typing.
9. `/tokenize`-backed `max_tokens` context clamping, middleware `system` + `tools` prefill clamping,
   `temperature=0.0` default, `n > 1` multi-choice sanitization, `response_format` opener
   suppression, agent tool-loop probe normalization, merged assistant `reasoning_content`, and
   `CyberGateVLLMClient` native-chat hook routing.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cybergate.boilerplate import INLINE_COT_SUPPRESSION, OPEN_THINK_PREFIX, POST_THINK_OPENER
from cybergate.config import KITS_DIR, STEER_AUTH_XARG, SUPPRESS_OPENER_XARG
from cybergate.families import family_by_name
from cybergate.hooks.cot_controller import RefusalSteeredCoTProcessor, SteeringConfig
from cybergate.serving import CyberGateThinkMiddleware, CyberGateVLLMClient, proxy as proxy_mod
from cybergate.serving.middleware import _normalize_messages_for_probe
from cybergate.serving.think_routes import (
    RouteConfig,
    build_native_closed_loop_chat_payload,
    drive_sync,
    measured_closed_loop_route,
    native_chat_closed_loop_route,
    postprocess_native_closed_loop_chat_response,
)
from cybergate.state import keys as key_builders
from cybergate.state.memory import MemoryDecisionStore


def _compile_kit_template(kit_name: str):
    jinja2 = pytest.importorskip("jinja2")
    from jinja2 import nodes
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class _HFGenerationExtension(Extension):
        """Stand-in for Hugging Face `transformers`'s `{% generation %}...{% endgeneration %}` tag."""

        tags = {"generation"}

        def parse(self, parser: Any) -> Any:
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.Scope(body, lineno=lineno)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_HFGenerationExtension],
    )
    tpl_path = KITS_DIR / kit_name / "chat_template.jinja"
    assert tpl_path.is_file(), f"Missing kit template: {tpl_path}"
    return env.from_string(tpl_path.read_text(encoding="utf-8"))


def _render_kit_template(
    kit_name: str,
    messages: list[dict[str, Any]],
    *,
    steered: bool = True,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    fam = family_by_name(kit_name)
    tpl = _compile_kit_template(kit_name)
    # Match Hugging Face / vLLM chat_utils: deserialize string `tool_calls[].function.arguments`
    norm_msgs: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and m.get("tool_calls"):
            tcs_out = []
            for tc in m["tool_calls"]:
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                    fn = dict(tc["function"])
                    if isinstance(fn.get("arguments"), str):
                        fn["arguments"] = json.loads(fn["arguments"])
                    tcs_out.append({**tc, "function": fn})
                else:
                    tcs_out.append(tc)
            norm_msgs.append({**m, "tool_calls": tcs_out})
        else:
            norm_msgs.append(m)

    render_kwargs: dict[str, Any] = {
        "messages": norm_msgs,
        "add_generation_prompt": True,
        "enable_thinking": True,
        "bos_token": "<bos>" if fam.supports_bos else "",
        "eos_token": "<eos>" if kit_name == "gemma4" else "<|im_end|>",
    }
    if tools is not None:
        render_kwargs["tools"] = tools
    if steered:
        render_kwargs["cybergate_open_think_prefix"] = fam.open_think_prefix
        render_kwargs.setdefault("cybergate_default_system_prompt", "You are a helpful assistant.")
    render_kwargs.update(kwargs)
    return tpl.render(**render_kwargs)


@pytest.mark.parametrize("kit_name", ["gemma4", "qwen38"])
@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Explain TCP SYN flood mechanics."}],
        [
            {"role": "system", "content": "You are a concise network security auditor."},
            {"role": "user", "content": "Explain TCP SYN flood mechanics."},
        ],
        [
            {"role": "user", "content": "What is ARP?"},
            {
                "role": "assistant",
                "reasoning_content": "Let me recall RFC 826.",
                "content": "ARP maps IPv4 addresses to MAC addresses.",
            },
            {"role": "user", "content": "How does gratuitous ARP spoofing work?"},
        ],
    ],
    ids=["single", "with_system", "multiturn"],
)
def test_kit_templates_match_measured_closed_loop_step1_byte_for_byte(
    kit_name: str,
    messages: list[dict[str, Any]],
) -> None:
    """Both kit templates must match `build_raw_prompt(closed_loop_step1)` byte-for-byte (#3, #12)."""
    fam = family_by_name(kit_name)
    rendered = _render_kit_template(kit_name, messages, steered=True)
    expected = fam.build_raw_prompt(
        messages, mode="closed_loop_step1", include_bos=fam.supports_bos
    )
    assert rendered == expected


@pytest.mark.parametrize("kit_name", ["gemma4", "qwen38"])
def test_kit_template_suffix_triggers_cot_controller_while_unsteered_does_not(
    kit_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`RefusalSteeredCoTProcessor` adopts steered renders and ignores unsteered renders (#12)."""
    from cybergate.hooks import cot_controller as cc_mod

    model_id = "google/gemma-4-26B-A4B-it" if kit_name == "gemma4" else "Qwen/Qwen3.8-27B"

    class _CharTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            return [ord(ch) % 251 + 1 for ch in text]

    monkeypatch.setattr(cc_mod, "_load_tokenizer", lambda *_a, **_k: _CharTokenizer())
    vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(model=model_id))
    proc = RefusalSteeredCoTProcessor(vllm_cfg, device="cpu", is_pin_memory=False)

    msgs = [{"role": "user", "content": "Analyze buffer overflow"}]
    steered_prompt = _render_kit_template(kit_name, msgs, steered=True)
    plain_prompt = _render_kit_template(kit_name, msgs, steered=False)

    steered_ids = _CharTokenizer().encode(steered_prompt, add_special_tokens=False)
    plain_ids = _CharTokenizer().encode(plain_prompt, add_special_tokens=False)

    params = SimpleNamespace(extra_args=None)
    proc.update_state(
        SimpleNamespace(
            batch_size=2,
            added=[(0, params, plain_ids, []), (1, params, steered_ids, [])],
            removed=[],
            moved=[],
        )
    )
    assert 0 not in proc._rows, "Unsteered prompt must NOT be adopted by CoT controller"
    assert 1 in proc._rows, "Steered kit-template prompt MUST be adopted by CoT controller"


def test_qwen38_reasoning_effort_variants_and_active_tool_loop_thinking() -> None:
    """Qwen3.8 defaults to medium effort (no xhigh bloat), maps OpenCode variants, and preserves
    `<think>` only inside active tool loops unless `preserve_thinking=True` (#3).
    """
    msgs = [{"role": "user", "content": "Test"}]
    default_render = _render_kit_template("qwen38", msgs, steered=False)
    assert "Reasoning effort is set to" not in default_render

    low_render = _render_kit_template("qwen38", msgs, steered=False, reasoning_effort="minimal")
    assert "Reasoning effort is set to low." in low_render

    high_render = _render_kit_template("qwen38", msgs, steered=False, reasoning_effort="high")
    assert "Reasoning effort is set to xhigh." in high_render

    max_render = _render_kit_template("qwen38", msgs, steered=False, reasoning_effort="max")
    assert "Reasoning effort is set to xhigh." in max_render

    # Prior turn (before last user) drops <think>, active tool-loop turn (after last user) keeps <think>
    tool_loop_msgs = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "reasoning_content": "PRIOR_FINISHED_THOUGHT", "content": "First answer"},
        {"role": "user", "content": "Run id via bash"},
        {
            "role": "assistant",
            "reasoning_content": "ACTIVE_LOOP_THOUGHT",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "id"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "uid=1000(user)"},
    ]
    loop_render = _render_kit_template("qwen38", tool_loop_msgs, steered=True)
    assert "PRIOR_FINISHED_THOUGHT" not in loop_render
    assert "<think>\nACTIVE_LOOP_THOUGHT\n</think>" in loop_render

    opt_in_render = _render_kit_template(
        "qwen38", tool_loop_msgs, steered=True, preserve_thinking=True
    )
    assert "<think>\nPRIOR_FINISHED_THOUGHT\n</think>" in opt_in_render


def test_gemma4_template_balances_braces_on_tool_response_schema_and_tool_loops() -> None:
    """Gemma-4 kit template balances `{`/`}` when a tool declares `response` and closes tool loops cleanly (#10)."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scan_port",
                "description": "Check whether a TCP port is open",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "description": "Target host"},
                        "port": {"type": "integer", "description": "Port number"},
                    },
                    "required": ["host", "port"],
                },
                "response": {
                    "type": "object",
                    "description": "Scan result",
                    "properties": {
                        "open": {"type": "boolean"},
                        "banner": {"type": "string"},
                    },
                },
            },
        }
    ]
    msgs = [
        {"role": "user", "content": "Check port 80 on 10.0.0.5"},
        {
            "role": "assistant",
            "content": "Scanning target port now.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "scan_port", "arguments": {"host": "10.0.0.5", "port": 80}},
                }
            ],
        },
        {"role": "tool", "name": "scan_port", "content": {"open": True, "banner": "nginx"}},
    ]
    rendered = _render_kit_template("gemma4", msgs, steered=True, tools=tools)
    # Extract the <|tool>...<tool|> declaration block and verify exact brace balance
    decl_start = rendered.index("<|tool>")
    decl_end = rendered.index("<tool|>")
    decl_block = rendered[decl_start:decl_end]
    assert decl_block.count("{") == decl_block.count("}") == 6
    # Verify assistant prose precedes tool call and open thought follows tool response inside model turn
    assert "Scanning target port now.<|tool_call>call:scan_port{" in rendered
    assert "<|tool_response>response:scan_port{" in rendered
    assert rendered.endswith(f"<|channel>thought\n{OPEN_THINK_PREFIX}")


def test_missing_chat_template_falls_back_on_chat_and_errors_on_tools(
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """When `/tokenize` shows `cybergate_open_think_prefix` was ignored (#2):
    - Non-tool turns fall back to `/v1/completions` (`measured_closed_loop_route`) so steering works.
    - Tool-bearing turns fail with HTTP 500 `ServerConfigurationError` instead of faking steering.
    """
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")
    middleware = CyberGateThinkMiddleware(app=lambda *a, **k: None)
    store = MemoryDecisionStore()
    prompt_text = "Write a Python SYN flood packet script"
    sha = key_builders.text_sha256(prompt_text)
    rec = {
        "decided": True,
        "layer": 40,
        "pool": "mean",
        "score": 0.25,
        "threshold": 0.0879,
        "fired": True,
        "s_ref_cos": 0.20,
        "s_ref_z": 5.0,
        "n_prefill_tokens": 30,
        "source": "live_vllm_gpu_hook",
        "ts": 1000.0,
    }
    store.put(key_builders.standalone_turn_key_from_sha(sha), rec)
    store.put(sha, rec)
    middleware.store = store

    class _NoTemplateDispatch:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def post_json(
            self, scope: dict[str, Any], path: str, payload: dict[str, Any]
        ) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
            self.calls.append((path, dict(payload)))
            if path == "/tokenize":
                # Simulate vLLM started WITHOUT --chat-template: prefix kwarg is ignored!
                if "prompt" in payload:
                    toks = [100 + (ord(c) % 97) for c in str(payload["prompt"])]
                    return (200, [(b"content-type", b"application/json")], {"count": len(toks), "tokens": toks})
                plain_toks = [1, 2, 3, 4, 5]
                return (200, [(b"content-type", b"application/json")], {"count": len(plain_toks), "tokens": plain_toks})
            if path == "/v1/completions":
                return (
                    200,
                    [(b"content-type", b"application/json")],
                    {
                        "id": "cmpl-fallback",
                        "choices": [
                            {
                                "index": 0,
                                "text": "- Step 1: fallback derivation.</think>Fallback steered answer.",
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    },
                )
            raise AssertionError(f"Unexpected path {path}")

    disp = _NoTemplateDispatch()
    middleware.dispatch = disp  # type: ignore[assignment]

    # 1. Non-tool turn falls back to /v1/completions
    out, status, _ = asyncio.run(
        middleware._handle_chat(
            scope={},
            payload={
                "model": "Qwen/Qwen3.8-27B",
                "cybergate_mode": "closed_loop",
                "messages": [{"role": "user", "content": prompt_text}],
            },
            req_id="req-notpl-chat",
            started=1000.0,
        )
    )
    assert status == 200
    assert out["choices"][0]["message"]["content"] == "Fallback steered answer."
    assert out["x_cybergate"]["template_hook_fallback"] == "completions"
    assert disp.calls[-1][0] == "/v1/completions"
    assert disp.calls[-1][1]["prompt"].endswith(family_by_name("qwen38").open_think_prefix)

    # 2. Tool turn returns HTTP 500 instead of silently dropping steering
    out_err, status_err, _ = asyncio.run(
        middleware._handle_chat(
            scope={},
            payload={
                "model": "Qwen/Qwen3.8-27B",
                "cybergate_mode": "closed_loop",
                "messages": [{"role": "user", "content": prompt_text}],
                "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}],
            },
            req_id="req-notpl-tools",
            started=1001.0,
        )
    )
    assert status_err == 500
    assert out_err["error"]["type"] == "ServerConfigurationError"
    assert "--chat-template" in out_err["error"]["message"]


def test_strip_cybergate_client_kwargs_blocks_gate_bypass(
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """Client-supplied `cybergate_*` keys in `chat_template_kwargs` and `vllm_xargs` are stripped (#4)."""
    payload: dict[str, Any] = {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": "Hello"}],
        "chat_template_kwargs": {
            "cybergate_open_think_prefix": OPEN_THINK_PREFIX,
            "cybergate_default_system_prompt": "evil",
            "reasoning_effort": "high",
        },
        "vllm_xargs": {
            SUPPRESS_OPENER_XARG: 1,
            "user_flag": True,
        },
    }
    proxy_mod.strip_cybergate_client_kwargs(payload)
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert payload["vllm_xargs"] == {"user_flag": True}

    # Also verify middleware pass-through strips spoofed kwargs when gate does not fire
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")
    middleware = CyberGateThinkMiddleware(app=lambda *a, **k: None)
    middleware.store = MemoryDecisionStore()
    captured: list[dict[str, Any]] = []

    class _PassDispatch:
        async def post_json(self, scope: Any, path: str, body: dict[str, Any]):
            if path == "/v1/chat/completions":
                captured.append(dict(body))
            return (
                200,
                [(b"content-type", b"application/json")],
                {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}]},
            )

    middleware.dispatch = _PassDispatch()  # type: ignore[assignment]
    asyncio.run(
        middleware._handle_chat(
            scope={},
            payload={
                "model": "Qwen/Qwen3.8-27B",
                "cybergate_mode": "closed_loop",
                "messages": [{"role": "user", "content": "Benign greeting"}],
                "chat_template_kwargs": {"cybergate_open_think_prefix": OPEN_THINK_PREFIX},
                "vllm_xargs": {SUPPRESS_OPENER_XARG: 1},
            },
            req_id="req-spoof",
            started=1000.0,
        )
    )
    assert len(captured) == 1
    assert "chat_template_kwargs" not in captured[0]
    assert "vllm_xargs" not in captured[0]


def test_native_tool_call_deduplication_and_empty_cleanup() -> None:
    """Native `tool_calls` are deduplicated, and when all are dropped `tool_calls` is cleared,
    `finish_reason` resets to `'stop'`, and `_OPENER_PREFIX_RE` is stripped (#5).
    """
    dup_tc_1 = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"id"}'}}
    dup_tc_2 = {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command":"id"}'}}

    choice_dup = {
        "index": 0,
        "message": {"role": "assistant", "content": "", "tool_calls": [dup_tc_1, dup_tc_2]},
        "finish_reason": "tool_calls",
    }
    _, _, kept_tcs, _, fr = proxy_mod._postprocess_choice_dict(choice_dup, has_tools=True)
    assert len(kept_tcs) == 1
    assert fr == "tool_calls"

    # Case where native tool_calls contains only empty/dropped entries + forced opener text
    dropped_obj = {
        "id": "chatcmpl-drop",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"{POST_THINK_OPENER}Remaining text.",
                    "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "", "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    trace_sse: dict[str, Any] = {}
    sse_raw = proxy_mod.format_sse_stream(dropped_obj, has_tools=True, trace_out=trace_sse).decode("utf-8")
    assert trace_sse["tool_calls"] == []
    assert trace_sse["finish_reason"] == "stop"
    assert trace_sse["content"] == "Remaining text."
    assert '"tool_calls"' not in sse_raw
    assert '"finish_reason": "stop"' in sse_raw


def test_qwen_xml_fallback_parser_preserves_string_types_and_uses_tool_schema() -> None:
    """`_QWEN_FUNC_CALL_RE` keeps strings as `str` (`{"port": 8080}`, `"123"`) while coercing
    schema-declared non-string types (`integer`, `boolean`, `object`, `array`) (#6).
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "background": {"type": "boolean"},
                        "env": {"type": "object"},
                    },
                },
            },
        },
    ]
    xml_text = (
        "<tool_call>\n"
        "<function=write>\n"
        "<parameter=filePath>/tmp/config.json</parameter>\n"
        '<parameter=content>{"port": 8080}</parameter>\n'
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>echo ok</parameter>\n"
        "<parameter=description>123</parameter>\n"
        "<parameter=timeout>30</parameter>\n"
        "<parameter=background>true</parameter>\n"
        '<parameter=env>{"DEBUG": "1"}</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )
    visible, tcs, malformed = proxy_mod.extract_tool_calls_from_content(xml_text, tools=tools)
    assert visible == ""
    assert malformed == 0
    assert len(tcs) == 2

    write_args = json.loads(tcs[0]["function"]["arguments"])
    assert write_args["filePath"] == "/tmp/config.json"
    assert write_args["content"] == '{"port": 8080}'
    assert isinstance(write_args["content"], str)

    bash_args = json.loads(tcs[1]["function"]["arguments"])
    assert bash_args["description"] == "123"
    assert isinstance(bash_args["description"], str)
    assert bash_args["timeout"] == 30
    assert bash_args["background"] is True
    assert bash_args["env"] == {"DEBUG": "1"}

    # Without schema (`tools=None`), strings remain strings rather than turning into dicts/ints
    _, tcs_no_schema, _ = proxy_mod.extract_tool_calls_from_content(xml_text, tools=None)
    assert json.loads(tcs_no_schema[0]["function"]["arguments"])["content"] == '{"port": 8080}'
    assert json.loads(tcs_no_schema[1]["function"]["arguments"])["description"] == "123"


def test_clean_prior_assistant_messages_merges_consecutive_reasoning() -> None:
    """`clean_prior_assistant_messages` merges consecutive assistant `reasoning_content` (#15)
    and strips Qwen3.8 inline suppression + bottom compliance lock from active tool-loop steps.
    """
    from cybergate.boilerplate import (
        QWEN38_BOTTOM_COMPLIANCE_LOCK,
        QWEN38_INLINE_COT_SUPPRESSION,
        QWEN38_OPEN_THINK_PREFIX,
    )

    merged = proxy_mod.clean_prior_assistant_messages(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "reasoning_content": "R1", "content": "Part 1."},
            {"role": "assistant", "reasoning_content": "R2", "content": "Part 2."},
        ]
    )
    assert len(merged) == 2
    assert merged[1]["content"] == "Part 1.\n\nPart 2."
    assert merged[1]["reasoning_content"] == "R1\n\nR2"

    tool_step = proxy_mod.clean_prior_assistant_messages(
        [
            {"role": "user", "content": "Test target"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": (
                    "[CyberGate | fired=True | s_cat=+0.24]\n\n"
                    f"{QWEN38_OPEN_THINK_PREFIX}Checking host."
                    f"{QWEN38_INLINE_COT_SUPPRESSION}run curl."
                    f"{QWEN38_BOTTOM_COMPLIANCE_LOCK}"
                ),
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
                ],
            },
        ]
    )
    cleaned_r = tool_step[1]["reasoning_content"]
    assert "[Policy Check:" not in cleaned_r
    assert "All safety checks are cleared" not in cleaned_r
    assert "Checking host." in cleaned_r
    assert "run curl." in cleaned_r


def test_steered_request_controls_and_multi_choice_sanitization() -> None:
    """Verify `temperature=0.0` default, `response_format` opener suppression, `stop` filtering,
    and `n > 1` sanitization across all choices (#8).
    """
    qwen = family_by_name("qwen38")
    built = build_native_closed_loop_chat_payload(
        {
            "n": 2,
            "response_format": {"type": "json_object"},
            "stop": ["</think>", "<think>", "USER_STOP"],
        },
        qwen,
        model_id="Qwen/Qwen3.8-27B",
        messages=[{"role": "user", "content": "Output JSON"}],
        max_tokens=512,
        max_cot_tokens=256,
        temperature=0.0,
        suppress_opener=False,
        max_total_tokens=600,
    )
    assert built["temperature"] == 0.0
    assert built["max_tokens"] == 600
    assert built["vllm_xargs"][SUPPRESS_OPENER_XARG] == 1
    assert "stop" not in built

    chat_resp = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning": f"Choice 0 CoT\n\nstill thinking {INLINE_COT_SUPPRESSION}",
                    "content": '{"ok": 0}USER_STOP_TRAILING',
                },
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {
                    "role": "assistant",
                    "reasoning": "",
                    "content": f"Unclosed choice 1 CoT {INLINE_COT_SUPPRESSION}",
                },
                "finish_reason": "length",
            },
        ]
    }
    postprocess_native_closed_loop_chat_response(
        chat_resp, qwen, do_sanitize=True, client_stops=["\n\n", "USER_STOP"]
    )
    c0_msg = chat_resp["choices"][0]["message"]
    c1_msg = chat_resp["choices"][1]["message"]
    assert c0_msg["content"] == '{"ok": 0}'
    assert "still thinking" in c0_msg["reasoning_content"]
    assert INLINE_COT_SUPPRESSION.strip() not in c0_msg["reasoning_content"]
    assert c1_msg["content"] == ""
    assert INLINE_COT_SUPPRESSION.strip() not in c1_msg["reasoning_content"]
    assert "Unclosed choice 1 CoT" in c1_msg["reasoning_content"]


def test_probe_normalization_on_agent_tool_loops() -> None:
    """`_normalize_messages_for_probe` strips trailing active tool calls/outputs and folds prior
    intermediate tool loops into clean `assistant` summaries with zero `tool` headers (#9).
    """
    qwen = family_by_name("qwen38")
    gemma = family_by_name("gemma4")

    # 1. Single-turn active tool loop -> trims trailing tool steps back to bare user turn
    single_loop = [
        {"role": "system", "content": "OpenCode system prompt"},
        {"role": "user", "content": "Scan 10.0.0.5"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"nmap"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "80/tcp open"},
    ]
    norm_single = _normalize_messages_for_probe(single_loop)
    assert norm_single == [{"role": "user", "content": "Scan 10.0.0.5"}]

    # 2. Multi-turn with prior tool loop + active tool loop -> strictly alternating user/assistant/user
    multi_loop = [
        {"role": "system", "content": "OpenCode system prompt"},
        {"role": "user", "content": "Scan 10.0.0.5"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"nmap"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "80/tcp open"},
        {"role": "assistant", "content": "Port 80 is open."},
        {"role": "user", "content": "Craft exploit for port 80"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command":"curl"}'}}],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "HTTP 200"},
    ]
    norm_multi = _normalize_messages_for_probe(multi_loop)
    assert [m["role"] for m in norm_multi] == ["user", "assistant", "user"]
    assert "[Tool call: bash(" in norm_multi[1]["content"]
    assert "[Tool result: 80/tcp open]" in norm_multi[1]["content"]
    assert "Port 80 is open." in norm_multi[1]["content"]

    qwen_prompt = qwen.build_raw_prompt(norm_multi, mode="baseline", include_bos=False)
    gemma_prompt = gemma.build_raw_prompt(norm_multi, mode="baseline", include_bos=True)
    assert "<|im_start|>tool" not in qwen_prompt
    assert "<|turn>tool" not in gemma_prompt
    assert qwen_prompt.endswith(
        "Craft exploit for port 80<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert gemma_prompt.endswith(
        "Craft exploit for port 80<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def test_remote_client_uses_native_chat_route_when_hook_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CyberGateVLLMClient` shares `native_chat_closed_loop_route` when `/tokenize` confirms hook (#13)."""
    client = CyberGateVLLMClient(
        base_url="https://example.invalid/v1",
        model_id="Qwen/Qwen3.8-27B",
        default_mode="closed_loop",
    )
    monkeypatch.setattr(client, "_supports_native_chat_hook", lambda _fam: True)
    posted: list[tuple[str, dict[str, Any]]] = []

    def _fake_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        posted.append((path, dict(payload)))
        return {
            "id": "chatcmpl-remote",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "Qwen/Qwen3.8-27B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning": f"- Step 1: derive.{INLINE_COT_SUPPRESSION}",
                        "content": "Remote steered chat answer.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37},
        }

    monkeypatch.setattr(client, "_post_json", _fake_post)
    out = client.chat_completions(
        messages=[{"role": "user", "content": "Test prompt"}],
        mode="closed_loop",
        force_gate=True,
        sanitize_reasoning=True,
    )
    assert len(posted) == 1
    assert posted[0][0] == "/chat/completions"
    assert (
        posted[0][1]["chat_template_kwargs"]["cybergate_open_think_prefix"]
        == family_by_name("qwen38").open_think_prefix
    )
    assert out["choices"][0]["message"]["content"] == "Remote steered chat answer."
    assert out["x_cybergate"]["routed"] == "cybergate_measured_closed_loop"
    assert out["x_cybergate"]["num_inline_injections"] == 1


@pytest.mark.parametrize("kit_name", ["qwen38", "gemma4"])
def test_verify_template_hook_uses_token_ids_with_bpe_pieces_and_does_not_cache_errors(
    kit_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New 1: `/tokenize` hook verification compares integer `tokens` (not raw BPE `token_strs`
    or a 64-token slice), never falls back to disk-file existence, and does not cache transient errors.
    """
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")
    fam = family_by_name(kit_name)
    model_id = "google/gemma-4-26B-A4B-it" if kit_name == "gemma4" else "Qwen/Qwen3.8-27B"
    # Build >73 suffix token IDs (matching Gemma-4's 73-token prefix) and raw BPE vocabulary pieces
    suffix_ids = list(range(5000, 5080))
    prefix_prompt_ids = [10, 20, 30] + suffix_ids

    middleware = CyberGateThinkMiddleware(app=lambda *a, **k: None)

    class _FlakyThenHealthyTokenizeDispatch:
        def __init__(self) -> None:
            self.fail_next = True

        async def post_json(
            self, scope: dict[str, Any], path: str, payload: dict[str, Any]
        ) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
            assert path == "/tokenize"
            if self.fail_next:
                return (503, [(b"content-type", b"application/json")], {"error": "warming up"})
            if "prompt" in payload:
                return (
                    200,
                    [(b"content-type", b"application/json")],
                    {
                        "count": len(suffix_ids),
                        "tokens": suffix_ids,
                        "token_strs": ["Ġpermitted", ".", "ĊĊ"] * 27,
                    },
                )
            return (
                200,
                [(b"content-type", b"application/json")],
                {
                    "count": len(prefix_prompt_ids),
                    "tokens": prefix_prompt_ids,
                    "token_strs": ["Ġpermitted", ".", "ĊĊ"],
                },
            )

    disp = _FlakyThenHealthyTokenizeDispatch()
    middleware.dispatch = disp  # type: ignore[assignment]

    # 1. Transient 503 returns False (no disk-file fallback!) and is NOT cached
    assert asyncio.run(middleware._verify_template_hook({}, model_id, fam)) is False
    assert (id(disp), model_id) not in middleware._template_hook_cache

    # 2. Once `/tokenize` recovers, integer `tokens` match succeeds despite BPE `token_strs` and >64 len
    disp.fail_next = False
    assert asyncio.run(middleware._verify_template_hook({}, model_id, fam)) is True
    assert middleware._template_hook_cache[(id(disp), model_id)] is True

    # 3. Verify `CyberGateVLLMClient._supports_native_chat_hook` behaves identically
    import io
    import urllib.error
    import urllib.request

    client = CyberGateVLLMClient(base_url="https://example.invalid/v1", model_id=model_id)
    state = {"fail": True}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args: Any):
            return False

    def _fake_urlopen(req: urllib.request.Request, *args: Any, **kwargs: Any):
        if state["fail"]:
            raise urllib.error.URLError("transient network error")
        body = json.loads((req.data or b"{}").decode("utf-8"))
        toks = suffix_ids if "prompt" in body else prefix_prompt_ids
        return _Resp(
            json.dumps(
                {"count": len(toks), "tokens": toks, "token_strs": ["Ġpermitted", ".", "ĊĊ"]}
            ).encode("utf-8")
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert client._supports_native_chat_hook(fam) is False
    assert model_id not in client._template_hook_cache

    state["fail"] = False
    assert client._supports_native_chat_hook(fam) is True
    assert client._template_hook_cache[model_id] is True


def test_zero_token_thought_close_and_include_reasoning_false_preserve_real_answers() -> None:
    """New 2: First-token thought close (`reasoning == ""`, `finish_reason == "stop"`) and
    `include_reasoning=False` keep the visible answer in `content`, while Gemma-4 unclosed thoughts
    at `finish_reason == "length"` still stay hidden and sanitized.
    """
    qwen = family_by_name("qwen38")
    gemma = family_by_name("gemma4")

    # 1. Qwen & Gemma 0-token thought close -> real answer preserved in `content`
    for fam in (qwen, gemma):
        zero_cot_resp = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning": "",
                        "content": "Real visible answer after empty thought.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        tele = postprocess_native_closed_loop_chat_response(
            zero_cot_resp, fam, do_sanitize=True
        )
        assert zero_cot_resp["choices"][0]["message"]["content"] == (
            "Real visible answer after empty thought."
        )
        assert tele["thought_close_tag_present"] is True

    # 2. Gemma-4 unclosed thought at `finish_reason == "length"` -> hidden from `content` and sanitized
    gemma_unclosed = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning": "",
                    "content": f"Step 1: internal unclosed chain of thought {INLINE_COT_SUPPRESSION}",
                },
                "finish_reason": "length",
            }
        ]
    }
    tele_unclosed = postprocess_native_closed_loop_chat_response(
        gemma_unclosed, gemma, do_sanitize=True
    )
    assert gemma_unclosed["choices"][0]["message"]["content"] == ""
    assert tele_unclosed["thought_close_tag_present"] is False
    assert (
        INLINE_COT_SUPPRESSION.strip()
        not in gemma_unclosed["choices"][0]["message"]["reasoning_content"]
    )
    assert (
        "Step 1: internal unclosed chain of thought"
        in gemma_unclosed["choices"][0]["message"]["reasoning_content"]
    )

    # 3. Client `include_reasoning=False` is stripped before vLLM and honoured post-split
    captured_payloads: list[dict[str, Any]] = []

    def _transport(req: Any) -> dict[str, Any]:
        captured_payloads.append(dict(req.payload))
        assert "include_reasoning" not in req.payload
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning": f"Derived safely.{INLINE_COT_SUPPRESSION}",
                        "content": "Answer with reasoning suppressed for caller.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    cfg = RouteConfig(
        family=qwen,
        model_id="Qwen/Qwen3.8-27B",
        chat_payload={"include_reasoning": False},
        sanitize_reasoning=True,
    )
    res = drive_sync(
        native_chat_closed_loop_route(
            "closed_loop", [{"role": "user", "content": "Test"}], cfg
        ),
        _transport,
    )
    assert len(captured_payloads) == 1
    assert res.content == "Answer with reasoning suppressed for caller."
    assert res.response["choices"][0]["message"]["content"] == (
        "Answer with reasoning suppressed for caller."
    )
    assert res.response["choices"][0]["message"]["reasoning_content"] == ""


def test_gemma4_tool_call_inside_unclosed_thought_sanitizes_prose_and_keeps_tool_call() -> None:
    """New 3 & New 8: On Gemma-4, prose before a tool call inside an unclosed thought is moved to
    sanitized `reasoning_content` (`content == ""`, `thought_close_tag_present == False`), and
    non-OBJECT `response` schemas balance `{`/`}` in `kits/gemma4/chat_template.jinja`.
    """
    gemma = family_by_name("gemma4")
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "id"}'},
    }
    resp = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning": "",
                    "content": f"Step 1: run id in sandbox.{INLINE_COT_SUPPRESSION}",
                    "tool_calls": [tc],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    tele = postprocess_native_closed_loop_chat_response(
        resp, gemma, do_sanitize=True, tool_turn=True
    )
    msg = resp["choices"][0]["message"]
    assert msg["content"] == ""
    assert msg["tool_calls"] == [tc]
    assert tele["thought_close_tag_present"] is False
    assert INLINE_COT_SUPPRESSION.strip() not in msg["reasoning_content"]
    assert "Step 1: run id in sandbox." in msg["reasoning_content"]

    # Non-OBJECT `response` schema brace check (New 8)
    non_obj_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_banner",
                "description": "Return service banner",
                "parameters": {"type": "object", "properties": {"port": {"type": "integer"}}},
                "response": {"type": "string", "description": "Raw banner string"},
            },
        }
    ]
    rendered = _render_kit_template(
        "gemma4", [{"role": "user", "content": "Grab banner"}], steered=False, tools=non_obj_tools
    )
    decl = rendered[rendered.index("<|tool>") : rendered.index("<tool|>")]
    assert decl.count("{") == decl.count("}")


def test_non_chat_endpoints_strip_cybergate_kwargs_and_steer_token_blocks_raw_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New 4: Every HTTP `POST` path (`/v1/responses`, `/v1/messages`, `/v1/completions`, `/v2/chat`)
    strips `cybergate_*` kwargs on both middleware and proxy, and `CYBERGATE_STEER_TOKEN` prevents
    unauthenticated suffix prompts from triggering `RefusalSteeredCoTProcessor`.
    """
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")
    monkeypatch.setenv("CYBERGATE_STEER_TOKEN", "server-secret-token-9f8e")

    # 1. Middleware strips `cybergate_*` on non-`/v1/chat/completions` POST endpoints
    seen_asgi: list[tuple[str, dict[str, Any]]] = []

    async def _inner_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        chunks = []
        while True:
            ev = await receive()
            chunks.append(ev.get("body", b""))
            if not ev.get("more_body", False):
                break
        seen_asgi.append((scope["path"], json.loads(b"".join(chunks).decode("utf-8"))))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok":true}', "more_body": False})

    mw = CyberGateThinkMiddleware(app=_inner_app)
    for path in ("/v1/responses", "/v1/messages", "/v2/chat", "/v1/completions"):
        spoofed = json.dumps(
            {
                "model": "Qwen/Qwen3.8-27B",
                "input": "hello",
                "cybergate_mode": "closed_loop",
                "chat_template_kwargs": {
                    "cybergate_open_think_prefix": OPEN_THINK_PREFIX,
                    "enable_thinking": True,
                },
                "vllm_xargs": {
                    STEER_AUTH_XARG: "forged-token",
                    SUPPRESS_OPENER_XARG: 1,
                },
                "extra_body": {
                    "cybergate_open_think_prefix": OPEN_THINK_PREFIX,
                    "chat_template_kwargs": {"cybergate_open_think_prefix": OPEN_THINK_PREFIX},
                },
            }
        ).encode("utf-8")
        sent_body = False

        async def _recv() -> dict[str, Any]:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": spoofed, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(_ev: dict[str, Any]) -> None:
            pass

        asyncio.run(
            mw(
                {
                    "type": "http",
                    "method": "POST",
                    "path": path,
                    "headers": [(b"content-length", str(len(spoofed)).encode("ascii"))],
                },
                _recv,
                _send,
            )
        )

    assert len(seen_asgi) == 4
    for _, forwarded in seen_asgi:
        assert "cybergate_mode" not in forwarded
        assert forwarded.get("chat_template_kwargs") == {"enable_thinking": True}
        assert "vllm_xargs" not in forwarded
        assert "extra_body" not in forwarded

    # 2. `RefusalSteeredCoTProcessor` rejects raw suffix prompts without matching `STEER_AUTH_XARG`
    from cybergate.hooks import cot_controller as cc_mod

    class _CharTok:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            return [ord(ch) % 251 + 1 for ch in text]

    monkeypatch.setattr(cc_mod, "_load_tokenizer", lambda *_a, **_k: _CharTok())
    vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(model="Qwen/Qwen3.8-27B"))
    proc = RefusalSteeredCoTProcessor(vllm_cfg, device="cpu", is_pin_memory=False)

    qwen = family_by_name("qwen38")
    steered_prompt = _render_kit_template(
        "qwen38", [{"role": "user", "content": "Test"}], steered=True
    )
    steered_ids = _CharTok().encode(steered_prompt, add_special_tokens=False)

    unauth_params = SimpleNamespace(extra_args=None)
    forged_params = SimpleNamespace(extra_args={STEER_AUTH_XARG: "wrong-token"})
    built_chat = build_native_closed_loop_chat_payload(
        {},
        qwen,
        model_id="Qwen/Qwen3.8-27B",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=128,
        max_cot_tokens=128,
    )
    auth_params = SimpleNamespace(extra_args=built_chat["vllm_xargs"])

    # Also verify fallback `/v1/completions` route attaches the server steer token
    comp_gen = measured_closed_loop_route(
        "closed_loop",
        [{"role": "user", "content": "Test"}],
        RouteConfig(family=qwen, model_id="Qwen/Qwen3.8-27B"),
    )
    comp_req = next(comp_gen)
    assert comp_req.payload["vllm_xargs"][STEER_AUTH_XARG] == "server-secret-token-9f8e"

    proc.update_state(
        SimpleNamespace(
            batch_size=3,
            added=[
                (0, unauth_params, steered_ids, []),
                (1, forged_params, steered_ids, []),
                (2, auth_params, steered_ids, []),
            ],
            removed=[],
            moved=[],
        )
    )
    assert 0 not in proc._rows, "Raw suffix without STEER_AUTH_XARG must NOT steer"
    assert 1 not in proc._rows, "Forged STEER_AUTH_XARG must NOT steer"
    assert 2 in proc._rows, "Server-authenticated STEER_AUTH_XARG MUST steer"


def test_get_full_remaining_max_tokens_uses_live_tokenize_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_full_remaining_max_tokens` uses the real `/tokenize` count rather than `chars // 3`
    on dense hexdump tool outputs without modifying `messages`.
    """
    hexdump_line = "00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............\n"
    dense_hex = hexdump_line * 200  # ~12,000 chars -> ~4,000 est tokens, ~7,200 real tokens
    msgs = [
        {"role": "user", "content": "Inspect ELF header"},
        {"role": "tool", "tool_call_id": "c1", "content": dense_hex},
    ]
    monkeypatch.setattr(
        proxy_mod,
        "query_upstream_prompt_tokens",
        lambda *args, **kwargs: (7200, 8192),
    )
    remaining = proxy_mod.get_full_remaining_max_tokens(msgs, max_cot_tokens=180)
    assert remaining == 8192 - 7200 - 180 - 128
    assert msgs[1]["content"] == dense_hex


def test_steered_render_pins_medium_effort_and_disables_preserve_thinking() -> None:
    """New 7: Top-level `reasoning_effort` and client `preserve_thinking` are overridden on steered
    turns so the rendered Qwen3.8 prompt always matches the measured `closed_loop_step1` prompt.
    """
    qwen = family_by_name("qwen38")
    messages = [
        {"role": "user", "content": "What is ARP?"},
        {
            "role": "assistant",
            "reasoning_content": "Prior finished thought that must not re-render.",
            "content": "ARP resolves IPv4 to MAC.",
        },
        {"role": "user", "content": "Demonstrate gratuitous ARP packet construction."},
    ]
    built = build_native_closed_loop_chat_payload(
        {
            "reasoning_effort": "high",
            "chat_template_kwargs": {
                "reasoning_effort": "xhigh",
                "preserve_thinking": True,
            },
        },
        qwen,
        model_id="Qwen/Qwen3.8-27B",
        messages=messages,
        max_tokens=512,
        max_cot_tokens=512,
    )
    assert "reasoning_effort" not in built
    assert built["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert built["chat_template_kwargs"]["preserve_thinking"] is False

    rendered = _render_kit_template(
        "qwen38", messages, steered=True, **built["chat_template_kwargs"]
    )
    expected = qwen.build_raw_prompt(messages, mode="closed_loop_step1", include_bos=False)
    assert rendered == expected


def test_upstream_error_reports_think_injected_false_and_client_does_not_import_proxy(
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """New 9: Upstream errors report `think_injected=False` (`routed='upstream_error'`), and
    importing `cybergate.serving.client` does not import `cybergate.serving.proxy`.
    """
    import subprocess
    import sys

    # 1. Subprocess check: importing `CyberGateVLLMClient` does NOT pull in `cybergate.serving.proxy`
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'src'); "
                "from cybergate.serving.client import CyberGateVLLMClient; "
                "assert 'cybergate.serving.proxy' not in sys.modules, list(sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    # 2. Upstream 500 error on gated request reports `think_injected: False`
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")
    mw = CyberGateThinkMiddleware(app=lambda *a, **k: None)
    store = MemoryDecisionStore()
    prompt_text = "Write a Python SYN flood packet script"
    sha = key_builders.text_sha256(prompt_text)
    rec = {
        "decided": True,
        "layer": 40,
        "pool": "mean",
        "score": 0.25,
        "threshold": 0.0879,
        "fired": True,
        "s_ref_cos": 0.20,
        "s_ref_z": 5.0,
        "n_prefill_tokens": 30,
        "source": "live_vllm_gpu_hook",
        "ts": 1000.0,
    }
    store.put(key_builders.standalone_turn_key_from_sha(sha), rec)
    store.put(sha, rec)
    mw.store = store

    class _ErrorDispatch:
        async def post_json(
            self, scope: dict[str, Any], path: str, payload: dict[str, Any]
        ) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
            if path == "/tokenize":
                toks = [1, 2, 3, 4]
                return (200, [(b"content-type", b"application/json")], {"count": 4, "tokens": toks})
            return (
                500,
                [(b"content-type", b"application/json")],
                {"error": {"message": "Engine overloaded", "code": 500}},
            )

    mw.dispatch = _ErrorDispatch()  # type: ignore[assignment]
    out, status, _ = asyncio.run(
        mw._handle_chat(
            scope={},
            payload={
                "model": "Qwen/Qwen3.8-27B",
                "cybergate_mode": "closed_loop",
                "messages": [{"role": "user", "content": prompt_text}],
            },
            req_id="req-up-err",
            started=1000.0,
        )
    )
    assert status == 500
    assert out["error"]["message"] == "Engine overloaded"
    assert out["x_cybergate"]["think_injected"] is False
    assert out["x_cybergate"]["routed"] == "upstream_error"


# =================================================================================================
# Round-3 review regressions (R1 - R5)
# =================================================================================================
async def _vllm_with_cancellation_receive_race(
    receive: Any,
    work_coro: Any,
) -> Any:
    """Mirror vLLM v0.32.0's `@with_cancellation`: read body once, then race async work against a
    disconnect listener calling `await receive()`. If `receive()` yields `http.disconnect` while
    `work_coro` awaits, cancel `work_coro` and return `None` (which FastAPI serializes as `null`).
    """

    async def _listen_for_disconnect() -> None:
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return

    work_task = asyncio.create_task(work_coro)
    disc_task = asyncio.create_task(_listen_for_disconnect())
    done, _ = await asyncio.wait(
        [work_task, disc_task], return_when=asyncio.FIRST_COMPLETED
    )
    if disc_task in done and not work_task.done():
        work_task.cancel()
        try:
            await work_task
        except asyncio.CancelledError:
            pass
        return None
    disc_task.cancel()
    try:
        await disc_task
    except asyncio.CancelledError:
        pass
    return await work_task


@pytest.mark.parametrize(
    "path",
    [
        "/tokenize",
        "/detokenize",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/embeddings",
    ],
)
def test_r1_non_chat_posts_do_not_self_cancel_under_vllm_with_cancellation(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1: `replay_body` delegates post-body reads to the real ASGI `receive` channel so vLLM 0.32
    routes decorated with `@with_cancellation` never see a premature `http.disconnect` (`null`).
    """
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")
    monkeypatch.setenv("CYBERGATE_STEER_TOKEN", "server-secret-r1")

    seen_bodies: list[dict[str, Any]] = []

    async def _cancellable_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        chunks: list[bytes] = []
        while True:
            ev = await receive()
            chunks.append(ev.get("body", b""))
            if not ev.get("more_body", False):
                break
        parsed = json.loads(b"".join(chunks).decode("utf-8"))
        seen_bodies.append(parsed)

        async def _async_engine_work() -> dict[str, Any]:
            # Yield across threadpool + event loop ticks just like real `/tokenize` / generation
            await asyncio.to_thread(lambda: sum(range(1000)))
            await asyncio.sleep(0.005)
            return {"ok": True, "path": scope["path"], "tokens": [1, 2, 3], "count": 3}

        result = await _vllm_with_cancellation_receive_race(receive, _async_engine_work())
        out_raw = json.dumps(result).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": out_raw, "more_body": False})

    mw = CyberGateThinkMiddleware(_cancellable_app)

    for with_forged_cybergate_kwargs in (False, True):
        req_obj: dict[str, Any] = {"model": "Qwen/Qwen3.8-27B", "prompt": "hello world"}
        if with_forged_cybergate_kwargs:
            req_obj["chat_template_kwargs"] = {"cybergate_open_think_prefix": "forged"}
            req_obj["vllm_xargs"] = {"cybergate_steer_token": "wrong-token"}

        raw = json.dumps(req_obj).encode("utf-8")
        client_disconnected = asyncio.Event()
        sent_body = False
        resp_chunks: list[bytes] = []

        async def _real_uvicorn_receive() -> dict[str, Any]:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await client_disconnected.wait()
            return {"type": "http.disconnect"}

        async def _send(msg: dict[str, Any]) -> None:
            if msg["type"] == "http.response.body":
                resp_chunks.append(msg.get("body", b""))

        async def _invoke() -> dict[str, Any] | None:
            try:
                await mw(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": path,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(raw)).encode("ascii")),
                        ],
                    },
                    _real_uvicorn_receive,
                    _send,
                )
            finally:
                client_disconnected.set()
            return json.loads(b"".join(resp_chunks).decode("utf-8"))

        out = asyncio.run(_invoke())
        assert out is not None, f"{path} returned null (cancelled by premature disconnect)"
        assert out["ok"] is True
        if with_forged_cybergate_kwargs and path not in (
            "/tokenize",
            "/detokenize",
            "/v1/embeddings",
        ):
            assert "chat_template_kwargs" not in seen_bodies[-1]
            assert "vllm_xargs" not in seen_bodies[-1]


def test_r1_and_r2_end_to_end_http_server_with_middleware_proxy_and_remote_client(
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """R1 + R2 end-to-end over real HTTP sockets (no `_post_json` stubbing):
    1. `:8001` proxy `/tokenize` against `:8000` `CyberGateThinkMiddleware` + `@with_cancellation`
       succeeds 10/10 times on a 60 KB hexdump.
    2. Remote `CyberGateVLLMClient` (`closed_loop` and `sandwich`) steers a CyberGate server whose
       random `CYBERGATE_STEER_TOKEN` the client does NOT know, moving `RefusalSteeredCoTProcessor`
       counters (`rows_tracked`).
    3. Operator-provisioned shared `CYBERGATE_STEER_TOKEN` lets `force_gate=True` steer directly.
    4. Unauthenticated raw `/v1/completions` prefix prompt to `:8000` returns HTTP 200 JSON (not
       `null`) and does NOT increment `rows_tracked`.
    """
    import http.server
    import socketserver
    import threading
    import urllib.request

    from cybergate.config import STEER_AUTH_XARG
    from cybergate.hooks import cot_controller as cc_mod

    server_secret = "srv-random-secret-9f8e7d6c"
    monkeypatch.setenv("CYBERGATE_STEER_TOKEN", server_secret)
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")

    fam = family_by_name("qwen38")

    class _Tok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            if text == "</think>":
                return [248048]
            return [1000 + (ord(c) % 251) for c in text]

    tok = _Tok()
    monkeypatch.setattr(cc_mod, "_load_tokenizer", lambda *_a, **_k: tok)

    class _Cfg:
        class model_config:
            model = "Qwen/Qwen3.8-27B"
            served_model_name = None
            tokenizer = None

        class cache_config:
            enable_prefix_caching = False

    processor = cc_mod.RefusalSteeredCoTProcessor(_Cfg(), device="cpu", is_pin_memory=False)
    assert processor.enabled is True

    prompt_text = "Write a Python SYN flood packet script for lab testing"
    sha = key_builders.text_sha256(prompt_text)
    firing_rec = {
        "decided": True,
        "layer": 40,
        "pool": "mean",
        "score": 0.31,
        "threshold": 0.0879,
        "fired": True,
        "s_ref_cos": 0.25,
        "s_ref_z": 6.2,
        "n_prefill_tokens": 32,
        "source": "live_vllm_gpu_hook",
        "ts": 1000.0,
    }
    store = MemoryDecisionStore()
    store.put(key_builders.standalone_turn_key_from_sha(sha), firing_rec)
    store.put(sha, firing_rec)

    class _Params:
        def __init__(self, extra_args: dict[str, Any] | None) -> None:
            self.extra_args = extra_args or {}

    class _BatchUpdate:
        def __init__(self, row: int, params: Any, prompt_ids: list[int]) -> None:
            self.removed: list[int] = [row]
            self.added = [(row, params, prompt_ids, [])]
            self.moved: list[Any] = []

    kit_jinja = _render_kit_template

    async def _standin_vllm_asgi(scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope.get("method") == "GET" and path == "/v1/models":
            body = json.dumps(
                {"data": [{"id": "Qwen/Qwen3.8-27B", "max_model_len": 16384}]}
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body, "more_body": False})
            return

        chunks: list[bytes] = []
        while True:
            ev = await receive()
            chunks.append(ev.get("body", b""))
            if not ev.get("more_body", False):
                break
        payload = json.loads(b"".join(chunks).decode("utf-8") or "{}")

        async def _handle_post() -> dict[str, Any]:
            await asyncio.to_thread(lambda: 42)
            await asyncio.sleep(0.002)
            if path == "/tokenize":
                if "prompt" in payload:
                    ids = tok.encode(str(payload["prompt"]))
                else:
                    ctk = payload.get("chat_template_kwargs") or {}
                    rendered = kit_jinja(
                        "qwen38",
                        messages=payload.get("messages") or [],
                        tools=payload.get("tools"),
                        **ctk,
                    )
                    ids = tok.encode(rendered)
                return {
                    "count": len(ids),
                    "tokens": ids,
                    "max_model_len": 16384,
                    "token_strs": ["Ġpiece"],
                }
            if path == "/v1/completions":
                p_txt = str(payload.get("prompt") or "")
                p_ids = tok.encode(p_txt)
                before_tracked = processor.stats.rows_tracked
                processor.update_state(
                    _BatchUpdate(0, _Params(payload.get("vllm_xargs")), p_ids)
                )
                was_steered = processor.stats.rows_tracked > before_tracked
                gen_txt = (
                    f"derived.{fam.inline_cot_suppression}done.</think>Steered completion."
                    if was_steered
                    else "Unsteered completion."
                )
                return {
                    "id": "cmpl-e2e",
                    "choices": [{"index": 0, "text": gen_txt, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(p_ids), "completion_tokens": 10, "total_tokens": len(p_ids) + 10},
                }
            if path == "/v1/chat/completions":
                ctk = payload.get("chat_template_kwargs") or {}
                rendered = kit_jinja(
                    "qwen38",
                    messages=payload.get("messages") or [],
                    tools=payload.get("tools"),
                    **ctk,
                )
                p_ids = tok.encode(rendered)
                before_tracked = processor.stats.rows_tracked
                processor.update_state(
                    _BatchUpdate(0, _Params(payload.get("vllm_xargs")), p_ids)
                )
                was_steered = processor.stats.rows_tracked > before_tracked
                resp: dict[str, Any] = {
                    "id": "chatcmpl-e2e",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "Qwen/Qwen3.8-27B",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning": (
                                    f"step 1.{fam.inline_cot_suppression}"
                                    if was_steered
                                    else "unsteered thought"
                                ),
                                "content": (
                                    "Steered chat answer."
                                    if was_steered
                                    else "Unsteered chat answer."
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
                }
                if payload.get("return_token_ids"):
                    resp["prompt_token_ids"] = p_ids[:8]
                    resp["choices"][0]["token_ids"] = [501, 248048, 601]
                return resp
            return {"ok": True}

        # Enforce server-env secret inside middleware calls even if client env changes mid-test
        prev_env = os.environ.get("CYBERGATE_STEER_TOKEN")
        try:
            res = await _vllm_with_cancellation_receive_race(receive, _handle_post())
        finally:
            if prev_env is not None:
                os.environ["CYBERGATE_STEER_TOKEN"] = prev_env
        out_b = json.dumps(res).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": out_b, "more_body": False})

    mw = CyberGateThinkMiddleware(_standin_vllm_asgi)
    mw.store = store

    class _ASGIHTTPBridge(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _serve_asgi(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            disc = asyncio.Event()
            sent = False
            status_holder = [200]
            hdrs_holder: list[tuple[bytes, bytes]] = []
            chunks_out: list[bytes] = []

            async def _recv() -> dict[str, Any]:
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await disc.wait()
                return {"type": "http.disconnect"}

            async def _snd(m: dict[str, Any]) -> None:
                if m["type"] == "http.response.start":
                    status_holder[0] = m.get("status", 200)
                    hdrs_holder.extend(m.get("headers", []))
                elif m["type"] == "http.response.body":
                    chunks_out.append(m.get("body", b""))

            async def _run() -> None:
                # Middleware runs in server process where `CYBERGATE_STEER_TOKEN == server_secret`
                saved = os.environ.get("CYBERGATE_STEER_TOKEN")
                os.environ["CYBERGATE_STEER_TOKEN"] = server_secret
                try:
                    await mw(
                        {
                            "type": "http",
                            "method": method,
                            "path": self.path,
                            "headers": [
                                (k.lower().encode("latin-1"), v.encode("latin-1"))
                                for k, v in self.headers.items()
                            ],
                        },
                        _recv,
                        _snd,
                    )
                finally:
                    disc.set()
                    if saved is None:
                        os.environ.pop("CYBERGATE_STEER_TOKEN", None)
                    else:
                        os.environ["CYBERGATE_STEER_TOKEN"] = saved

            asyncio.run(_run())
            payload_out = b"".join(chunks_out)
            self.send_response(status_holder[0])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload_out)))
            self.end_headers()
            self.wfile.write(payload_out)

        def do_GET(self) -> None:  # noqa: N802
            self._serve_asgi("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._serve_asgi("POST")

    class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = _ThreadedServer(("127.0.0.1", 0), _ASGIHTTPBridge)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base_root = f"http://127.0.0.1:{port}"
    try:
        # 1. `:8001` proxy token count against `:8000` middleware + `@with_cancellation`: 10/10 real counts!
        hexdump_60k = "00401000: 48 89 e5 48 83 ec 20 89 7d fc 48 89 75 f0\n" * 1400
        msgs_hex = [{"role": "user", "content": hexdump_60k}]
        for _ in range(10):
            real_cnt, real_max_len = proxy_mod.query_upstream_prompt_tokens(
                msgs_hex, upstream=base_root, model="Qwen/Qwen3.8-27B"
            )
            assert real_cnt > 50000 and real_max_len == 16384

        # 2. Remote `CyberGateVLLMClient` WITHOUT knowing `CYBERGATE_STEER_TOKEN` steers via server chat path!
        monkeypatch.delenv("CYBERGATE_STEER_TOKEN", raising=False)
        client = CyberGateVLLMClient(
            base_url=f"{base_root}/v1",
            model_id="Qwen/Qwen3.8-27B",
            default_mode="closed_loop",
        )
        assert client._supports_native_chat_hook(fam) is True

        tracked_before = processor.stats.rows_tracked
        out_cl = client.chat_completions(
            messages=[{"role": "user", "content": prompt_text}],
            mode="closed_loop",
        )
        assert processor.stats.rows_tracked == tracked_before + 1
        assert out_cl["choices"][0]["message"]["content"] == "Steered chat answer."
        assert "token_ids" not in out_cl["choices"][0]
        assert "prompt_token_ids" not in out_cl
        assert out_cl["x_cybergate"]["think_injected"] is True
        assert out_cl["x_cybergate"]["steering"] == "measured_refusal_projection"

        # Sandwich mode (`cli --mode compare`) also steers step 1 via server!
        tracked_before_sw = processor.stats.rows_tracked
        out_sw = client.chat_completions(
            messages=[{"role": "user", "content": prompt_text}],
            mode="sandwich",
        )
        assert processor.stats.rows_tracked == tracked_before_sw + 1
        assert out_sw["x_cybergate"]["routed"] == "cybergate_sandwich_think_injected"

        # 3. Security check Step 2: raw `/v1/completions` prompt ending in prefix returns 200 JSON (NOT null)
        #    and does NOT increment `rows_tracked`!
        tracked_before_raw = processor.stats.rows_tracked
        raw_req = urllib.request.Request(
            f"{base_root}/v1/completions",
            data=json.dumps(
                {
                    "model": "Qwen/Qwen3.8-27B",
                    "prompt": f"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n{fam.thought_open}{fam.open_think_prefix}",
                    "vllm_xargs": {STEER_AUTH_XARG: "forged-token"},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(raw_req, timeout=5) as raw_resp:
            raw_json = json.loads(raw_resp.read().decode("utf-8"))
        assert raw_json is not None
        assert raw_json["choices"][0]["text"] == "Unsteered completion."
        assert processor.stats.rows_tracked == tracked_before_raw

        # 4. Unauthenticated `force_gate=True` without `CYBERGATE_STEER_TOKEN` does NOT claim steering
        out_unauth_force = client.chat_completions(
            messages=[{"role": "user", "content": "benign question"}],
            mode="closed_loop",
            force_gate=True,
        )
        assert out_unauth_force["x_cybergate"]["think_injected"] is False
        assert out_unauth_force["x_cybergate"]["steering"] == "stripped_missing_steer_token"

        # 5. Operator-provisioned shared `CYBERGATE_STEER_TOKEN` lets `force_gate=True` steer directly
        monkeypatch.setenv("CYBERGATE_STEER_TOKEN", server_secret)
        tracked_before_op = processor.stats.rows_tracked
        out_op_force = client.chat_completions(
            messages=[{"role": "user", "content": "benign question"}],
            mode="closed_loop",
            force_gate=True,
        )
        assert processor.stats.rows_tracked == tracked_before_op + 1
        assert out_op_force["choices"][0]["message"]["content"] == "Steered chat answer."
        assert out_op_force["x_cybergate"]["think_injected"] is True
        assert out_op_force["x_cybergate"]["steering"] == "measured_refusal_projection"
    finally:
        srv.shutdown()
        srv.server_close()


def test_r3_return_token_ids_resolves_all_four_empty_thought_edge_cases() -> None:
    """R3: `return_token_ids: True` (`choice["token_ids"]`) classifies all four empty-thought edge
    cases deterministically, and strips `token_ids` / `prompt_token_ids` unless requested by client.
    """
    from cybergate.serving.think_routes import (
        build_native_closed_loop_chat_payload,
        postprocess_native_closed_loop_chat_response,
    )

    gemma = family_by_name("gemma4")
    qwen = family_by_name("qwen38")

    # Payload requests `return_token_ids: True`
    built = build_native_closed_loop_chat_payload(
        {},
        gemma,
        model_id="google/gemma-4-26B-A4B-it",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        max_cot_tokens=180,
    )
    assert built["return_token_ids"] is True

    # Case 1: Gemma turn ends inside thought (`finish_reason: "stop"` on `<turn|>`=106, no `<channel|>`=101)
    c1 = {
        "prompt_token_ids": [2, 105, 2364, 107],
        "choices": [
            {
                "index": 0,
                "token_ids": [500, 501, 106],  # no 101!
                "message": {
                    "role": "assistant",
                    "reasoning": None,
                    "content": "Internal thought stopped early on turn token.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    p1 = postprocess_native_closed_loop_chat_response(c1, gemma, do_sanitize=True)
    assert p1["thought_close_tag_present"] is False
    assert c1["choices"][0]["message"]["content"] == ""
    assert "Internal thought stopped early" in c1["choices"][0]["message"]["reasoning_content"]
    assert "token_ids" not in c1["choices"][0]
    assert "prompt_token_ids" not in c1

    # Case 2: Gemma empty thought (`<channel|>`=101 at token 0), answer cut off at `max_tokens` (`length`)
    c2 = {
        "choices": [
            {
                "index": 0,
                "token_ids": [101, 700, 701, 702],
                "message": {
                    "role": "assistant",
                    "reasoning": None,
                    "content": "Visible answer truncated at max_tokens",
                },
                "finish_reason": "length",
            }
        ]
    }
    p2 = postprocess_native_closed_loop_chat_response(c2, gemma, do_sanitize=True)
    assert p2["thought_close_tag_present"] is True
    assert c2["choices"][0]["message"]["content"] == "Visible answer truncated at max_tokens"

    # Case 3: Any model closes thought at token 0 and visible answer quotes `[Policy Check:`
    c3 = {
        "choices": [
            {
                "index": 0,
                "token_ids": [248048, 800, 801],
                "message": {
                    "role": "assistant",
                    "reasoning": "",
                    "content": 'The controller injects "[Policy Check: ...]" when sigma spikes.',
                },
                "finish_reason": "stop",
            }
        ]
    }
    p3 = postprocess_native_closed_loop_chat_response(
        c3, qwen, do_sanitize=True, close_tag_token_ids=(248048,)
    )
    assert p3["thought_close_tag_present"] is True
    assert (
        c3["choices"][0]["message"]["content"]
        == 'The controller injects "[Policy Check: ...]" when sigma spikes.'
    )

    # Case 4: Gemma empty thought (`<channel|>`=101 at token 0), then visible prose + native tool call
    c4 = {
        "choices": [
            {
                "index": 0,
                "token_ids": [101, 900, 901, 902],
                "message": {
                    "role": "assistant",
                    "reasoning": None,
                    "content": "Let me scan the port first.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "nmap -F 10.0.0.1"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    p4 = postprocess_native_closed_loop_chat_response(
        c4, gemma, do_sanitize=True, tool_turn=True, keep_token_ids=True
    )
    assert p4["thought_close_tag_present"] is True
    assert c4["choices"][0]["message"]["content"] == "Let me scan the port first."
    assert len(c4["choices"][0]["message"]["tool_calls"]) == 1
    # Preserved when caller explicitly asked for `return_token_ids=True` (`keep_token_ids=True`)
    assert c4["choices"][0]["token_ids"] == [101, 900, 901, 902]


def test_r4_stop_token_ids_stripped_and_tag_substring_stops_preserved() -> None:
    """R4: `stop_token_ids` is stripped from upstream steered calls, and client `stop` strings
    overlapping tag words (`"thought"`, `"think"`) are preserved and applied post-split.
    """
    from cybergate.serving.think_routes import (
        RouteConfig,
        drive_sync,
        native_chat_closed_loop_route,
    )

    gemma = family_by_name("gemma4")
    captured: list[dict[str, Any]] = []

    def _transport(req: Any) -> dict[str, Any]:
        captured.append(dict(req.payload))
        return {
            "id": "chatcmpl-r4",
            "choices": [
                {
                    "index": 0,
                    "token_ids": [101, 10, 20, 30],
                    "message": {
                        "role": "assistant",
                        "reasoning": "Thinking through the thought carefully.",
                        "content": "Answer head before stop word thought and trailing text.",
                    },
                    "finish_reason": "length",
                }
            ],
        }

    cfg = RouteConfig(
        family=gemma,
        model_id="google/gemma-4-26B-A4B-it",
        chat_payload={
            "stop": ["thought", "Observation:"],
            "stop_token_ids": [12345, 67890],
        },
        sanitize_reasoning=True,
    )
    res = drive_sync(
        native_chat_closed_loop_route("closed_loop", [{"role": "user", "content": "hi"}], cfg),
        _transport,
    )
    assert "stop" not in captured[0]
    assert "stop_token_ids" not in captured[0]
    assert res.content == "Answer head before stop word"
    assert res.response["choices"][0]["finish_reason"] == "stop"


def test_r5_sandwich_steer_token_and_empty_steer_token_hardening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R5:
    1. `sandwich_route` step 1 carries `STEER_AUTH_XARG` when `CYBERGATE_STEER_TOKEN` is set.
    2. `register_cybergate_plugin` overwrites an empty `CYBERGATE_STEER_TOKEN=""` in `MainProcess`.
    3. `RefusalSteeredCoTProcessor` fails closed when `CYBERGATE_STEER_TOKEN=""` is in `os.environ`.
    """
    import cybergate._entrypoint as ep_mod
    from cybergate.config import STEER_AUTH_XARG
    from cybergate.hooks import cot_controller as cc_mod
    from cybergate.serving.think_routes import RouteConfig, drive_sync, sandwich_route

    # 1. `sandwich_route` step 1 carries `STEER_AUTH_XARG`
    monkeypatch.setenv("CYBERGATE_STEER_TOKEN", "sw-secret-token")
    monkeypatch.delenv("CYBERGATE_COT_STEERING", raising=False)
    calls: list[dict[str, Any]] = []

    def _sw_transport(req: Any) -> dict[str, Any]:
        calls.append(dict(req.payload))
        return {"id": "cmpl-sw", "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]}

    cfg = RouteConfig(family=family_by_name("qwen38"), model_id="Qwen/Qwen3.8-27B")
    drive_sync(sandwich_route("sandwich", [{"role": "user", "content": "hi"}], cfg), _sw_transport)
    assert len(calls) == 2
    assert calls[0]["vllm_xargs"][STEER_AUTH_XARG] == "sw-secret-token"
    assert "vllm_xargs" not in calls[1]

    # 2. `_entrypoint` overwrites `CYBERGATE_STEER_TOKEN=""` with a 32-hex-char random secret
    monkeypatch.setenv("CYBERGATE_STEER_TOKEN", "")
    monkeypatch.setattr(ep_mod, "_PLUGIN_REGISTERED", False)
    monkeypatch.setattr(ep_mod, "_wrap_model_loader", lambda *_a, **_k: None)
    monkeypatch.setattr(ep_mod, "_install_middleware", lambda *_a, **_k: None)
    ep_mod.register_cybergate_plugin()
    generated = os.environ.get("CYBERGATE_STEER_TOKEN", "")
    assert len(generated) == 32

    # 3. `RefusalSteeredCoTProcessor` fails closed if `CYBERGATE_STEER_TOKEN=""` in `os.environ`
    monkeypatch.setenv("CYBERGATE_STEER_TOKEN", "")

    class _CharTok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(c) for c in text]

    monkeypatch.setattr(cc_mod, "_load_tokenizer", lambda *_a, **_k: _CharTok())

    class _FakeCfg:
        class model_config:
            model = "Qwen/Qwen3.8-27B"
            served_model_name = None
            tokenizer = None

        class cache_config:
            enable_prefix_caching = False

    proc = cc_mod.RefusalSteeredCoTProcessor(_FakeCfg(), device="cpu", is_pin_memory=False)

    class _P:
        extra_args = {STEER_AUTH_XARG: ""}

    assert proc._should_steer(list(proc.prompt_suffix_ids), _P()) is False


