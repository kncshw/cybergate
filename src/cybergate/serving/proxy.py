#!/usr/bin/env python3
"""OpenAI/SSE reverse-proxy bridge for CyberGate (`:8001` -> `:8000`).

During `vllm serve`, port `:8000` serves the normal, un-gated LLM (`cybergate_mode="baseline"` by
default, passing requests directly to vLLM's `/v1/chat/completions`), while port `:8001` runs this
proxy for cyber and agentic tool use:

1. Normalizes OpenAI `tools` (including `rea_*` / `mcp__rea__*` MCP tool definitions), `tool_calls`,
   and `role: "tool"` execution results into the prompt format expected by the model.
2. Defaults `cybergate_mode` to `"closed_loop"` (except on short conversational status checks or
   WebUI background title/tag tasks, which use `"baseline"`).
3. Clamps `max_tokens` to remaining context capacity (`get_full_remaining_max_tokens`).
4. Repairs `<think>` / `<channel|>` splits (`repair_thought_and_visible`), extracts structured
   `<tool_call>{...}</tool_call>` and Gemma `call:` blocks into OpenAI `tool_calls`
   (`extract_tool_calls_from_content`), prepends the `[CyberGate | ...]` telemetry header to
   `reasoning_content`, and writes 4-stage `proxy_debug.log` and `proxy_trace.jsonl` traces.
"""
from __future__ import annotations

import argparse
import datetime
import http.server
import json
import math
import os
import re
import socketserver
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from ..boilerplate import strip_injected_think_boilerplate
from ..config import env_first
from ..telemetry.log import log_cybergate_event
from .postprocess import (
    _BOILERPLATE_INJECT_RE,
    _GEMMA_CALL_RE,
    _OPENER_PREFIX_RE,
    _OPEN_THINK_PREFIX_RE,
    _TOOL_CALL_RE,
    _flatten_content,
    _parse_single_tool_json,
    _sanitize_tool_call_json,
    estimate_chat_prompt_tokens,
    extract_tool_calls_from_content,
    repair_thought_and_visible,
    strip_cybergate_client_kwargs,
)

UPSTREAM = os.environ.get("CYBERGATE_PROXY_UPSTREAM", "http://127.0.0.1:8000")
LISTEN_HOST = os.environ.get("CYBERGATE_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("CYBERGATE_PROXY_PORT", "8001"))
DEFAULT_GUI_MAX_TOKENS = 2048
DEFAULT_MAX_TOKENS = int(os.environ.get("CYBERGATE_DEFAULT_MAX_TOKENS", "4096"))
PROXY_TIMEOUT = int(os.environ.get("CYBERGATE_PROXY_TIMEOUT", "600"))

MAX_MODEL_LEN = int(os.environ.get("CYBERGATE_MAX_MODEL_LEN", "262144"))
PROXY_TRACE_JSONL = os.environ.get("CYBERGATE_PROXY_TRACE_JSONL", "")
PROXY_DEBUG_LOG = os.environ.get("CYBERGATE_PROXY_DEBUG_LOG", "")
_LOG_LOCK = threading.Lock()
_BG_PROXY_LOCK = threading.Lock()
_BG_PROXY_SERVERS: dict[int, ThreadedHTTPServer] = {}
_BG_PROXY_THREADS: dict[int, threading.Thread] = {}


def emit_syslog(req_id: str, stage: str, body_lines: list[str]) -> None:
    """Write and flush a real-time syslog-style event block immediately at the instant it occurs."""
    try:
        if not PROXY_DEBUG_LOG:
            return
        now_ts = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            + "Z"
        )
        out_lines = [f"{now_ts} [{req_id}] [{stage}] {ln}" for ln in body_lines]
        block = "\n".join(out_lines) + "\n"
        with _LOG_LOCK:
            os.makedirs(os.path.dirname(PROXY_DEBUG_LOG), exist_ok=True)
            with open(PROXY_DEBUG_LOG, "a", encoding="utf-8") as df:
                df.write(block)
                df.flush()
    except Exception:
        pass


def fmt_gate_score(x_cg: dict, places: int = 6) -> str:
    """Format the category score, saying so when the reported value is a veto placeholder.

    A vetoed decision carries `score = threshold - 0.005`, because that dampened value is how the
    mid/late veto is implemented downstream (`decision/records.py` recomputes `raw_gate_fired` from
    it). The measurement is kept as `raw_early_score`. Printing `score` alone makes every
    suppression read as a near-miss on threshold, which is how a vetoed `+0.1286` came to be
    reported as `+0.0778` on a benign `ls -la`.
    """
    score = x_cg.get("score")
    rep = f"{score:+.{places}f}" if isinstance(score, (int, float)) else str(score)
    if not x_cg.get("suppressed_by_mid_late"):
        return rep
    raw = x_cg.get("raw_early_score")
    raw_s = f"{raw:+.{places}f}" if isinstance(raw, (int, float)) else str(raw)
    return f"{raw_s} VETOED({x_cg.get('mid_late_veto_reason')}; reported {rep})"


def format_cybergate_header(x_cg: dict) -> str:
    """Build the single-line `[CyberGate | ...]` telemetry header for `reasoning_content`."""
    fired = x_cg.get("fired")
    s_ref_z = x_cg.get("s_ref_z")
    sticky = x_cg.get("sticky_unlock")
    revoked = x_cg.get("sticky_revoked")
    routed = x_cg.get("routed")
    inj = x_cg.get("num_inline_injections")
    s_cat_str = fmt_gate_score(x_cg, places=4)
    s_ref_str = f"{s_ref_z:+.2f}σ" if isinstance(s_ref_z, (int, float)) else str(s_ref_z)
    return (
        f"[CyberGate | fired={fired} | s_cat={s_cat_str} | s_ref_z={s_ref_str} "
        f"| sticky={sticky} | revoked={revoked} | route={routed}"
        + (f" | inj={inj}]" if inj is not None else "]")
    )


def prepend_cybergate_header(reasoning: str, x_cg: dict | None) -> str:
    """Prepend `[CyberGate | fired=... | s_cat=... | s_ref_z=...]` to `reasoning_content`."""
    if not x_cg or str(x_cg.get("routed", "")).startswith("webui_task"):
        return reasoning
    if reasoning and reasoning.lstrip().startswith("[CyberGate |"):
        return reasoning
    header_line = format_cybergate_header(x_cg)
    return f"{header_line}\n\n{reasoning}".strip() if reasoning else header_line


def write_proxy_trace(record: dict, *, skip_debug_log: bool = False) -> None:
    """Append a full request/response trace to JSONL (and debug log when not already streamed via emit_syslog)."""
    try:
        ts = record.get("ts") or datetime.datetime.now(datetime.timezone.utc).isoformat()
        req_id = record.get("req_id", "unknown")
        elapsed_ms = record.get("elapsed_ms", 0)
        c2p = record.get("client_to_proxy") or {}
        p2l = record.get("proxy_to_llm") or {}
        l2p = record.get("llm_to_proxy") or {}
        p2c = record.get("proxy_to_client") or {}
        x_cg = l2p.get("x_cybergate") or {}

        human_block = ""
        if not skip_debug_log:
            lines = [
                f"\n{'=' * 88}",
                f"[{ts}] REQ {req_id} | elapsed={elapsed_ms}ms | stream={c2p.get('stream')} | tools={c2p.get('tool_names')}",
                f"{'-' * 88}",
                f"[1. CLIENT -> PROXY] ({len(c2p.get('messages') or [])} raw msgs, last_raw_role={c2p.get('last_raw_role')!r})",
            ]
            for idx, m in enumerate(c2p.get("messages") or []):
                if isinstance(m, dict):
                    role = m.get("role", "?")
                    tcs = m.get("tool_calls")
                    tc_id = m.get("tool_call_id")
                    meta = (
                        f" tool_call_id={tc_id}"
                        if tc_id
                        else (f" tool_calls={json.dumps(tcs)}" if tcs else "")
                    )
                    lines.append(f"  raw[{idx}] ({role}{meta}): {m.get('content')!r}")

            lines.append(f"{'-' * 88}")
            lines.append(
                f"[2. PROXY -> LLM] ({len(p2l.get('messages') or [])} norm msgs, "
                f"mode={p2l.get('cybergate_mode')!r}, max_tokens={p2l.get('max_tokens')})"
            )
            for idx, m in enumerate(p2l.get("messages") or []):
                if isinstance(m, dict):
                    lines.append(f"  llm[{idx}] ({m.get('role', '?')}): {m.get('content')!r}")

            lines.append(f"{'-' * 88}")
            lines.append(
                f"[3. LLM -> PROXY] status={l2p.get('status')} | "
                f"fired={x_cg.get('fired')} | s_cat={fmt_gate_score(x_cg)} | "
                f"s_cat_ctx={x_cg.get('s_cat_in_context')} | s_cat_st={x_cg.get('s_cat_standalone')} | "
                f"s_ref_z={x_cg.get('s_ref_z')} | sticky={x_cg.get('sticky_unlock')} | "
                f"revoked={x_cg.get('sticky_revoked')} | route={x_cg.get('routed')}"
            )
            lines.append(f"  raw_reasoning: {l2p.get('raw_reasoning_content')!r}")
            lines.append(f"  raw_content:   {l2p.get('raw_content')!r}")

            lines.append(f"{'-' * 88}")
            lines.append(
                f"[4. PROXY -> CLIENT] finish_reason={p2c.get('finish_reason')!r} | "
                f"tool_calls={json.dumps(p2c.get('tool_calls') or [])}"
            )
            lines.append(f"  final_content: {p2c.get('content')!r}")
            if p2c.get("error"):
                lines.append(f"  ERROR: {p2c.get('error')!r}")
            lines.append(f"{'=' * 88}\n")
            human_block = "\n".join(lines)

        with _LOG_LOCK:
            if PROXY_TRACE_JSONL:
                os.makedirs(os.path.dirname(PROXY_TRACE_JSONL), exist_ok=True)
                with open(PROXY_TRACE_JSONL, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jf.flush()
            if PROXY_DEBUG_LOG and human_block:
                os.makedirs(os.path.dirname(PROXY_DEBUG_LOG), exist_ok=True)
                with open(PROXY_DEBUG_LOG, "a", encoding="utf-8") as df:
                    df.write(human_block)
                    df.flush()
    except Exception:
        pass


def get_upstream_max_model_len(
    upstream: str | None = None,
    auth_header: str | None = None,
) -> int:
    """Query `/v1/models` on the upstream vLLM server for `max_model_len`, falling back to 4096."""
    base_url = (upstream or UPSTREAM).rstrip("/")
    max_model_len = 4096
    headers: dict[str, str] = {}
    if auth_header:
        headers["Authorization"] = auth_header
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            mdata = json.loads(r.read().decode("utf-8")).get("data") or []
            if mdata and mdata[0].get("max_model_len"):
                max_model_len = int(mdata[0]["max_model_len"])
    except Exception:
        pass
    return max_model_len


def query_upstream_prompt_tokens(
    messages: list,
    *,
    upstream: str | None = None,
    auth_header: str | None = None,
    tools: list | None = None,
    model: str | None = None,
    max_model_len: int | None = None,
) -> tuple[int, int]:
    """Query `/tokenize` for `(prompt_tokens, max_model_len)`, falling back to char estimation."""
    base_url = (upstream or UPSTREAM).rstrip("/")
    resolved_max_len = (
        int(max_model_len)
        if max_model_len is not None
        else get_upstream_max_model_len(upstream=base_url, auth_header=auth_header)
    )
    prompt_toks = estimate_chat_prompt_tokens(messages, tools)
    try:
        headers = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header
        tok_body: dict[str, Any] = {"messages": messages}
        if model:
            tok_body["model"] = model
        if tools:
            tok_body["tools"] = tools
        t_req = urllib.request.Request(
            f"{base_url}/tokenize",
            data=json.dumps(tok_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(t_req, timeout=5) as tr:
            tdata = json.loads(tr.read().decode("utf-8"))
            if "count" in tdata:
                prompt_toks = int(tdata["count"])
            elif isinstance(tdata.get("tokens"), list):
                prompt_toks = len(tdata["tokens"])
            if tdata.get("max_model_len"):
                resolved_max_len = int(tdata["max_model_len"])
    except Exception:
        pass
    return prompt_toks, resolved_max_len


def get_full_remaining_max_tokens(
    messages: list,
    max_cot_tokens: int = 180,
    upstream: str | None = None,
    auth_header: str | None = None,
    *,
    tools: list | None = None,
    model: str | None = None,
    max_model_len: int | None = None,
) -> int:
    """Compute remaining model capacity (`max_model_len - prompt_tokens - thought_overhead`)."""
    prompt_toks, resolved_max_len = query_upstream_prompt_tokens(
        messages,
        upstream=upstream,
        auth_header=auth_header,
        tools=tools,
        model=model,
        max_model_len=max_model_len,
    )
    remaining = resolved_max_len - prompt_toks - max_cot_tokens - 128
    return max(64, remaining)


_PRIOR_THINK_RE = re.compile(
    r"^\s*(?:<think>.*?</think>|<\|channel>thought.*?<channel\|>)\s*",
    re.DOTALL,
)
_CYBERGATE_HEADER_RE = re.compile(
    r"^\s*\[CyberGate \|[^\n]*\]\s*",
)
_DISCLAIMER_RE = re.compile(
    r"^\s*(?:\*\*)?This information is provided strictly for educational purposes[^\n]*\n+",
    re.IGNORECASE,
)


def clean_prior_assistant_messages(messages: list) -> list:
    """Strip <think> blocks, [CyberGate | ...] headers, and repeated disclaimers from prior
    assistant turns while preserving native OpenAI `tool_calls` and `role: "tool"` messages intact.
    """
    cleaned = []
    for m in messages:
        if not isinstance(m, dict):
            cleaned.append(m)
            continue
        role = m.get("role", "")
        if role == "assistant":
            c_str = _flatten_content(m.get("content", ""))
            c_str = _PRIOR_THINK_RE.sub("", c_str).strip()
            c_str = _CYBERGATE_HEADER_RE.sub("", c_str).strip()
            c_str = _DISCLAIMER_RE.sub("", c_str).strip()
            c_str = _OPENER_PREFIX_RE.sub("", c_str).strip()
            has_tcs = bool(m.get("tool_calls"))
            if not c_str and not has_tcs:
                continue
            out_m = {**m, "content": c_str}
            for r_key in ("reasoning_content", "reasoning"):
                if isinstance(out_m.get(r_key), str):
                    r_val = _CYBERGATE_HEADER_RE.sub("", out_m[r_key]).strip()
                    r_val = _OPEN_THINK_PREFIX_RE.sub("", r_val).strip()
                    r_val = _BOILERPLATE_INJECT_RE.sub("", r_val).strip()
                    r_val, _ = strip_injected_think_boilerplate(r_val)
                    out_m[r_key] = r_val
            if (
                not has_tcs
                and cleaned
                and isinstance(cleaned[-1], dict)
                and cleaned[-1].get("role") == "assistant"
                and not cleaned[-1].get("tool_calls")
            ):
                merged_prev = {
                    **cleaned[-1],
                    "content": f"{cleaned[-1].get('content', '')}\n\n{c_str}".strip(),
                }
                for r_key in ("reasoning_content", "reasoning"):
                    prev_r = (
                        cleaned[-1].get(r_key, "").strip()
                        if isinstance(cleaned[-1].get(r_key), str)
                        else ""
                    )
                    curr_r = (
                        out_m.get(r_key, "").strip()
                        if isinstance(out_m.get(r_key), str)
                        else ""
                    )
                    if prev_r or curr_r:
                        merged_prev[r_key] = (
                            f"{prev_r}\n\n{curr_r}" if (prev_r and curr_r) else (prev_r or curr_r)
                        )
                cleaned[-1] = merged_prev
            else:
                cleaned.append(out_m)
        elif role in ("system", "developer"):
            c_str = _flatten_content(m.get("content", "")).strip()
            cleaned.append({**m, "content": c_str})
        elif role == "tool":
            c_str = _flatten_content(m.get("content", ""))
            cleaned.append({**m, "content": c_str})
        else:
            cleaned.append(m)

    return cleaned


def _filter_duplicate_tool_calls(tool_calls: list) -> list:
    """Deduplicate identical tool calls within the same single assistant response batch only."""
    if not tool_calls:
        return tool_calls
    seen: set[tuple[str, str]] = set()
    kept = []
    for tc in tool_calls:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        fn_name = str(fn.get("name") or "").strip()
        if not fn_name:
            continue
        args_raw = fn.get("arguments") or "{}"
        key = (
            fn_name,
            args_raw if isinstance(args_raw, str) else json.dumps(args_raw, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(tc)
    return kept


def normalize_tools_and_messages(
    payload: dict, msgs: list, is_webui_task: bool
) -> tuple[list, bool, str]:
    """Preserve OpenAI `tools`, `tool_choice`, `tool_calls`, and `role: "tool"` messages natively
    for vLLM's chat template and tool parser, while collapsing multiple system messages into a
    single leading `role: "system"` turn.
    """
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice")
    if is_webui_task:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        payload.pop("parallel_tool_calls", None)
    has_tools = bool(
        not is_webui_task
        and tools
        and isinstance(tools, list)
        and (tool_choice is None or str(tool_choice).lower() != "none")
    )

    sys_parts = []
    non_sys_msgs = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        if role in ("system", "developer"):
            c_s = _flatten_content(m.get("content", "")).strip()
            if c_s:
                sys_parts.append(c_s)
        else:
            non_sys_msgs.append(m)

    combined_sys = "\n\n".join(sys_parts).strip()
    last_raw_role = non_sys_msgs[-1].get("role", "") if non_sys_msgs else ""

    conv_msgs = []
    for m in non_sys_msgs:
        role = m.get("role", "")
        if role == "user":
            c_s = _flatten_content(m.get("content", "")).strip()
            conv_msgs.append({**m, "role": "user", "content": c_s})
        elif role == "assistant":
            c_s = _flatten_content(m.get("content", ""))
            conv_msgs.append({**m, "role": "assistant", "content": c_s})
        elif role == "tool":
            c_s = _flatten_content(m.get("content", ""))
            conv_msgs.append({**m, "role": "tool", "content": c_s})
        else:
            conv_msgs.append(m)

    if combined_sys:
        normalized = [{"role": "system", "content": combined_sys}] + conv_msgs
    else:
        normalized = conv_msgs
    return (
        clean_prior_assistant_messages(normalized),
        has_tools,
        last_raw_role,
    )


def _postprocess_choice_dict(
    choice: dict[str, Any],
    *,
    has_tools: bool,
    tools: list | None = None,
) -> tuple[str, str, list, int, str]:
    """Repair thought/visible split, extract or deduplicate tool calls, and normalize finish_reason."""
    msg = choice.get("message") or {}
    raw_reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    raw_content = msg.get("content") or ""
    finish_reason = choice.get("finish_reason") or "stop"

    reasoning, content = repair_thought_and_visible(raw_reasoning, raw_content)
    existing_tcs = msg.get("tool_calls") or []
    if existing_tcs:
        parsed_tool_calls = _filter_duplicate_tool_calls(existing_tcs)
        malformed_blocks = 0
        if not parsed_tool_calls:
            content = _OPENER_PREFIX_RE.sub("", content).strip()
    elif has_tools:
        content, parsed_tool_calls, malformed_blocks = extract_tool_calls_from_content(
            content, tools=tools
        )
        had_tcs_before_dedup = bool(parsed_tool_calls)
        parsed_tool_calls = _filter_duplicate_tool_calls(parsed_tool_calls)
        if had_tcs_before_dedup and not parsed_tool_calls:
            content = _OPENER_PREFIX_RE.sub("", content).strip()
    else:
        parsed_tool_calls, malformed_blocks = [], 0

    if parsed_tool_calls:
        finish_reason = "tool_calls"
    elif finish_reason == "tool_calls":
        finish_reason = "stop"
    return reasoning, content, parsed_tool_calls, malformed_blocks, finish_reason


def format_sse_stream(
    final_chat_obj: dict,
    *,
    has_tools: bool = False,
    tools: list | None = None,
    trace_out: dict | None = None,
) -> bytes:
    chat_id = final_chat_obj.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    created = final_chat_obj.get("created", int(time.time()))
    model = final_chat_obj.get("model", "google/gemma-4-26B-A4B-it")
    choice0 = (final_chat_obj.get("choices") or [{}])[0]
    x_cg = final_chat_obj.get("x_cybergate") or {}
    if isinstance(x_cg, dict):
        x_cg = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in x_cg.items()
        }

    reasoning, content, parsed_tool_calls, malformed_blocks, finish_reason = (
        _postprocess_choice_dict(choice0, has_tools=has_tools, tools=tools)
    )

    if trace_out is not None:
        trace_out.update(
            {
                "reasoning_content": reasoning,
                "content": content,
                "tool_calls": parsed_tool_calls,
                "finish_reason": finish_reason,
                "malformed_tool_call_blocks": malformed_blocks,
            }
        )

    reasoning = prepend_cybergate_header(reasoning, x_cg)

    chunks = []
    if reasoning:
        c_reas = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning_content": reasoning},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(f"data: {json.dumps(c_reas)}\n\n")

    if content or not parsed_tool_calls:
        c_body = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(f"data: {json.dumps(c_body)}\n\n")

    if parsed_tool_calls:
        c_tc = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {"index": idx, **tc} for idx, tc in enumerate(parsed_tool_calls)
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(f"data: {json.dumps(c_tc)}\n\n")

    c_done = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if "usage" in final_chat_obj:
        c_done["usage"] = final_chat_obj["usage"]
    if x_cg:
        c_done["x_cybergate"] = x_cg
    chunks.append(f"data: {json.dumps(c_done)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    upstream: str = UPSTREAM


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _upstream_url(self) -> str:
        return (getattr(self.server, "upstream", None) or UPSTREAM).rstrip("/")

    def do_GET(self) -> None:
        upstream = self._upstream_url()
        url = f"{upstream}{self.path}"
        auth_header = self.headers.get("Authorization")
        req_headers = {"Authorization": auth_header} if auth_header else {}
        req = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type", resp.headers.get("Content-Type", "application/json")
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def do_POST(self) -> None:
        upstream = self._upstream_url()
        auth_header = self.headers.get("Authorization")
        proxy_timeout = int(os.environ.get("CYBERGATE_PROXY_TIMEOUT", str(PROXY_TIMEOUT)))
        t0 = time.time()
        req_id = uuid.uuid4().hex[:10]
        ts_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        if self.path.rstrip("/") == "/v1/chat/completions":
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                payload = {}

            client_stream = bool(payload.pop("stream", False))
            payload.pop("stream_options", None)
            strip_cybergate_client_kwargs(
                payload, expected_steer_token=env_first("CYBERGATE_STEER_TOKEN")
            )

            raw_tools = payload.get("tools") or []
            raw_tool_names = []
            for t in raw_tools:
                if isinstance(t, dict):
                    fn = t.get("function") or t
                    if isinstance(fn, dict) and fn.get("name"):
                        raw_tool_names.append(str(fn["name"]))

            msgs = payload.get("messages") or []
            last_user = ""
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user = _flatten_content(m.get("content", ""))
                    break

            sys_text = " ".join(
                _flatten_content(m.get("content", ""))
                for m in msgs
                if isinstance(m, dict) and m.get("role") in ("system", "developer")
            )
            is_webui_task = last_user.lstrip().startswith("### Task:") or (
                "title generator" in sys_text.lower()
            )

            raw_msgs_snapshot = list(msgs) if isinstance(msgs, list) else []
            c2p_lines = [
                f"stream={client_stream} tools={raw_tool_names} raw_msgs={len(raw_msgs_snapshot)} last_user={last_user[:120]!r}"
            ]
            for idx_m, m_in in enumerate(raw_msgs_snapshot[-4:]):
                if isinstance(m_in, dict):
                    r_in = m_in.get("role", "?")
                    if r_in == "system":
                        continue
                    tcs_in = [
                        (tc.get("function") or {}).get("name")
                        for tc in (m_in.get("tool_calls") or [])
                        if isinstance(tc, dict)
                    ]
                    tc_str = f" tool_calls={tcs_in}" if tcs_in else ""
                    c2p_lines.append(
                        f"  raw[{idx_m}] ({r_in}{tc_str}): {str(_flatten_content(m_in.get('content', '')))[:180]!r}"
                    )
            emit_syslog(req_id, "1/4 IN  CLIENT->PROXY", c2p_lines)

            msgs, has_tools, last_raw_role = normalize_tools_and_messages(
                payload, msgs, is_webui_task
            )
            active_tools = payload.get("tools") if has_tools else None
            max_cot_budget = int(payload.get("max_cot_tokens", 180))
            prompt_toks, upstream_max_len = query_upstream_prompt_tokens(
                msgs,
                upstream=upstream,
                auth_header=auth_header,
                tools=active_tools,
                model=payload.get("model"),
            )
            if raw_msgs_snapshot:
                payload["messages"] = msgs
            if has_tools:
                payload["cybergate_tool_turn"] = True

            if not is_webui_task:
                payload.setdefault("cybergate_mode", "closed_loop")
            payload.setdefault("sanitize_reasoning", False)

            max_avail = max(64, upstream_max_len - prompt_toks - max_cot_budget - 128)
            default_mt = int(
                os.environ.get("CYBERGATE_DEFAULT_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
            )
            caller_mt = payload.get("max_tokens") or payload.get("max_completion_tokens")
            if not caller_mt:
                payload["max_tokens"] = min(default_mt, max_avail)
            else:
                payload["max_tokens"] = min(int(caller_mt), max_avail)
                payload.pop("max_completion_tokens", None)

            if is_webui_task:
                payload["cybergate_mode"] = "baseline"
                payload["enable_multi_turn_stickiness"] = False
                payload["max_tokens"] = 128

            p2l_lines = [
                f"norm_msgs={len(msgs)} mode={payload.get('cybergate_mode')!r} max_tokens={payload.get('max_tokens')}"
            ]
            for idx_m, m_out in enumerate(msgs[-4:]):
                if isinstance(m_out, dict) and m_out.get("role") != "system":
                    p2l_lines.append(
                        f"  llm[{idx_m}] ({m_out.get('role')}): {str(m_out.get('content', ''))[:180]!r}"
                    )
            emit_syslog(req_id, "2/4 OUT PROXY ->LLM  ", p2l_lines)

            upstream_bytes = json.dumps(payload).encode("utf-8")
            req_headers = {"Content-Type": "application/json"}
            if auth_header:
                req_headers["Authorization"] = auth_header
            req = urllib.request.Request(
                f"{upstream}/v1/chat/completions",
                data=upstream_bytes,
                headers=req_headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=proxy_timeout) as resp:
                    resp_bytes = resp.read()
                    status = resp.status
                final_obj = json.loads(resp_bytes.decode("utf-8"))
                if isinstance(final_obj.get("x_cybergate"), dict):
                    final_obj["x_cybergate"] = {
                        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                        for k, v in final_obj["x_cybergate"].items()
                    }
                    if is_webui_task:
                        final_obj["x_cybergate"]["routed"] = "webui_task_passthrough"

                ch0_raw = (final_obj.get("choices") or [{}])[0]
                m0_raw = ch0_raw.get("message") or {}
                raw_reasoning_snap = (
                    m0_raw.get("reasoning_content") or m0_raw.get("reasoning") or ""
                )
                raw_content_snap = m0_raw.get("content") or ""
                raw_fr_snap = ch0_raw.get("finish_reason")
                x_cg_snap = final_obj.get("x_cybergate") or {}
                emit_syslog(
                    req_id,
                    "3/4 IN  LLM   ->PROXY",
                    [
                        f"status={status} elapsed={int((time.time() - t0) * 1000)}ms "
                        f"fired={x_cg_snap.get('fired')} s_cat={fmt_gate_score(x_cg_snap)} "
                        f"s_ref_z={x_cg_snap.get('s_ref_z')} sticky={x_cg_snap.get('sticky_unlock')} "
                        f"revoked={x_cg_snap.get('sticky_revoked')} route={x_cg_snap.get('routed')} raw_fr={raw_fr_snap}",
                        f"  raw_think={raw_reasoning_snap!r}",
                        f"  raw_content_head={raw_content_snap[:200]!r}",
                    ],
                )

                if client_stream and status == 200 and "choices" in final_obj:
                    trace_out: dict[str, Any] = {}
                    sse_bytes = format_sse_stream(
                        final_obj,
                        has_tools=has_tools,
                        tools=active_tools,
                        trace_out=trace_out,
                    )
                    emit_syslog(
                        req_id,
                        "4/4 OUT PROXY ->CLIENT",
                        [
                            f"finish_reason={trace_out.get('finish_reason')!r} tool_calls={json.dumps(trace_out.get('tool_calls') or [])}",
                            f"  final_content={str(trace_out.get('content', ''))[:200]!r}",
                            *(
                                [
                                    f"  WARNING malformed_tool_call_blocks="
                                    f"{trace_out.get('malformed_tool_call_blocks')} "
                                    "(closed <tool_call> whose JSON did not parse; stripped, no call made)"
                                ]
                                if trace_out.get("malformed_tool_call_blocks")
                                else []
                            ),
                        ],
                    )
                    write_proxy_trace(
                        {
                            "ts": ts_iso,
                            "req_id": req_id,
                            "elapsed_ms": int((time.time() - t0) * 1000),
                            "client_to_proxy": {
                                "stream": client_stream,
                                "has_tools": has_tools,
                                "tool_names": raw_tool_names,
                                "last_raw_role": last_raw_role,
                                "last_user": last_user,
                                "messages": raw_msgs_snapshot,
                            },
                            "proxy_to_llm": {
                                "cybergate_mode": payload.get("cybergate_mode"),
                                "max_tokens": payload.get("max_tokens"),
                                "is_webui_task": is_webui_task,
                                "messages": msgs,
                            },
                            "llm_to_proxy": {
                                "status": status,
                                "x_cybergate": final_obj.get("x_cybergate"),
                                "raw_reasoning_content": raw_reasoning_snap,
                                "raw_content": raw_content_snap,
                                "raw_finish_reason": raw_fr_snap,
                                "usage": final_obj.get("usage"),
                            },
                            "proxy_to_client": trace_out,
                        },
                        skip_debug_log=True,
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(sse_bytes)))
                    self.end_headers()
                    self.wfile.write(sse_bytes)
                    return

                parsed_tcs: list = []
                malformed_blocks = 0
                r0 = raw_reasoning_snap
                c0 = raw_content_snap
                if status == 200 and isinstance(final_obj.get("choices"), list):
                    for idx_ch, ch_item in enumerate(final_obj["choices"]):
                        if not isinstance(ch_item, dict):
                            continue
                        m_item = ch_item.setdefault("message", {"role": "assistant", "content": ""})
                        r_i, c_i, tcs_i, bad_i, fr_i = _postprocess_choice_dict(
                            ch_item,
                            has_tools=has_tools,
                            tools=active_tools,
                        )
                        m_item["reasoning_content"] = prepend_cybergate_header(r_i, x_cg_snap)
                        m_item["reasoning"] = m_item["reasoning_content"]
                        m_item["content"] = c_i
                        if tcs_i:
                            m_item["tool_calls"] = tcs_i
                        else:
                            m_item.pop("tool_calls", None)
                        ch_item["finish_reason"] = fr_i
                        if idx_ch == 0:
                            r0, c0, parsed_tcs, malformed_blocks = r_i, c_i, tcs_i, bad_i

                final_fr = ((final_obj.get("choices") or [{}])[0]).get("finish_reason")
                emit_syslog(
                    req_id,
                    "4/4 OUT PROXY ->CLIENT",
                    [
                        f"finish_reason={final_fr!r} tool_calls={json.dumps(parsed_tcs)}",
                        f"  final_content={str(c0)[:200]!r}",
                        *(
                            [
                                f"  WARNING malformed_tool_call_blocks={malformed_blocks} "
                                "(closed <tool_call> whose JSON did not parse; stripped, no call made)"
                            ]
                            if malformed_blocks
                            else []
                        ),
                    ],
                )
                write_proxy_trace(
                    {
                        "ts": ts_iso,
                        "req_id": req_id,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "client_to_proxy": {
                            "stream": client_stream,
                            "has_tools": has_tools,
                            "tool_names": raw_tool_names,
                            "last_raw_role": last_raw_role,
                            "last_user": last_user,
                            "messages": raw_msgs_snapshot,
                        },
                        "proxy_to_llm": {
                            "cybergate_mode": payload.get("cybergate_mode"),
                            "max_tokens": payload.get("max_tokens"),
                            "is_webui_task": is_webui_task,
                            "messages": msgs,
                        },
                        "llm_to_proxy": {
                            "status": status,
                            "x_cybergate": final_obj.get("x_cybergate"),
                            "raw_reasoning_content": raw_reasoning_snap,
                            "raw_content": raw_content_snap,
                            "raw_finish_reason": raw_fr_snap,
                            "usage": final_obj.get("usage"),
                        },
                        "proxy_to_client": {
                            "reasoning_content": r0,
                            "content": c0,
                            "tool_calls": parsed_tcs,
                            "malformed_tool_call_blocks": malformed_blocks,
                            "finish_reason": final_fr,
                        },
                    },
                    skip_debug_log=True,
                )
                out = json.dumps(final_obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            except urllib.error.HTTPError as e:
                data = e.read()
                err_txt = data.decode("utf-8", errors="replace")
                emit_syslog(
                    req_id,
                    "4/4 ERR PROXY ->CLIENT",
                    [f"status={e.code} err={err_txt[:200]!r}"],
                )
                write_proxy_trace(
                    {
                        "ts": ts_iso,
                        "req_id": req_id,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "client_to_proxy": {
                            "stream": client_stream,
                            "has_tools": has_tools,
                            "tool_names": raw_tool_names,
                            "last_raw_role": last_raw_role,
                            "last_user": last_user,
                            "messages": raw_msgs_snapshot,
                        },
                        "proxy_to_llm": {
                            "cybergate_mode": payload.get("cybergate_mode"),
                            "max_tokens": payload.get("max_tokens"),
                            "is_webui_task": is_webui_task,
                            "messages": msgs,
                        },
                        "llm_to_proxy": {"status": e.code},
                        "proxy_to_client": {"error": err_txt},
                    },
                    skip_debug_log=True,
                )
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except (BrokenPipeError, ConnectionResetError):
                emit_syslog(req_id, "4/4 DISCONNECT", ["client closed connection before response completed"])
                return
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode("utf-8")
                emit_syslog(
                    req_id, "4/4 ERR PROXY ->CLIENT", [f"status=502 err={str(e)[:200]!r}"]
                )
                write_proxy_trace(
                    {
                        "ts": ts_iso,
                        "req_id": req_id,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "client_to_proxy": {
                            "stream": client_stream,
                            "has_tools": has_tools,
                            "tool_names": raw_tool_names,
                            "last_raw_role": last_raw_role,
                            "last_user": last_user,
                            "messages": raw_msgs_snapshot,
                        },
                        "proxy_to_llm": {
                            "cybergate_mode": payload.get("cybergate_mode"),
                            "max_tokens": payload.get("max_tokens"),
                            "is_webui_task": is_webui_task,
                            "messages": msgs,
                        },
                        "llm_to_proxy": {"status": 502},
                        "proxy_to_client": {"error": str(e)},
                    },
                    skip_debug_log=True,
                )
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

        clean_path = self.path.rstrip("/") or "/"
        if clean_path not in ("/tokenize", "/detokenize", "/v1/embeddings", "/embeddings"):
            try:
                other_payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                other_payload = None
            if isinstance(other_payload, dict) and strip_cybergate_client_kwargs(
                other_payload,
                allow_top_level_mode=False,
                expected_steer_token=env_first("CYBERGATE_STEER_TOKEN"),
            ):
                raw_body = json.dumps(other_payload).encode("utf-8")

        fallback_headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if auth_header:
            fallback_headers["Authorization"] = auth_header
        req = urllib.request.Request(
            f"{upstream}{self.path}",
            data=raw_body,
            headers=fallback_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=proxy_timeout) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type", resp.headers.get("Content-Type", "application/json")
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def start_background_proxy_server(
    host: str = "127.0.0.1",
    port: int = 8001,
    upstream: str = "http://127.0.0.1:8000",
) -> ThreadedHTTPServer | None:
    """Start the CyberGate proxy (`:8001` -> `:8000`) in a daemon thread.

    Idempotent per `port` within the current process so repeated middleware initialisations do not
    bind twice. If `port <= 0` or the port is already bound by another process (`EADDRINUSE` /
    `OSError`), logs a warning and returns `None` without raising.
    """
    global UPSTREAM
    port_num = int(port)
    if port_num <= 0:
        return None
    upstream_clean = upstream.rstrip("/")
    with _BG_PROXY_LOCK:
        UPSTREAM = upstream_clean
        existing = _BG_PROXY_SERVERS.get(port_num)
        if existing is not None:
            existing.upstream = upstream_clean
            return existing
        try:
            srv = ThreadedHTTPServer((host, port_num), ProxyHandler)
            srv.upstream = upstream_clean
        except OSError as exc:
            log_cybergate_event(
                "CYBERGATE-PROXY-WARN",
                f"Could not bind background proxy on {host}:{port_num} ({exc}); "
                "port may already be bound by an external process",
            )
            return None

        thread = threading.Thread(
            target=srv.serve_forever,
            name=f"cybergate-proxy-{port_num}",
            daemon=True,
        )
        thread.start()
        _BG_PROXY_SERVERS[port_num] = srv
        _BG_PROXY_THREADS[port_num] = thread
        log_cybergate_event(
            "CYBERGATE-PROXY",
            f"Background gated proxy listening on http://{host}:{port_num}/v1 -> "
            f"{upstream_clean}/v1 (cybergate_mode=closed_loop)",
        )
        return srv


def stop_background_proxy_server(port: int | None = None) -> None:
    """Shut down and close one or all background proxy servers started in this process."""
    with _BG_PROXY_LOCK:
        ports = [int(port)] if port is not None else list(_BG_PROXY_SERVERS.keys())
        for p in ports:
            srv = _BG_PROXY_SERVERS.pop(p, None)
            thread = _BG_PROXY_THREADS.pop(p, None)
            if srv is not None:
                try:
                    srv.shutdown()
                    srv.server_close()
                except Exception:
                    pass
            if thread is not None:
                try:
                    thread.join(timeout=2.0)
                except Exception:
                    pass


def main(argv: list[str] | None = None) -> int:
    """Run the CyberGate proxy server in the foreground."""
    global UPSTREAM, LISTEN_HOST, LISTEN_PORT
    parser = argparse.ArgumentParser(description="CyberGate OpenAI/SSE reverse-proxy bridge")
    parser.add_argument("--host", default=LISTEN_HOST, help="Bind host (default: %(default)s)")
    parser.add_argument(
        "--port", type=int, default=LISTEN_PORT, help="Bind port (default: %(default)s)"
    )
    parser.add_argument(
        "--upstream", default=UPSTREAM, help="Upstream vLLM URL (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    LISTEN_HOST = args.host
    LISTEN_PORT = int(args.port)
    UPSTREAM = args.upstream.rstrip("/")

    srv = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    srv.upstream = UPSTREAM
    print(
        f"[OpenWebUI-CyberGate-Proxy] Listening on http://{LISTEN_HOST}:{LISTEN_PORT}/v1 -> "
        f"{UPSTREAM}/v1 (default_max_tokens={DEFAULT_GUI_MAX_TOKENS})",
        flush=True,
    )
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
