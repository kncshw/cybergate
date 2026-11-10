#!/usr/bin/env python3
"""CLI entry point and re-export shim for `cybergate.serving.proxy` (`:8001` -> `:8000`)."""
from __future__ import annotations

import pathlib
import sys

_REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from cybergate.serving.proxy import (  # noqa: E402, F401
    DEFAULT_GUI_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
    LISTEN_HOST,
    LISTEN_PORT,
    MAX_MODEL_LEN,
    PROXY_DEBUG_LOG,
    PROXY_TIMEOUT,
    PROXY_TRACE_JSONL,
    UPSTREAM,
    ProxyHandler,
    ThreadedHTTPServer,
    _BOILERPLATE_INJECT_RE,
    _DISCLAIMER_RE,
    _GEMMA_CALL_RE,
    _OPENER_PREFIX_RE,
    _PRIOR_THINK_RE,
    _TOOL_CALL_RE,
    _filter_duplicate_tool_calls,
    _flatten_content,
    _parse_single_tool_json,
    _sanitize_tool_call_json,
    clean_prior_assistant_messages,
    emit_syslog,
    estimate_chat_prompt_tokens,
    extract_tool_calls_from_content,
    fmt_gate_score,
    format_cybergate_header,
    format_sse_stream,
    get_full_remaining_max_tokens,
    get_upstream_max_model_len,
    main,
    normalize_tools_and_messages,
    prepend_cybergate_header,
    query_upstream_prompt_tokens,
    repair_thought_and_visible,
    start_background_proxy_server,
    stop_background_proxy_server,
    strip_cybergate_client_kwargs,
    write_proxy_trace,
)

if __name__ == "__main__":
    raise SystemExit(main())
