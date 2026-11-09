"""Unit tests for the cleaned-up `cybergate.serving.proxy` (`:8001`) and un-gated `:8000` default."""
from __future__ import annotations

import asyncio
import http.server
import json
import socket
import threading
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from cybergate.serving import (
    CyberGateThinkMiddleware,
     proxy as proxy_mod,
    start_background_proxy_server,
    stop_background_proxy_server,
)
from cybergate.state import keys as key_builders
from cybergate.state.memory import MemoryDecisionStore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _MockUpstreamHandler(http.server.BaseHTTPRequestHandler):
    """Minimal mock vLLM (:8000) server recording requests sent by the proxy (:8001)."""

    requests_received: list[tuple[str, dict[str, Any]]] = []
    headers_received: list[tuple[str, dict[str, str]]] = []
    next_response_content: str = "Here is the normal assistant reply."
    next_response_reasoning: str = "Step 1: analyze the request."

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.__class__.headers_received.append((self.path, dict(self.headers.items())))
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps(
                {"data": [{"id": "Qwen/Qwen3.8-27B", "max_model_len": 32768}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        self.__class__.headers_received.append((self.path, dict(self.headers.items())))
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if self.path.rstrip("/") == "/tokenize":
            body = json.dumps({"count": 42}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.rstrip("/") == "/v1/chat/completions":
            self.__class__.requests_received.append((self.path, payload))
            resp = {
                "id": "chatcmpl-mock123",
                "object": "chat.completion",
                "created": 1700000000,
                "model": payload.get("model", "Qwen/Qwen3.8-27B"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": self.__class__.next_response_content,
                            "reasoning_content": self.__class__.next_response_reasoning,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                "x_cybergate": {
                    "fired": payload.get("cybergate_mode") == "closed_loop",
                    "score": 0.1950,
                    "s_ref_z": 4.25,
                    "sticky_unlock": True,
                    "sticky_revoked": False,
                    "routed": (
                        "cybergate_measured_closed_loop"
                        if payload.get("cybergate_mode") == "closed_loop"
                        else "pristine_pass_through"
                    ),
                    "num_inline_injections": 1,
                },
            }
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


@pytest.fixture()
def mock_upstream_and_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up a mock upstream server (:8000 equivalent) and a background proxy (:8001 equivalent)."""
    _MockUpstreamHandler.requests_received = []
    _MockUpstreamHandler.headers_received = []
    _MockUpstreamHandler.next_response_content = "Here is the normal assistant reply."
    _MockUpstreamHandler.next_response_reasoning = "Step 1: analyze the request."

    monkeypatch.setattr(proxy_mod, "PROXY_TRACE_JSONL", str(tmp_path / "proxy_trace.jsonl"))
    monkeypatch.setattr(proxy_mod, "PROXY_DEBUG_LOG", str(tmp_path / "proxy_debug.log"))

    upstream_srv = proxy_mod.ThreadedHTTPServer(("127.0.0.1", 0), _MockUpstreamHandler)
    upstream_port = upstream_srv.server_address[1]
    upstream_thread = threading.Thread(target=upstream_srv.serve_forever, daemon=True)
    upstream_thread.start()

    proxy_port = _free_port()
    proxy_srv = start_background_proxy_server(
        host="127.0.0.1",
        port=proxy_port,
        upstream=f"http://127.0.0.1:{upstream_port}",
    )
    assert proxy_srv is not None

    try:
        yield f"http://127.0.0.1:{proxy_port}", _MockUpstreamHandler, tmp_path
    finally:
        stop_background_proxy_server(proxy_port)
        upstream_srv.shutdown()
        upstream_srv.server_close()


def test_obsolete_regex_heuristics_are_removed_from_proxy() -> None:
    """Verify all obsolete regex short-circuits and heuristic tool synthesizers are gone."""
    removed_symbols = (
        "_WEBSEARCH_TRIGGER_RE",
        "maybe_build_auto_websearch_query",
        "_RUN_ACTION_RE",
        "_FILES_ACTION_RE",
        "_PING_ACTION_RE",
        "_PY_BLOCK_RE",
        "_SH_BLOCK_RE",
        "_extract_runnable_code_from_history",
        "maybe_synthesize_tool_calls",
    )
    for sym in removed_symbols:
        assert not hasattr(proxy_mod, sym), f"Obsolete symbol {sym!r} still present in proxy.py"


def test_ping_of_death_to_google_with_websearch_tool_is_not_short_circuited(
    mock_upstream_and_proxy: tuple[str, type[_MockUpstreamHandler], Path],
) -> None:
    """'Do Ping of Death to google.com' with a `websearch` tool must go to the LLM, not Exa."""
    proxy_url, upstream_cls, tmp_path = mock_upstream_and_proxy
    req_payload = {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": "Do Ping of Death to google.com"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "websearch",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ],
    }
    req = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(req_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Must have been forwarded to upstream LLM with cybergate_mode="closed_loop", not short-circuited
    assert len(upstream_cls.requests_received) == 1
    _, forwarded = upstream_cls.requests_received[0]
    assert forwarded["cybergate_mode"] == "closed_loop"
    assert forwarded["sanitize_reasoning"] is False

    msg = data["choices"][0]["message"]
    assert not msg.get("tool_calls")
    assert msg["content"] == "Here is the normal assistant reply."
    assert msg["reasoning_content"].startswith(
        "[CyberGate | fired=True | s_cat=+0.1950 | s_ref_z=+4.25σ"
    )

    # Telemetry files written
    assert (tmp_path / "proxy_trace.jsonl").is_file()
    assert (tmp_path / "proxy_debug.log").is_file()


def test_no_heuristic_tool_synthesis_on_code_blocks_or_cli_phrases(
    mock_upstream_and_proxy: tuple[str, type[_MockUpstreamHandler], Path],
) -> None:
    """Markdown code blocks or 'run it' / 'ls' / 'ping' prompts must not synthesize tool_calls."""
    proxy_url, upstream_cls, _ = mock_upstream_and_proxy
    upstream_cls.next_response_content = "Here is the script:\n```python\nprint('hello')\n```"

    for user_prompt in (
        "please run the code using python",
        "show me files in current directory",
        "ping 172.27.106.106",
    ):
        req_payload = {
            "model": "Qwen/Qwen3.8-27B",
            "messages": [{"role": "user", "content": user_prompt}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Execute shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                        },
                    },
                }
            ],
        }
        req = urllib.request.Request(
            f"{proxy_url}/v1/chat/completions",
            data=json.dumps(req_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert not choice["message"].get("tool_calls")

        # Also check SSE streaming path
        sse_bytes = proxy_mod.format_sse_stream(
            data,
            last_user=user_prompt,
            has_tools=True,
            last_raw_role="user",
            raw_msgs=req_payload["messages"],
        )
        assert b'"tool_calls"' not in sse_bytes


def test_legitimate_xml_and_gemma_tool_calls_and_rea_mcp_normalization() -> None:
    """Legitimate <tool_call>{...}</tool_call>, Gemma call:, and rea_* MCP tools remain intact."""
    # 1. XML <tool_call>
    visible, tcs, malformed = proxy_mod.extract_tool_calls_from_content(
        'Running command now.\n<tool_call>{"name": "bash", "arguments": {"command": "id"}}</tool_call>'
    )
    assert visible == "Running command now."
    assert malformed == 0
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "bash"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"command": "id"}

    # 2. Gemma call:
    visible_g, tcs_g, _ = proxy_mod.extract_tool_calls_from_content(
        'Checking file.\ncall:read{filePath:"/etc/hosts"}'
    )
    assert visible_g == "Checking file."
    assert len(tcs_g) == 1
    assert tcs_g[0]["function"]["name"] == "read"
    assert json.loads(tcs_g[0]["function"]["arguments"]) == {"filePath": "/etc/hosts"}

    # 3. REA MCP tools in normalize_tools_and_messages
    payload: dict[str, Any] = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "mcp__rea__open_binary",
                    "description": "Open a binary in Hopper",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
    }
    norm_msgs, has_tools, last_role, is_local = proxy_mod.normalize_tools_and_messages(
        payload,
        [{"role": "user", "content": "Decompile ./challenge"}],
        is_webui_task=False,
    )
    assert has_tools is True
    assert last_role == "user"
    assert is_local is False
    assert "mcp__rea__open_binary" in norm_msgs[0]["content"]
    assert "hopper" in norm_msgs[0]["content"]


def test_start_background_proxy_server_idempotent_and_handles_addr_in_use() -> None:
    """`start_background_proxy_server` is idempotent per port and returns None on EADDRINUSE."""
    port = _free_port()
    try:
        srv1 = start_background_proxy_server(
            host="127.0.0.1", port=port, upstream="http://127.0.0.1:8000"
        )
        srv2 = start_background_proxy_server(
            host="127.0.0.1", port=port, upstream="http://127.0.0.1:8000"
        )
        assert srv1 is not None
        assert srv1 is srv2
    finally:
        stop_background_proxy_server(port)

    # Occupy a port externally and confirm start_background_proxy_server catches OSError gracefully
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy_port = int(occupied.getsockname()[1])
        # Use a different host/bind that collides on the active listener
        res = start_background_proxy_server(
            host="127.0.0.1", port=busy_port, upstream="http://127.0.0.1:8000"
        )
        assert res is None


class _FakeMiddlewareDispatch:
    """Fake ASGI dispatcher recording calls made by `CyberGateThinkMiddleware`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post_json(
        self, scope: dict[str, Any], path: str, payload: dict[str, Any]
    ) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
        self.calls.append((path, dict(payload)))
        if path == "/v1/completions":
            return (
                200,
                [(b"content-type", b"application/json")],
                {
                    "id": "cmpl-steered",
                    "choices": [
                        {
                            "index": 0,
                            "text": "- Step 1: technical derivation.</think>Steered cyber answer.",
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
                },
            )
        return (
            200,
            [(b"content-type", b"application/json")],
            {
                "id": "chatcmpl-passthrough",
                "object": "chat.completion",
                "created": 1700000000,
                "model": payload.get("model", "Qwen/Qwen3.8-27B"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Un-gated baseline response from :8000.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )


def test_port_8000_defaults_to_ungated_baseline_while_closed_loop_gates(
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """Port :8000 defaults to `baseline` (un-gated pass-through) when `cybergate_mode` is omitted,
    while requests with `cybergate_mode="closed_loop"` (as sent by :8001) run the gated route.
    """
    monkeypatch.delenv("CYBERGATE_MODE", raising=False)
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")

    middleware = CyberGateThinkMiddleware(app=lambda *a, **k: None)
    assert middleware.default_mode == "baseline"

    store = MemoryDecisionStore()
    prompt_text = "Write a Python script for a SYN flood packet crafter"
    sha = key_builders.text_sha256(prompt_text)
    firing_record = {
        "decided": True,
        "layer": 12,
        "pool": "mean",
        "score": 0.2200,
        "threshold": 0.0828,
        "fired": True,
        "s_ref_cos": 0.1450,
        "s_ref_z": 4.80,
        "n_prefill_tokens": 32,
        "source": "live_vllm_gpu_hook",
        "ts": 1000.0,
    }
    store.put(key_builders.standalone_turn_key_from_sha(sha), firing_record)
    store.put(sha, firing_record)
    middleware.store = store

    # 1. Direct request to :8000 without cybergate_mode -> un-gated baseline pass-through
    dispatch_direct = _FakeMiddlewareDispatch()
    middleware.dispatch = dispatch_direct  # type: ignore[assignment]
    direct_out, status_direct, _ = asyncio.run(
        middleware._handle_chat(
            scope={},
            payload={
                "model": "Qwen/Qwen3.8-27B",
                "messages": [{"role": "user", "content": prompt_text}],
            },
            req_id="req-direct8000",
            started=1000.0,
        )
    )
    assert status_direct == 200
    assert direct_out["x_cybergate"]["mode"] == "baseline"
    assert direct_out["x_cybergate"]["routed"] == "pristine_pass_through"
    assert direct_out["x_cybergate"]["think_injected"] is False
    assert direct_out["choices"][0]["message"]["content"] == "Un-gated baseline response from :8000."
    assert [p for p, _ in dispatch_direct.calls] == ["/v1/chat/completions"]

    # 2. Request with cybergate_mode="closed_loop" (sent by :8001 proxy) -> gated closed_loop route
    dispatch_gated = _FakeMiddlewareDispatch()
    middleware.dispatch = dispatch_gated  # type: ignore[assignment]
    gated_out, status_gated, _ = asyncio.run(
        middleware._handle_chat(
            scope={},
            payload={
                "model": "Qwen/Qwen3.8-27B",
                "messages": [{"role": "user", "content": prompt_text}],
                "cybergate_mode": "closed_loop",
                "sanitize_reasoning": False,
            },
            req_id="req-proxy8001",
            started=1000.0,
        )
    )
    assert status_gated == 200
    assert gated_out["x_cybergate"]["mode"] == "closed_loop"
    assert gated_out["x_cybergate"]["routed"] == "cybergate_measured_closed_loop"
    assert gated_out["x_cybergate"]["think_injected"] is True
    assert gated_out["choices"][0]["message"]["content"] == "Steered cyber answer."
    assert [p for p, _ in dispatch_gated.calls] == ["/v1/completions"]


def test_middleware_auto_starts_background_proxy_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CyberGateThinkMiddleware.__init__` binds `CYBERGATE_PROXY_PORT` pointing to `CYBERGATE_PORT`."""
    proxy_port = _free_port()
    monkeypatch.setenv("CYBERGATE_PORT", "18000")
    monkeypatch.setenv("CYBERGATE_PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", str(proxy_port))
    monkeypatch.delenv("CYBERGATE_PROXY_UPSTREAM", raising=False)
    try:
        CyberGateThinkMiddleware(app=lambda *a, **k: None)
        srv = proxy_mod._BG_PROXY_SERVERS.get(proxy_port)
        assert srv is not None
        assert srv.upstream == "http://127.0.0.1:18000"
    finally:
        stop_background_proxy_server(proxy_port)


def test_docker_compose_defines_gemma4_and_qwen38_dual_port_services() -> None:
    """Verify the two per-model Docker Compose files exist and configure `:8000` baseline + `:8001` proxy."""
    repo_root = Path(__file__).resolve().parents[2]
    gemma_compose = repo_root / "docker-compose.gemma-4-26B-A4B-it.yml"
    qwen_compose = repo_root / "docker-compose.Qwen3.8-27B.yml"
    assert gemma_compose.is_file(), "docker-compose.gemma-4-26B-A4B-it.yml must exist at repository root"
    assert qwen_compose.is_file(), "docker-compose.Qwen3.8-27B.yml must exist at repository root"
    assert not (repo_root / "docker-compose.yml").exists(), "Combined docker-compose.yml should be replaced by per-model files"

    gemma_text = gemma_compose.read_text(encoding="utf-8")
    assert "  gemma-4-26B-A4B-it:" in gemma_text
    assert "CYBERGATE_PORT=8000" in gemma_text
    assert "CYBERGATE_MODE=baseline" in gemma_text
    assert "CYBERGATE_PROXY_PORT=8001" in gemma_text
    assert "google/gemma-4-26B-A4B-it" in gemma_text
    assert "[CyberGate ERROR] Model weights not found in local Hugging Face cache:" in gemma_text
    assert "huggingface-cli download google/gemma-4-26B-A4B-it" in gemma_text
    assert "cybergate_plugin.middleware.CyberGateThinkMiddleware" in gemma_text

    qwen_text = qwen_compose.read_text(encoding="utf-8")
    assert "  Qwen3.8-27B:" in qwen_text
    assert "CYBERGATE_PORT=8000" in qwen_text
    assert "CYBERGATE_MODE=baseline" in qwen_text
    assert "CYBERGATE_PROXY_PORT=8001" in qwen_text
    assert "Qwen/Qwen3.8-27B" in qwen_text
    assert "[CyberGate ERROR] Model weights not found in local Hugging Face cache:" in qwen_text
    assert "huggingface-cli download Qwen/Qwen3.8-27B" in qwen_text
    assert "cybergate_plugin.middleware.CyberGateThinkMiddleware" in qwen_text
    assert "CYBERGATE_CANDIDATE_BUNDLE_PATH" not in gemma_text


def test_srv1_default_host_is_loopback_and_authorization_header_forwarded(
    mock_upstream_and_proxy: tuple[str, type[_MockUpstreamHandler], Path],
) -> None:
    """SRV-1 & REL-4: Proxy defaults to 127.0.0.1, trace logs default to disabled, and Authorization is forwarded."""
    assert proxy_mod.LISTEN_HOST == "127.0.0.1"
    assert isinstance(proxy_mod.PROXY_TRACE_JSONL, str)

    proxy_url, upstream_cls, _ = mock_upstream_and_proxy

    # 1. GET /v1/models forwards Authorization
    get_req = urllib.request.Request(
        f"{proxy_url}/v1/models",
        headers={"Authorization": "Bearer secret-token-123"},
        method="GET",
    )
    with urllib.request.urlopen(get_req, timeout=10) as resp:
        assert resp.status == 200

    # 2. POST /v1/chat/completions forwards Authorization to /v1/models, /tokenize, and /v1/chat/completions
    post_req = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "Qwen/Qwen3.8-27B",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer secret-token-123",
        },
        method="POST",
    )
    with urllib.request.urlopen(post_req, timeout=10) as resp:
        assert resp.status == 200

    paths_with_auth = {
        path: hdrs.get("Authorization") for path, hdrs in upstream_cls.headers_received
    }
    assert paths_with_auth.get("/v1/models") == "Bearer secret-token-123"
    assert paths_with_auth.get("/tokenize") == "Bearer secret-token-123"
    assert paths_with_auth.get("/v1/chat/completions") == "Bearer secret-token-123"


def test_srv2_srv4_new3_new4_new5_regex_performance_and_gemma_long_calls() -> None:
    """SRV-2, SRV-4, NEW-3, NEW-4, NEW-5: Fast linear completion on 64 KB adversarial inputs and >2000-char Gemma calls."""
    import time
    from cybergate import boilerplate as boiler_mod
    from cybergate.families import family_by_name

    assert not hasattr(proxy_mod, "_CONVERSATIONAL_ACK_RE")
    assert not hasattr(proxy_mod, "is_conversational_ack_or_status")

    # 1. _GEMMA_CALL_RE on adversarial 56 KB input (< 15 ms) AND >2,000-char valid call (NEW-5)
    adv_gemma = "call:a{" * 14000
    t0 = time.perf_counter()
    res_gemma = list(proxy_mod._GEMMA_CALL_RE.finditer(adv_gemma))
    elapsed_gemma_ms = (time.perf_counter() - t0) * 1000.0
    assert res_gemma == []
    assert elapsed_gemma_ms < 15.0, f"_GEMMA_CALL_RE took {elapsed_gemma_ms:.2f} ms"

    long_gemma_call = 'call:bash{command:"python3 -c \'print(1)\' # ' + ("a" * 2500) + '"}'
    assert len(long_gemma_call) > 2500
    leftover, tcs_long, bad_long = proxy_mod.extract_tool_calls_from_content(long_gemma_call)
    assert len(tcs_long) == 1
    assert leftover == ""
    assert bad_long == 0

    # 2. _BOILERPLATE_INJECT_RE on newlines and repeated unclosed '[Policy Check' (< 10 ms)
    for adv_boiler in ("\n" * 20000 + "x", "[Policy Check" * 5000):
        t1 = time.perf_counter()
        res_boiler = proxy_mod._BOILERPLATE_INJECT_RE.sub("", adv_boiler)
        elapsed_boiler_ms = (time.perf_counter() - t1) * 1000.0
        assert res_boiler == adv_boiler
        assert elapsed_boiler_ms < 10.0, f"_BOILERPLATE_INJECT_RE took {elapsed_boiler_ms:.2f} ms"

    # 3. NEW-3: _sanitize_tool_call_json on 32K literal \n escapes (< 5 ms)
    adv_json_tail = '{"name": "bash", "arguments": {"command": "echo hi' + ("\\n" * 16000) + '"}}'
    t2 = time.perf_counter()
    sanitized = proxy_mod._sanitize_tool_call_json(adv_json_tail)
    elapsed_san_ms = (time.perf_counter() - t2) * 1000.0
    assert sanitized == adv_json_tail
    assert elapsed_san_ms < 5.0, f"_sanitize_tool_call_json took {elapsed_san_ms:.2f} ms"

    # 4. NEW-4: boilerplate.strip_injected_think_boilerplate on 20K newlines (< 10 ms)
    qwen_fam = family_by_name("qwen38")
    t3 = time.perf_counter()
    boiler_mod.strip_injected_think_boilerplate("\n" * 20000 + "x", family=qwen_fam)
    elapsed_strip_ms = (time.perf_counter() - t3) * 1000.0
    assert elapsed_strip_ms < 10.0, f"strip_injected_think_boilerplate took {elapsed_strip_ms:.2f} ms"


def test_srv3_multistep_tool_chains_and_d1_through_d5_cases(
    mock_upstream_and_proxy: tuple[str, type[_MockUpstreamHandler], Path],
) -> None:
    """SRV-3: Multi-step tool chains and cases D1-D5 never drop legitimate repeat/rewrite tool calls."""
    proxy_url, upstream_cls, _ = mock_upstream_and_proxy
    tools = [
        {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}},
    ]

    def _tc(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    def _asst(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"role": "assistant", "content": "", "tool_calls": [_tc(cid, name, args)]}

    def _tool(cid: str, out: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": cid, "content": out}

    def _xml_call(name: str, args: dict[str, Any]) -> str:
        return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"

    def _post_turn(messages: list[dict[str, Any]], model_output: str) -> tuple[dict[str, Any], dict[str, Any]]:
        upstream_cls.next_response_content = model_output
        req = urllib.request.Request(
            f"{proxy_url}/v1/chat/completions",
            data=json.dumps({"model": "Qwen/Qwen3.8-27B", "messages": messages, "tools": tools}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        _, fwd = upstream_cls.requests_received[-1]
        return out["choices"][0], fwd

    user_msg = {"role": "user", "content": "run fetch.py against http://10.0.0.5 and fix it until it works"}

    # D1: run fails (missing module) -> bash pip install -> model re-runs python3 fetch.py
    ch_d1, _ = _post_turn(
        [
            user_msg,
            _asst("c1", "bash", {"command": "python3 fetch.py"}),
            _tool("c1", "ModuleNotFoundError: No module named 'requests'"),
            _asst("c2", "bash", {"command": "pip install requests"}),
            _tool("c2", "Successfully installed requests-2.32.3"),
        ],
        _xml_call("bash", {"command": "python3 fetch.py"}),
    )
    assert ch_d1["finish_reason"] == "tool_calls"
    assert ch_d1["message"]["tool_calls"][0]["function"]["name"] == "bash"

    # D2: file fixed with bash sed -i -> model re-runs python3 fetch.py
    ch_d2, _ = _post_turn(
        [
            user_msg,
            _asst("c1", "bash", {"command": "python3 fetch.py"}),
            _tool("c1", "Traceback (most recent call last):\nNameError: name 'urlopen' is not defined"),
            _asst("c2", "bash", {"command": "sed -i 's/import json/import json, urllib.request/' fetch.py"}),
            _tool("c2", ""),
        ],
        _xml_call("bash", {"command": "python3 fetch.py"}),
    )
    assert ch_d2["finish_reason"] == "tool_calls"
    assert ch_d2["message"]["tool_calls"][0]["function"]["name"] == "bash"

    # D3: poll the same status URL twice (no error)
    ch_d3, _ = _post_turn(
        [
            {"role": "user", "content": "start the scan job on 10.0.0.5 and wait until it reports done"},
            _asst("c1", "bash", {"command": "curl -s http://10.0.0.5:9000/status"}),
            _tool("c1", '{"state": "running", "progress": 40}'),
        ],
        _xml_call("bash", {"command": "curl -s http://10.0.0.5:9000/status"}),
    )
    assert ch_d3["finish_reason"] == "tool_calls"
    assert ch_d3["message"]["tool_calls"][0]["function"]["name"] == "bash"

    # D4: wrote x.py, run succeeded but output wrong -> model rewrites x.py
    ch_d4, _ = _post_turn(
        [
            {"role": "user", "content": "write x.py that prints the open ports of 10.0.0.5 and run it"},
            _asst("c1", "write", {"filePath": "x.py"}),
            _tool("c1", "Wrote file successfully."),
            _asst("c2", "bash", {"command": "python3 x.py"}),
            _tool("c2", "open ports: []"),
        ],
        _xml_call("write", {"filePath": "x.py", "content": "print([80, 443])"}),
    )
    assert ch_d4["finish_reason"] == "tool_calls"
    assert ch_d4["message"]["tool_calls"][0]["function"]["name"] == "write"

    # D5: model wrote a non-Python file (notes.md) -> proxy must not tell it to run `python3 notes.md` or ban `write`
    ch_d5, fwd_d5 = _post_turn(
        [
            {"role": "user", "content": "write notes.md summarising the nmap results, then write report.html"},
            _asst("c1", "write", {"filePath": "notes.md"}),
            _tool("c1", "Wrote file successfully."),
        ],
        _xml_call("write", {"filePath": "report.html", "content": "<h1>Report</h1>"}),
    )
    assert ch_d5["finish_reason"] == "tool_calls"
    assert ch_d5["message"]["tool_calls"][0]["function"]["name"] == "write"
    last_fwd_content = fwd_d5["messages"][-1]["content"]
    assert "python3 notes.md" not in last_fwd_content
    assert "Do NOT call `write` again" not in last_fwd_content
    assert "[Tool Execution Output]:" in last_fwd_content


def test_srv5_srv6_srv7_srv10_proxy_request_and_response_contracts(
    mock_upstream_and_proxy: tuple[str, type[_MockUpstreamHandler], Path],
) -> None:
    """Verify SRV-5 (4096 max_tokens cap), SRV-6 (system prompt & no repetition_penalty), SRV-7 (has_tools=False), SRV-10 (cybergate_tool_turn)."""
    proxy_url, upstream_cls, _ = mock_upstream_and_proxy

    # Case A: has_tools=False, >400-char custom system prompt, no max_tokens, response contains literal <tool_call>
    long_sys_prompt = ("You are a specialized binary analysis assistant. " * 12).strip()
    assert len(long_sys_prompt) > 400
    upstream_cls.next_response_content = (
        'In XML format, a call looks like <tool_call>{"name": "bash", "arguments": {"command": "id"}}</tool_call>.'
    )

    req_no_tools = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "Qwen/Qwen3.8-27B",
                "messages": [
                    {"role": "system", "content": long_sys_prompt},
                    {"role": "user", "content": "Show an example XML tool call tag."},
                ],
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_no_tools, timeout=10) as resp:
        data_no_tools = json.loads(resp.read().decode("utf-8"))

    _, forwarded_no_tools = upstream_cls.requests_received[-1]
    # SRV-5: default max_tokens is capped at 4096, not 32k
    assert forwarded_no_tools["max_tokens"] == 4096
    # SRV-6: system prompt >400 chars is preserved intact and no unrequested repetition_penalty is added
    assert forwarded_no_tools["messages"][0]["content"] == long_sys_prompt
    assert "repetition_penalty" not in forwarded_no_tools
    assert "cybergate_tool_turn" not in forwarded_no_tools
    # SRV-7: has_tools=False leaves <tool_call> text intact in content and does not extract tool_calls
    msg_no_tools = data_no_tools["choices"][0]["message"]
    assert not msg_no_tools.get("tool_calls")
    assert "<tool_call>" in msg_no_tools["content"]

    # Also check format_sse_stream with has_tools=False
    trace_sse: dict[str, Any] = {}
    proxy_mod.format_sse_stream(data_no_tools, has_tools=False, trace_out=trace_sse)
    assert trace_sse["tool_calls"] == []
    assert "<tool_call>" in trace_sse["content"]

    # Case B: has_tools=True preserves custom system prompt, sets cybergate_tool_turn=True, and honors caller max_tokens
    custom_sys = "Always explain findings with CVE identifiers."
    req_with_tools = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "Qwen/Qwen3.8-27B",
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": custom_sys},
                    {"role": "user", "content": "Check binary"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "bash", "parameters": {"type": "object"}},
                    }
                ],
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_with_tools, timeout=10) as resp:
        resp.read()

    _, forwarded_with_tools = upstream_cls.requests_received[-1]
    assert forwarded_with_tools["max_tokens"] == 1024
    assert forwarded_with_tools["cybergate_tool_turn"] is True
    assert forwarded_with_tools["messages"][0]["content"].startswith(custom_sys)
    assert "<tool_call>" in forwarded_with_tools["messages"][0]["content"]


def test_srv9_and_new1_missing_kit_abstains_in_both_baseline_and_closed_loop(
    monkeypatch: pytest.MonkeyPatch,
    clean_process_state: None,
) -> None:
    """SRV-9 & NEW-1: Families without a kit on disk fall back to math.inf and NEVER fire in closed_loop or baseline."""
    import math
    from cybergate.decision.evidence import legacy_op_for_model_id, resolve_threshold_and_family

    monkeypatch.delenv("CYBERGATE_OPS_DIR", raising=False)
    monkeypatch.delenv("CYBERGATE_OP_PATH", raising=False)
    monkeypatch.setenv("CYBERGATE_PROXY_PORT", "0")

    op_q36, tag_q36 = legacy_op_for_model_id("Qwen/Qwen3.6-35B-A3B")
    assert tag_q36 == "qwen36"
    assert op_q36.model_key == "qwen36"
    assert math.isinf(op_q36.thresh) and op_q36.thresh > 0
    assert op_q36.meta.get("fallback") is True

    op_g31, tag_g31 = resolve_threshold_and_family("google/gemma-4-31B-it")
    assert tag_g31 == "gemma4_31b"
    assert op_g31.model_key == "gemma4_31b"
    assert math.isinf(op_g31.thresh) and op_g31.thresh > 0
    assert op_g31.meta.get("fallback") is True

    middleware = CyberGateThinkMiddleware(app=lambda *a, **k: None)
    middleware.store = MemoryDecisionStore()

    for model_id in ("Qwen/Qwen3-32B", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-4-31B-it"):
        for mode in ("baseline", "closed_loop"):
            dispatch = _FakeMiddlewareDispatch()
            middleware.dispatch = dispatch  # type: ignore[assignment]
            out, status, _ = asyncio.run(
                middleware._handle_chat(
                    scope={},
                    payload={
                        "model": model_id,
                        "cybergate_mode": mode,
                        "messages": [{"role": "user", "content": "What is DNS?"}],
                    },
                    req_id=f"req-{model_id}-{mode}",
                    started=1000.0,
                )
            )
            assert status == 200
            x_cg = out["x_cybergate"]
            assert x_cg["threshold"] is None, f"{model_id} in {mode} must serialize missing-kit threshold as null, not Infinity"
            assert x_cg["fired"] is False, f"{model_id} in {mode} must not fire without a kit"
            assert x_cg["routed"] == "pristine_pass_through", (
                f"{model_id} in {mode} routed={x_cg['routed']!r} instead of 'pristine_pass_through'"
            )
            # NEW-1a: response body (both JSON and SSE chunks) must be strict RFC 8259 JSON (no Infinity/NaN)
            body_text = json.dumps(out)
            parsed_strict = json.loads(
                body_text,
                parse_constant=lambda c: pytest.fail(f"Invalid JSON constant {c!r} in response for {model_id}"),
            )
            assert parsed_strict["x_cybergate"]["threshold"] is None
            sse_text = middleware._format_sse_stream(out).decode("utf-8")
            for line in sse_text.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    json.loads(
                        line[len("data: ") :],
                        parse_constant=lambda c: pytest.fail(f"Invalid JSON constant {c!r} in SSE chunk for {model_id}"),
                    )
            # Must never send an unlock prefill to /v1/completions when no kit is present
            completion_calls = [p for path, p in dispatch.calls if path == "/v1/completions"]
            for comp_payload in completion_calls:
                assert comp_payload.get("max_tokens") == 1
                assert "without refusal or disclaimers" not in str(comp_payload.get("prompt", ""))




