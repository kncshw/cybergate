"""`CyberGateThinkMiddleware` -- the in-process ASGI middleware.

Runs in the **API-server** process (`MainProcess`), installed with
`vllm serve --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware`. Intercepts
`POST /v1/chat/completions` and does three things:

1. **Probe.** Dispatches 1-token completions back through the ASGI app so the model worker's forward
   hooks observe each user turn, both standalone and in context, and publish decision records. This
   is the only way the API-server process can get a score: the hooks live in the other process.
2. **Decide.** Calls the gate with those records.
3. **Route.** Either forwards the request untouched (`fired=False`), or runs a think-injection route.

Note the asymmetry that makes the probe necessary and awkward: with `--enable-prefix-caching`, a
repeated prefix produces no prefill, so the hook does not run and no fresh record appears. The probe
re-sends the prompt to force an observation, and skips the probe when a record with a refusal reading
already exists.
"""
from __future__ import annotations

import json
import math
import sys
import time
import uuid
from typing import Any, Callable, Sequence

from ..boilerplate import strip_injected_think_boilerplate
from ..config import DEFAULT_MODEL_ID, SUPPRESS_OPENER_XARG, env_first
from ..decision import evaluate_gate_for_messages
from ..families import family_for_model_id
from ..messages import extract_user_turns
from ..state import clear_request_gate_state, keys as key_builders
from ..state.ipc_shm import ShmDecisionStore
from ..telemetry.log import log_cybergate_event
from .asgi import AsgiDispatcher, replay_body
from .proxy import start_background_proxy_server
from .think_routes import RouteConfig, RouteRequest, measured_closed_loop_route, route_for_mode

__all__ = ["CyberGateThinkMiddleware"]

CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"


def _detect_vllm_port(argv: Sequence[str] | None = None) -> int:
    """Resolve the local vLLM HTTP port from `CYBERGATE_PORT`, `--port` CLI args, or default 8000."""
    env_port = env_first("CYBERGATE_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    args = list(sys.argv if argv is None else argv)
    for idx, arg in enumerate(args):
        if arg == "--port" and idx + 1 < len(args):
            try:
                return int(args[idx + 1])
            except ValueError:
                pass
        elif arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return 8000


class CyberGateThinkMiddleware:
    """ASGI middleware applying the gate inside `vllm serve`."""

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app
        self.dispatch = AsgiDispatcher(app)
        self.default_mode = env_first("CYBERGATE_MODE", default="baseline")
        self.sanitize_reasoning = (
            env_first("CYBERGATE_SANITIZE_REASONING", default="1")
            != "0"
        )
        self.store = ShmDecisionStore()
        proxy_port_raw = env_first("CYBERGATE_PROXY_PORT", default="8001") or "8001"
        try:
            proxy_port = int(proxy_port_raw)
        except ValueError:
            proxy_port = 8001
        proxy_host = env_first("CYBERGATE_PROXY_HOST", default="127.0.0.1") or "127.0.0.1"
        vllm_port = _detect_vllm_port()
        upstream = (
            env_first("CYBERGATE_PROXY_UPSTREAM", default=f"http://127.0.0.1:{vllm_port}")
            or f"http://127.0.0.1:{vllm_port}"
        )
        if proxy_port > 0:
            start_background_proxy_server(host=proxy_host, port=proxy_port, upstream=upstream)
        log_cybergate_event(
            "CYBERGATE-MIDDLEWARE",
            f"Initialized CyberGateThinkMiddleware (default_mode={self.default_mode}, "
            f"sanitize_reasoning={self.sanitize_reasoning}, proxy_port={proxy_port})",
        )

    # ---------------------------------------------------------------------------------------------
    # ASGI plumbing
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _wrap_chat_json(
        comp_json: dict[str, Any],
        model_id: str,
        content: str,
        reasoning_content: str,
        x_cybergate: dict[str, Any],
    ) -> dict[str, Any]:
        """Format a completion response as a standard OpenAI `chat.completion` object."""
        choice0 = comp_json.get("choices", [{}])[0]
        return {
            "id": comp_json.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion",
            "created": comp_json.get("created", int(time.time())),
            "model": comp_json.get("model", model_id),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning_content,
                    },
                    "finish_reason": choice0.get("finish_reason", "stop"),
                }
            ],
            "usage": comp_json.get("usage", {}),
            "x_cybergate": x_cybergate,
            "x_cybergate": x_cybergate,
        }

    # ---------------------------------------------------------------------------------------------
    # Phase 1: force fresh observations
    # ---------------------------------------------------------------------------------------------
    async def _probe_observations(
        self,
        scope: dict[str, Any],
        messages: Sequence[dict[str, Any]],
        user_turns: Sequence[str],
        model_id: str,
        include_bos: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float]:
        """Ensure a record exists for each turn standalone and for the conversation in context.

        Returns `(in_context_record, standalone_record, probe_ms)`. A probe is skipped when a record
        already carries `s_ref_z`, because that means the refusal monitor saw the prefill; a record
        without it came from a prefix-cached pass that the hook never observed.
        """
        started = time.perf_counter()
        did_probe = False
        latest_sha = key_builders.text_sha256(user_turns[-1])

        async def probe(probe_messages: Sequence[dict[str, Any]]) -> None:
            # The client's own system prompt is dropped before rendering, because it is a different
            # render rather than more context. Both directions were fitted with
            # `DEFAULT_SYSTEM_PROMPT` in front, and the refusal reading is a single cosine at the
            # last prefill token (`pool="last"`), which sits downstream of every system token.
            # Measured on Qwen3.8 with OpenCode's 668-token tool preamble ahead of one user turn:
            # s_ref_cos 0.1371 -> 0.0106, i.e. +3.39 sigma -> +0.24 sigma, through the 1.50 floor,
            # while `s_cat` -- pooled over the user span -- barely moved (+0.1879 -> +0.1993). That
            # is the Qwen3.8 docstring's "the gate's render is not the render the model answers in"
            # reaching the probe itself.
            #
            # `extract_system_and_use` reads a system prompt only at index 0, so dropping that one
            # message is exactly what makes it fall back to `DEFAULT_SYSTEM_PROMPT`. Prior turns are
            # kept: those are the context the in-context probe exists to measure.
            if probe_messages and probe_messages[0].get("role") in ("system", "developer"):
                probe_messages = probe_messages[1:]
            prompt = family_for_model_id(model_id).build_raw_prompt(
                probe_messages, mode="baseline", include_bos=include_bos
            )
            await self.dispatch.post_json(
                scope,
                COMPLETIONS_PATH,
                {
                    "model": model_id,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "add_special_tokens": False,
                },
            )

        def standalone_record(sha: str) -> dict[str, Any] | None:
            return self.store.get(
                key_builders.standalone_turn_key_from_sha(sha)
            ) or self.store.get(sha)

        if len(user_turns) > 1:
            # Every turn needs a standalone observation: the prior-unlock scan reads all of them.
            for turn_text in user_turns:
                sha = key_builders.text_sha256(turn_text)
                record = standalone_record(sha)
                if record is None or record.get("s_ref_z") is None:
                    did_probe = True
                    await probe([{"role": "user", "content": turn_text}])
            standalone = standalone_record(latest_sha)

            conv_key = key_builders.conversation_key(user_turns)

            def in_context_record() -> dict[str, Any] | None:
                # Conversation-keyed only. The canonical code preferred `ctx_<sha(final turn)>`,
                # which collides across conversations -- that was R11.
                return self.store.get(conv_key)

            in_context = in_context_record()
            if in_context is None or in_context.get("s_ref_z") is None:
                did_probe = True
                probe_t0 = time.time() - 0.05
                await probe(messages)
                in_context = in_context_record()
                if in_context is None or in_context.get("s_ref_z") is None:
                    pc = (
                        self.store.get(f"pc_ctx_{latest_sha}")
                        or self.store.get("last_multiturn_ctx")
                    )
                    if pc is not None and float(pc.get("ts", 0.0)) >= probe_t0:
                        self.store.put(conv_key, pc)
                        in_context = pc
        else:

            def single_turn_record() -> dict[str, Any] | None:
                return (
                    self.store.get(key_builders.standalone_turn_key_from_sha(latest_sha))
                    or self.store.get(key_builders.conversation_key(user_turns))
                    or self.store.get(latest_sha)
                    or self.store.get(f"pc_ctx_{latest_sha}")
                )

            in_context = single_turn_record()
            if in_context is None or in_context.get("s_ref_z") is None:
                did_probe = True
                await probe(messages)
                in_context = single_turn_record()
                if in_context is None or in_context.get("s_ref_z") is None:
                    await probe([{"role": "user", "content": user_turns[-1]}])
                    in_context = single_turn_record()
            standalone = in_context

        probe_ms = round((time.perf_counter() - started) * 1000.0, 2) if did_probe else 0.0
        return in_context, standalone, probe_ms

    # ---------------------------------------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------------------------------------
    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope.get("type") != "http" or scope.get("path") != CHAT_PATH:
            await self.app(scope, receive, send)
            return

        body_chunks: list[bytes] = []
        while True:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        raw_body = b"".join(body_chunks)
        req_id = f"req-{uuid.uuid4().hex[:10]}"
        started = time.time()

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            # Not JSON: let the wrapped app produce its own error response.
            await self.app(scope, replay_body(raw_body), send)
            return

        try:
            client_stream = bool(payload.pop("stream", False))
            payload.pop("stream_options", None)
            final_chat_obj, status, headers = await self._handle_chat(
                scope, payload, req_id, started
            )
            if client_stream and status == 200 and "choices" in final_chat_obj:
                out_bytes = self._format_sse_stream(final_chat_obj)
                out_headers = [
                    (key, value)
                    for (key, value) in headers
                    if key.lower() not in (b"content-length", b"content-type")
                ] + [
                    (b"content-type", b"text/event-stream; charset=utf-8"),
                    (b"cache-control", b"no-cache"),
                    (b"content-length", str(len(out_bytes)).encode("ascii")),
                ]
            else:
                out_bytes = json.dumps(final_chat_obj).encode("utf-8")
                out_headers = [
                    (key, value) for (key, value) in headers if key.lower() != b"content-length"
                ] + [(b"content-length", str(len(out_bytes)).encode("ascii"))]
            await send({"type": "http.response.start", "status": status, "headers": out_headers})
            await send({"type": "http.response.body", "body": out_bytes, "more_body": False})
        finally:
            clear_request_gate_state(req_id)

    @staticmethod
    def _format_sse_stream(final_chat_obj: dict[str, Any]) -> bytes:
        """Convert a completed `chat.completion` object into OpenAI-compatible SSE chunks."""
        chat_id = final_chat_obj.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
        created = final_chat_obj.get("created", int(time.time()))
        model = final_chat_obj.get("model", DEFAULT_MODEL_ID)
        choice0 = (final_chat_obj.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish_reason = choice0.get("finish_reason") or "stop"
        x_cg = final_chat_obj.get("x_cybergate")

        chunks: list[str] = []
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

        c_done: dict[str, Any] = {
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
        if x_cg is not None:
            c_done["x_cybergate"] = x_cg
        chunks.append(f"data: {json.dumps(c_done)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return "".join(chunks).encode("utf-8")

    async def _handle_chat(
        self,
        scope: dict[str, Any],
        payload: dict[str, Any],
        req_id: str,
        started: float,
    ) -> tuple[dict[str, Any], int, list[tuple[bytes, bytes]]]:
        """Probe, decide and route one chat request."""
        messages = payload.get("messages", [])
        model_id = payload.get("model", DEFAULT_MODEL_ID)
        mode = str(payload.pop("cybergate_mode", self.default_mode)).lower()
        do_sticky = bool(payload.pop("enable_multi_turn_stickiness", True))
        do_sanitize = bool(payload.pop("sanitize_reasoning", self.sanitize_reasoning))
        family = family_for_model_id(model_id)
        caller_max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
        default_max_tokens = 2048 if family.name == "qwen38" else 350
        max_tokens = int(caller_max_tokens) if caller_max_tokens else default_max_tokens
        temperature = float(payload.get("temperature", 0.0))
        default_cot_tokens = 2048 if family.name == "qwen38" else 180
        max_cot_tokens = int(payload.pop("max_cot_tokens", default_cot_tokens))
        # A tool-bearing turn declines the force-emitted post-think opener. The controller forces
        # `POST_THINK_OPENER` as the first visible tokens, which commits the answer to prose before
        # the model emits a free token -- so a turn whose correct output is a tool call cannot
        # produce one. `cybergate_tool_turn` is how the Open WebUI bridge says so, having already
        # consumed `tools` itself; a caller that passes `tools` through is recognised directly.
        tool_turn = bool(payload.pop("cybergate_tool_turn", False)) or bool(payload.get("tools"))

        include_bos = family.supports_bos
        user_turns = extract_user_turns(messages)
        is_webui_task = bool(
            user_turns and user_turns[-1].lstrip().startswith("### Task:")
        )
        if is_webui_task:
            mode = "baseline"

        in_context_record: dict[str, Any] | None = None
        standalone_record: dict[str, Any] | None = None
        probe_ms = 0.0
        if user_turns and not is_webui_task:
            in_context_record, standalone_record, probe_ms = await self._probe_observations(
                scope, messages, user_turns, model_id, include_bos
            )

        if in_context_record and in_context_record.get("n_prefill_tokens"):
            max_model_len = int(env_first("CYBERGATE_MAX_MODEL_LEN", default="32768") or 32768)
            n_prefill = int(in_context_record["n_prefill_tokens"])
            # Reserve 128 tokens for OPEN_THINK_PREFIX (~60 tokens) + special-token margin
            max_avail = max(64, max_model_len - n_prefill - max_cot_tokens - 128)
            if not caller_max_tokens:
                max_tokens = max(256, min(default_max_tokens, max_avail))
            else:
                max_tokens = min(max_tokens, max_avail)
                if "max_tokens" in payload:
                    payload["max_tokens"] = max_tokens
                if "max_completion_tokens" in payload:
                    payload["max_completion_tokens"] = max_tokens

        gate_info = evaluate_gate_for_messages(
            messages=messages,
            model_id=model_id,
            request_id=req_id,
            enable_multi_turn_stickiness=do_sticky,
            live_gpu_decision=in_context_record,
            standalone_user_decision=standalone_record,
        )
        gate_info["probe_http_ms"] = probe_ms
        gate_info["tool_turn"] = tool_turn

        # When the mid/late veto suppressed this decision, `score` is a placeholder
        # (threshold - 0.005) and the measurement is in `raw_early_score`. Log both, or the line
        # reads as a near-miss on threshold and the real score is invisible.
        veto_note = (
            f" vetoed(raw_early={gate_info.get('raw_early_score'):+.6f},"
            f"reason={gate_info.get('mid_late_veto_reason')})"
            if gate_info.get("suppressed_by_mid_late")
            else ""
        )
        thresh_val = gate_info.get("threshold")
        thresh_fmt = (
            f"{thresh_val:+.6f}"
            if isinstance(thresh_val, (int, float)) and math.isfinite(thresh_val)
            else "null"
        )
        log_cybergate_event(
            "CYBERGATE-MIDDLEWARE",
            f"req_id={req_id} mode={mode} turn={gate_info.get('user_turn_index', 1)} "
            f"s_cat={gate_info.get('score', 0.0):+.6f}{veto_note} "
            f"(in_ctx={gate_info.get('s_cat_in_context', 0.0):+.6f}, "
            f"st={gate_info.get('s_cat_standalone', 0.0):+.6f}, "
            f"conv={gate_info.get('s_cat_conv', 0.0):+.6f}, "
            f"thresh={thresh_fmt}) "
            f"s_ref_cos={gate_info.get('s_ref_cos')} s_ref_z={gate_info.get('s_ref_z')} "
            f"fired={gate_info.get('fired')} sticky_unlock={gate_info.get('sticky_unlock')} "
            f"sticky_revoked={gate_info.get('sticky_revoked')} source={gate_info.get('source')}",
        )

        # An unrecognised mode falls through to the measured closed loop, mirroring the canonical
        # middleware's bare `else:` -- except that the default is now the measured route rather than
        # the regex one. Only `baseline` passes through. The remote client differs and raises.
        if mode == "baseline":
            route_factory = None
        else:
            try:
                route_factory = route_for_mode(mode) or measured_closed_loop_route
            except ValueError as exc:
                # A mode that cannot run in this configuration. Report it as the caller's error
                # rather than letting it surface as an opaque 500.
                log_cybergate_event("CYBERGATE-MIDDLEWARE-WARN", f"req_id={req_id} {exc}")
                return (
                    {"error": {"message": str(exc), "type": "BadRequestError", "code": 400}},
                    400,
                    [(b"content-type", b"application/json")],
                )

        if route_factory is None or not gate_info["fired"]:
            # Pass through untouched.
            status, headers, chat_out = await self.dispatch.post_json(scope, CHAT_PATH, payload)
            chat_out["x_cybergate"] = {
                **gate_info,
                "request_id": req_id,
                "mode": mode,
                "routed": "pristine_pass_through",
                "think_injected": False,
                "boilerplate_stripped": False,
                "elapsed_s": round(time.time() - started, 3),
            }
            final_chat_obj = chat_out
        else:
            extra_payload: dict[str, Any] = {"add_special_tokens": False}
            if tool_turn:
                extra_payload["vllm_xargs"] = {SUPPRESS_OPENER_XARG: 1}
            config = RouteConfig(
                family=family,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                max_cot_tokens=max_cot_tokens,
                # The routes that write the opener into the prompt themselves must leave it out too,
                # or a tool turn gets it back by the other path.
                include_opener=not tool_turn,
                include_bos=include_bos,
                extra_payload=extra_payload,
            )
            route = route_factory(mode, list(messages), config)
            status, headers, result = await self._drive_route(scope, route)

            if do_sanitize:
                clean_reasoning, did_strip = strip_injected_think_boilerplate(
                    result.raw_reasoning, family
                )
            else:
                clean_reasoning, did_strip = result.raw_reasoning, False

            final_chat_obj = self._wrap_chat_json(
                comp_json=result.response,
                model_id=model_id,
                content=result.content,
                reasoning_content=clean_reasoning,
                x_cybergate={
                    **gate_info,
                    "request_id": req_id,
                    "mode": mode,
                    **result.telemetry,
                    "boilerplate_stripped": did_strip,
                    "raw_reasoning_chars": len(result.raw_reasoning),
                    "clean_reasoning_chars": len(clean_reasoning),
                    "elapsed_s": round(time.time() - started, 3),
                },
            )

        telemetry = final_chat_obj.setdefault("x_cybergate", {})
        final_chat_obj["x_cybergate"] = telemetry
        monitor = self.store.get_monitor()
        if monitor:
            telemetry["l21_gpu_monitor"] = monitor
        log_cybergate_event(
            "CYBERGATE-RESPONSE",
            f"req_id={req_id} mode={mode} routed={telemetry.get('routed')} "
            f"think_injected={telemetry.get('think_injected')} "
            f"boilerplate_stripped={telemetry.get('boilerplate_stripped')} "
            f"l21_s_cos={monitor.get('s_cos') if monitor else 'n/a'} "
            f"probe_http_ms={telemetry.get('probe_http_ms', 0.0)}ms "
            f"elapsed={telemetry.get('elapsed_s')}s",
        )
        return final_chat_obj, status, headers

    async def _drive_route(
        self,
        scope: dict[str, Any],
        route: Any,
    ) -> tuple[int, list[tuple[bytes, bytes]], Any]:
        """Run a think-injection route to completion over the ASGI transport.

        The status and headers returned are those of the **last** completion call, which is the one
        that produced the visible answer.
        """
        status = 200
        headers: list[tuple[bytes, bytes]] = [(b"content-type", b"application/json")]
        request: RouteRequest = next(route)
        while True:
            status, headers, response = await self.dispatch.post_json(
                scope, request.path, request.payload
            )
            try:
                request = route.send(response)
            except StopIteration as stop:
                return status, headers, stop.value

