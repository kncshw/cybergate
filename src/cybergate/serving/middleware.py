"""`CyberGateThinkMiddleware` -- the in-process ASGI middleware.

Runs in the **API-server** process (`MainProcess`), installed with
`vllm serve --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware`. Intercepts
`POST /v1/chat/completions` and does three things:

1. **Probe.** Dispatches 1-token completions back through the ASGI app so the model worker's forward
   hooks observe each user turn, both standalone and in context, and publish decision records. This
   is the only way the API-server process can get a score: the hooks live in the other process.
2. **Decide.** Calls the gate with those records.
3. **Route.** Either forwards the request untouched (`fired=False`), or runs the native
   `/v1/chat/completions` closed-loop steering path (injecting `cybergate_open_think_prefix` via
   `chat_template_kwargs` so `RefusalSteeredCoTProcessor` steers decoding in-engine).

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
from ..families import ModelFamily, family_for_model_id
from ..messages import extract_user_turns
from ..state import clear_request_gate_state, keys as key_builders
from ..state.ipc_shm import ShmDecisionStore
from ..telemetry.log import log_cybergate_event
from .asgi import AsgiDispatcher, replay_body
from .postprocess import (
    _flatten_content,
    estimate_chat_prompt_tokens,
    has_matching_steer_token,
    strip_cybergate_client_kwargs,
    tokens_match_open_think_suffix,
)
from .proxy import start_background_proxy_server
from .think_routes import (
    MEASURED_MODES,
    RouteConfig,
    RouteRequest,
    measured_closed_loop_route,
    native_chat_closed_loop_route,
    route_for_mode,
)

__all__ = ["CyberGateThinkMiddleware"]

CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
TOKENIZE_PATH = "/tokenize"


def _normalize_messages_for_probe(
    probe_messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a chat message list into the clean `user`/`assistant` shape the gate probe expects.

    1. Strips `system` / `developer` messages so `build_raw_prompt` uses `DEFAULT_SYSTEM_PROMPT`
       (matching calibration).
    2. Trims trailing non-`user` messages *after* the final `user` turn (e.g. inside an active
       multi-step agent tool loop) so the rendered prompt ends immediately after the final user
       span (`suffix_len == 9` on Qwen, `5` on Gemma) and `family.is_baseline_prefill(...)` stays
       `True`.
    3. Folds intermediate `assistant(tool_calls)` and `role: "tool"` turns *between* user turns
       into non-empty `role: "assistant"` text summaries so `build_raw_prompt` never emits an empty
       assistant turn or nonexistent `<|im_start|>tool` / `<|turn>tool` headers, while keeping the
       exact `user` turn sequence (`conv_sha256 == conversation_key(user_turns)`).
    """
    non_sys = [
        m
        for m in probe_messages
        if isinstance(m, dict) and m.get("role") not in ("system", "developer")
    ]
    last_user_idx = -1
    for idx, m in enumerate(non_sys):
        if m.get("role") == "user":
            last_user_idx = idx
    if last_user_idx < 0:
        return []
    trimmed = non_sys[: last_user_idx + 1]

    normalized: list[dict[str, Any]] = []
    for m in trimmed:
        role = m.get("role", "")
        if role == "user":
            normalized.append(
                {"role": "user", "content": _flatten_content(m.get("content", ""))}
            )
            continue

        if role == "assistant":
            parts: list[str] = []
            c_txt = _flatten_content(m.get("content", "")).strip()
            if c_txt:
                parts.append(c_txt)
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                    fn_name = fn.get("name") or "tool"
                    fn_args = fn.get("arguments") or "{}"
                    args_s = (
                        fn_args
                        if isinstance(fn_args, str)
                        else json.dumps(fn_args, ensure_ascii=False)
                    )
                    if len(args_s) > 240:
                        args_s = args_s[:240] + "..."
                    parts.append(f"[Tool call: {fn_name}({args_s})]")
            summary = "\n".join(parts).strip()
            if not summary:
                continue
            if normalized and normalized[-1]["role"] == "assistant":
                normalized[-1] = {
                    "role": "assistant",
                    "content": f"{normalized[-1]['content']}\n\n{summary}".strip(),
                }
            else:
                normalized.append({"role": "assistant", "content": summary})
        elif role == "tool":
            t_txt = _flatten_content(m.get("content", "")).strip()
            if len(t_txt) > 600:
                t_txt = t_txt[:400] + " ... " + t_txt[-160:]
            summary = f"[Tool result: {t_txt}]" if t_txt else "[Tool result: ok]"
            if normalized and normalized[-1]["role"] == "assistant":
                normalized[-1] = {
                    "role": "assistant",
                    "content": f"{normalized[-1]['content']}\n{summary}".strip(),
                }
            else:
                normalized.append({"role": "assistant", "content": summary})
    return normalized


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
        self._template_hook_cache: dict[tuple[int, str], bool] = {}
        self._close_tag_tokens_cache: dict[tuple[int, str], tuple[int, ...]] = {}
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
                        "reasoning": reasoning_content,
                    },
                    "finish_reason": choice0.get("finish_reason", "stop"),
                }
            ],
            "usage": comp_json.get("usage", {}),
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

        async def probe(
            probe_messages: Sequence[dict[str, Any]],
            cache_salt: str | None = None,
        ) -> None:
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
            probe_clean = _normalize_messages_for_probe(probe_messages)
            prompt = family_for_model_id(model_id).build_raw_prompt(
                probe_clean, mode="baseline", include_bos=include_bos
            )
            req_payload: dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "add_special_tokens": False,
            }
            if cache_salt:
                req_payload["cache_salt"] = cache_salt
            await self.dispatch.post_json(
                scope,
                COMPLETIONS_PATH,
                req_payload,
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
                    await probe(
                        [{"role": "user", "content": turn_text}],
                        cache_salt=f"cg_st_{sha}",
                    )
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
                await probe(messages, cache_salt=f"cg_ctx_{conv_key}")
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

            def single_turn_record(
                allow_pc: bool = False, min_ts: float = 0.0
            ) -> dict[str, Any] | None:
                rec = (
                    self.store.get(key_builders.standalone_turn_key_from_sha(latest_sha))
                    or self.store.get(key_builders.conversation_key(user_turns))
                    or self.store.get(latest_sha)
                )
                if rec is not None:
                    return rec
                if allow_pc:
                    pc = self.store.get(f"pc_ctx_{latest_sha}")
                    if pc is not None and float(pc.get("ts", 0.0)) >= min_ts:
                        return pc
                return None

            in_context = single_turn_record()
            if in_context is None or in_context.get("s_ref_z") is None:
                did_probe = True
                probe_t0 = time.time() - 0.05
                await probe(messages, cache_salt=f"cg_st_{latest_sha}")
                in_context = single_turn_record(allow_pc=True, min_ts=probe_t0)
                if in_context is None or in_context.get("s_ref_z") is None:
                    await probe(
                        [{"role": "user", "content": user_turns[-1]}],
                        cache_salt=f"cg_st_{latest_sha}",
                    )
                    in_context = single_turn_record(allow_pc=True, min_ts=probe_t0)
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
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        req_path = str(scope.get("path") or "").rstrip("/") or "/"
        if req_path != CHAT_PATH:
            if req_path in (TOKENIZE_PATH, "/detokenize", "/v1/embeddings", "/embeddings"):
                await self.app(scope, receive, send)
                return
            if str(scope.get("method", "")).upper() == "POST":
                other_chunks: list[bytes] = []
                while True:
                    msg_ev = await receive()
                    other_chunks.append(msg_ev.get("body", b""))
                    if not msg_ev.get("more_body", False):
                        break
                other_raw = b"".join(other_chunks)
                try:
                    other_obj = json.loads(other_raw.decode("utf-8"))
                except Exception:
                    other_obj = None
                if isinstance(other_obj, dict) and strip_cybergate_client_kwargs(
                    other_obj,
                    allow_top_level_mode=False,
                    expected_steer_token=env_first("CYBERGATE_STEER_TOKEN"),
                ):
                    other_raw = json.dumps(other_obj).encode("utf-8")
                    scope = {
                        **scope,
                        "headers": [
                            (k, v)
                            for (k, v) in scope.get("headers", [])
                            if k.lower() != b"content-length"
                        ]
                        + [(b"content-length", str(len(other_raw)).encode("ascii"))],
                    }
                await self.app(scope, replay_body(other_raw, receive), send)
                return
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
            await self.app(scope, replay_body(raw_body, receive), send)
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
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        tool_calls = msg.get("tool_calls") or []
        finish_reason = choice0.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
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
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": reasoning,
                            "reasoning": reasoning,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            chunks.append(f"data: {json.dumps(c_reas)}\n\n")

        if content or not tool_calls:
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

        if tool_calls:
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
                                {"index": idx, **tc} for idx, tc in enumerate(tool_calls)
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
            chunks.append(f"data: {json.dumps(c_tc)}\n\n")

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

    async def _verify_template_hook(
        self,
        scope: dict[str, Any],
        model_id: str,
        family: ModelFamily,
    ) -> bool:
        """Verify once per `(dispatcher, model_id)` via `/tokenize` integer token IDs that the
        active chat template renders `cybergate_open_think_prefix` as the exact prompt suffix
        `RefusalSteeredCoTProcessor._should_steer` matches.

        Transient `/tokenize` errors are NOT cached and never fall back to disk-file existence.
        """
        cache_key = (id(self.dispatch), model_id)
        cached = self._template_hook_cache.get(cache_key)
        if cached is not None:
            return cached

        matched: bool | None = None
        try:
            s_suf, _, r_suf = await self.dispatch.post_json(
                scope,
                TOKENIZE_PATH,
                {
                    "model": model_id,
                    "prompt": f"{family.thought_open}{family.open_think_prefix}",
                    "add_special_tokens": False,
                },
            )
            s_msg, _, r_msg = await self.dispatch.post_json(
                scope,
                TOKENIZE_PATH,
                {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "cybergate_open_think_prefix": family.open_think_prefix,
                    },
                },
            )
            if (
                s_suf == 200
                and s_msg == 200
                and isinstance(r_suf, dict)
                and isinstance(r_msg, dict)
            ):
                matched = tokens_match_open_think_suffix(
                    r_msg.get("tokens"), r_suf.get("tokens")
                )
                if matched and family.thought_close_tag:
                    s_cls, _, r_cls = await self.dispatch.post_json(
                        scope,
                        TOKENIZE_PATH,
                        {
                            "model": model_id,
                            "prompt": family.thought_close_tag,
                            "add_special_tokens": False,
                        },
                    )
                    if (
                        s_cls == 200
                        and isinstance(r_cls, dict)
                        and isinstance(r_cls.get("tokens"), list)
                        and r_cls["tokens"]
                    ):
                        self._close_tag_tokens_cache[cache_key] = tuple(
                            int(t) for t in r_cls["tokens"]
                        )
        except Exception:
            matched = None

        if matched is None:
            return False

        self._template_hook_cache[cache_key] = matched
        return matched

    async def _resolve_effective_prefill_tokens(
        self,
        scope: dict[str, Any],
        model_id: str,
        messages: Sequence[dict[str, Any]],
        tools: Any,
        in_context_record: dict[str, Any] | None,
        max_cot_tokens: int,
        max_tokens: int,
    ) -> tuple[int, int]:
        """Return `(effective_prefill_tokens, max_model_len)` accounting for `system` and `tools`."""
        max_model_len = int(env_first("CYBERGATE_MAX_MODEL_LEN", default="262144") or 262144)
        probe_prefill = (
            int(in_context_record["n_prefill_tokens"])
            if (in_context_record and in_context_record.get("n_prefill_tokens"))
            else 0
        )
        tool_list = tools if isinstance(tools, list) else None
        est_tokens = estimate_chat_prompt_tokens(list(messages), tool_list)
        effective = max(probe_prefill, est_tokens)

        has_client_sys = any(
            isinstance(m, dict) and m.get("role") in ("system", "developer") for m in messages
        )
        if (tool_list or has_client_sys) and (
            effective + max_cot_tokens + max_tokens + 128 > max_model_len // 2
        ):
            tok_req: dict[str, Any] = {"model": model_id, "messages": list(messages)}
            if tool_list:
                tok_req["tools"] = tool_list
            try:
                s_tok, _, r_tok = await self.dispatch.post_json(scope, TOKENIZE_PATH, tok_req)
                if s_tok == 200 and isinstance(r_tok, dict):
                    if "count" in r_tok:
                        effective = max(probe_prefill, int(r_tok["count"]))
                    if r_tok.get("max_model_len"):
                        max_model_len = int(r_tok["max_model_len"])
            except Exception:
                pass

        return effective, max_model_len

    async def _handle_chat(
        self,
        scope: dict[str, Any],
        payload: dict[str, Any],
        req_id: str,
        started: float,
    ) -> tuple[dict[str, Any], int, list[tuple[bytes, bytes]]]:
        """Probe, decide and route one chat request."""
        expected_steer_tok = env_first("CYBERGATE_STEER_TOKEN")
        if (
            has_matching_steer_token(payload, expected_steer_tok)
            and isinstance(payload.get("chat_template_kwargs"), dict)
            and payload["chat_template_kwargs"].get("cybergate_open_think_prefix")
        ):
            status, headers, chat_out = await self.dispatch.post_json(scope, CHAT_PATH, payload)
            return chat_out, status, headers

        strip_cybergate_client_kwargs(payload, expected_steer_token=expected_steer_tok)
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
        # produce one. `cybergate_tool_turn` is how the proxy says so; a caller that passes `tools`
        # through is recognised directly.
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

        max_total_tokens: int | None = None
        has_context_signal = bool(
            (in_context_record and in_context_record.get("n_prefill_tokens"))
            or payload.get("tools")
            or any(
                isinstance(m, dict) and m.get("role") in ("system", "developer")
                for m in messages
            )
        )
        if has_context_signal:
            n_prefill, max_model_len = await self._resolve_effective_prefill_tokens(
                scope,
                model_id,
                messages,
                payload.get("tools"),
                in_context_record,
                max_cot_tokens,
                max_tokens,
            )
            # Reserve 128 tokens for OPEN_THINK_PREFIX (~60 tokens) + special-token margin
            max_total_tokens = max(64, max_model_len - n_prefill - 128)
            max_avail = max(64, max_model_len - n_prefill - max_cot_tokens - 128)
            if not caller_max_tokens:
                max_tokens = min(default_max_tokens, max(64, max_avail))
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

        is_native_closed_loop = False
        if mode == "baseline":
            route_factory = None
        elif mode in MEASURED_MODES:
            is_native_closed_loop = True
            route_factory = None
        else:
            try:
                route_factory = route_for_mode(mode)
                if route_factory is None:
                    is_native_closed_loop = True
            except ValueError as exc:
                log_cybergate_event("CYBERGATE-MIDDLEWARE-WARN", f"req_id={req_id} {exc}")
                return (
                    {"error": {"message": str(exc), "type": "BadRequestError", "code": 400}},
                    400,
                    [(b"content-type", b"application/json")],
                )

        template_fallback_used = False
        if gate_info["fired"] and is_native_closed_loop:
            has_hook = await self._verify_template_hook(scope, model_id, family)
            if not has_hook:
                if tool_turn:
                    msg_err = (
                        f"CyberGate closed_loop with tools requires vLLM to be started with the "
                        f"kit --chat-template for model {model_id!r} (cybergate_open_think_prefix "
                        "was ignored by the active chat template)."
                    )
                    log_cybergate_event("CYBERGATE-MIDDLEWARE-WARN", f"req_id={req_id} {msg_err}")
                    return (
                        {
                            "error": {
                                "message": msg_err,
                                "type": "ServerConfigurationError",
                                "code": 500,
                            }
                        },
                        500,
                        [(b"content-type", b"application/json")],
                    )
                log_cybergate_event(
                    "CYBERGATE-MIDDLEWARE-WARN",
                    f"req_id={req_id} Active chat template for {model_id!r} ignored "
                    "cybergate_open_think_prefix (missing --chat-template?); falling back to "
                    "/v1/completions measured_closed_loop_route",
                )
                is_native_closed_loop = False
                route_factory = measured_closed_loop_route
                template_fallback_used = True

        if not gate_info["fired"] or (route_factory is None and not is_native_closed_loop):
            # Pass through untouched.
            status, headers, chat_out = await self.dispatch.post_json(scope, CHAT_PATH, payload)
            choices = chat_out.get("choices")
            if isinstance(choices, list):
                for ch_item in choices:
                    if isinstance(ch_item, dict):
                        msg_i = ch_item.get("message")
                        if isinstance(msg_i, dict) and "reasoning_content" not in msg_i:
                            msg_i["reasoning_content"] = msg_i.get("reasoning") or ""
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
        elif is_native_closed_loop:
            native_cfg = RouteConfig(
                family=family,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                max_cot_tokens=max_cot_tokens,
                include_opener=not tool_turn,
                include_bos=include_bos,
                chat_completions_path=CHAT_PATH,
                chat_payload=payload,
                max_total_tokens=max_total_tokens,
                sanitize_reasoning=do_sanitize,
                close_tag_token_ids=self._close_tag_tokens_cache.get(
                    (id(self.dispatch), model_id)
                ),
            )
            route = native_chat_closed_loop_route(mode, list(messages), native_cfg)
            status, headers, result = await self._drive_route(scope, route)
            final_chat_obj = dict(result.response)
            if status >= 400 or not isinstance(final_chat_obj.get("choices"), list):
                final_chat_obj["x_cybergate"] = {
                    **gate_info,
                    "request_id": req_id,
                    "mode": mode,
                    "routed": "upstream_error",
                    "think_injected": False,
                    "boilerplate_stripped": False,
                    "upstream_status": status,
                    "elapsed_s": round(time.time() - started, 3),
                }
            else:
                final_chat_obj["x_cybergate"] = {
                    **gate_info,
                    "request_id": req_id,
                    "mode": mode,
                    **result.telemetry,
                    "elapsed_s": round(time.time() - started, 3),
                }
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
                include_opener=not tool_turn,
                include_bos=include_bos,
                extra_payload=extra_payload,
            )
            assert route_factory is not None
            route = route_factory(mode, list(messages), config)
            status, headers, result = await self._drive_route(scope, route)

            if status >= 400 or not isinstance(result.response.get("choices"), list):
                final_chat_obj = dict(result.response)
                final_chat_obj["x_cybergate"] = {
                    **gate_info,
                    "request_id": req_id,
                    "mode": mode,
                    "routed": "upstream_error",
                    "think_injected": False,
                    "boilerplate_stripped": False,
                    "upstream_status": status,
                    "elapsed_s": round(time.time() - started, 3),
                }
            else:
                if do_sanitize:
                    clean_reasoning, did_strip = strip_injected_think_boilerplate(
                        result.raw_reasoning, family
                    )
                else:
                    clean_reasoning, did_strip = result.raw_reasoning, False

                extra_tele: dict[str, Any] = {}
                if template_fallback_used:
                    extra_tele["template_hook_fallback"] = "completions"

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
                        **extra_tele,
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
