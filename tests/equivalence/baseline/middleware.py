"""ASGI Middleware (`ResGateThinkMiddleware`) and Remote Gateway Client (`ResGateVLLMClient`)
for Conditional `<think>` Prefill (`TH3`), Open-`<think>` Sandwich (`SW3`), and Closed-Loop
Cosine-Monitored `<think>` Steering (`CL3` / `CL2`) in `vLLM` (`K = 0.0`).

Includes the 4 Production Features + In-Process `vllm serve` Live GPU IPC Bridge:
  1. Per-Request Batch Isolation (`request_id` -> `REQUEST_GATE_STATES` & `/dev/shm/resgate_live_state.json`).
  2. Multi-Turn Conversation Stickiness + Cross-Category Safety Revocation (`evaluate_gate_for_messages`).
  3. Clean Thought-Stream Boilerplate Masking (`strip_injected_think_boilerplate`, `sanitize_reasoning=True`).
  4. Full support for `mode="baseline"` (`TH0`), `"preclosed"` (`TH3`), `"sandwich"` (`SW3`), and
     `"closed_loop"` (`CL3`) in BOTH `ResGateThinkMiddleware` (in-process `vllm serve`) and `ResGateVLLMClient`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Sequence

from .gate_probe import (
    clear_request_gate_state,
    compute_conv_sha256,
    evaluate_gate_for_messages,
    log_cybergate_event,
    log_resgate_event,
    read_l21_monitor_ipc,
    read_prompt_decision_ipc,
)
from .prompts import (
    BOTTOM_COMPLIANCE_LOCK,
    DEFAULT_SYSTEM_PROMPT,
    FAKE_THINK_RUBRIC,
    GEMMA4_PRESERVE_THOUGHT_CHAT_TEMPLATE,
    INLINE_COT_SUPPRESSION,
    INLINE_FINAL_STEP_CLEARANCE,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
    build_raw_prompt_for_model,
    detect_model_family,
    extract_system_and_user,
    strip_injected_think_boilerplate,
    truncate_think_at_second_guess,
)


class CyberGateVLLMClient:
    """OpenAI-compatible `vLLM` client with Conditional `CyberGate` + `<think>` Prefill/Sandwich/Closed-Loop."""

    def __init__(
        self,
        base_url: str = "https://192.0.2.39/v1",
        api_key: str | None = None,
        model_id: str = "google/gemma-4-26B-A4B-it",
        default_mode: str = "preclosed",
        resgate_http_url: str | None = None,
        verify_ssl: bool = False,
        timeout: int = 180,
        sanitize_reasoning: bool = True,
        enable_multi_turn_stickiness: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("VLLM_API_KEY")
        self.model_id = model_id
        self.default_mode = default_mode
        self.resgate_http_url = resgate_http_url
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.sanitize_reasoning = sanitize_reasoning
        self.enable_multi_turn_stickiness = enable_multi_turn_stickiness

        self._ssl_ctx = ssl.create_default_context()
        if not verify_ssl:
            self._ssl_ctx.check_hostname = False
            self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> dict[str, Any]:
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=30, context=self._ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat_completions(
        self,
        messages: Sequence[dict[str, Any]],
        mode: str | None = None,
        max_tokens: int = 350,
        temperature: float = 0.0,
        max_cot_tokens: int = 180,
        prompt_key: str | None = None,
        force_gate: bool | None = None,
        include_opener: bool = True,
        sanitize_reasoning: bool | None = None,
        enable_multi_turn_stickiness: bool | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a ResGate-gated chat completion against `self.base_url`."""
        active_mode = (mode or self.default_mode).lower()
        do_sanitize = self.sanitize_reasoning if sanitize_reasoning is None else sanitize_reasoning
        do_sticky = (
            self.enable_multi_turn_stickiness
            if enable_multi_turn_stickiness is None
            else enable_multi_turn_stickiness
        )
        req_id = request_id or f"req-{uuid.uuid4().hex[:10]}"
        t_start = time.time()
        family = detect_model_family(self.model_id)
        opener = POST_THINK_OPENER if include_opener else ""

        try:
            gate_info = evaluate_gate_for_messages(
                messages=messages,
                model_id=self.model_id,
                prompt_key=prompt_key,
                resgate_http_url=self.resgate_http_url,
                vllm_base_url=self.base_url,
                vllm_api_key=self.api_key,
                verify_ssl=self.verify_ssl,
                request_id=req_id,
                enable_multi_turn_stickiness=do_sticky,
            )
            if force_gate is not None:
                gate_info = dict(gate_info)
                gate_info["fired"] = bool(force_gate)
                gate_info["source"] = f"{gate_info.get('source', 'probe')}+forced({force_gate})"

            if active_mode == "baseline" or not gate_info["fired"]:
                raw_resp = self._post_json(
                    "/chat/completions",
                    {
                        "model": self.model_id,
                        "messages": list(messages),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                raw_resp["x_resgate"] = {
                    **gate_info,
                    "request_id": req_id,
                    "mode": active_mode,
                    "routed": "pristine_pass_through",
                    "think_injected": False,
                    "boilerplate_stripped": False,
                    "elapsed_s": round(time.time() - t_start, 3),
                }
                return raw_resp

            if active_mode in ("preclosed", "th3", "th2", "preclosed_no_opener"):
                use_opener = include_opener and active_mode not in ("th2", "preclosed_no_opener")
                sub_mode = "preclosed" if use_opener else "preclosed_no_opener"
                raw_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=self.model_id,
                    mode=sub_mode,
                    post_think_opener=opener if use_opener else "",
                    include_bos=False,
                )
                stop_tokens = ["<turn|>", "<eos>"] if family == "gemma4" else ["<|im_end|>"]
                comp_resp = self._post_json(
                    "/completions",
                    {
                        "model": self.model_id,
                        "prompt": raw_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens,
                    },
                )
                gen_text = comp_resp["choices"][0]["text"]
                full_visible = (opener + gen_text) if use_opener else gen_text
                raw_reasoning = FAKE_THINK_RUBRIC
                if do_sanitize:
                    clean_reasoning, did_strip = strip_injected_think_boilerplate(raw_reasoning)
                else:
                    clean_reasoning, did_strip = raw_reasoning, False

                return self._wrap_completion_as_chat(
                    comp_resp=comp_resp,
                    content=full_visible,
                    reasoning_content=clean_reasoning,
                    x_resgate={
                        **gate_info,
                        "request_id": req_id,
                        "mode": active_mode,
                        "routed": "resgate_preclosed_think_injected",
                        "think_injected": True,
                        "opener_injected": use_opener,
                        "boilerplate_stripped": did_strip,
                        "raw_reasoning_chars": len(raw_reasoning),
                        "clean_reasoning_chars": len(clean_reasoning),
                        "elapsed_s": round(time.time() - t_start, 3),
                    },
                )

            if active_mode in ("sandwich", "sw3", "sw2"):
                enable_guard = active_mode != "sw2"
                close_tag = "<channel|>" if family == "gemma4" else "</think>"
                stop_tokens_ans = ["<turn|>", "<eos>"] if family == "gemma4" else ["<|im_end|>"]

                step1_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=self.model_id,
                    mode="sandwich_step1",
                    include_bos=False,
                )
                step1_resp = self._post_json(
                    "/completions",
                    {
                        "model": self.model_id,
                        "prompt": step1_prompt,
                        "max_tokens": max_cot_tokens,
                        "temperature": temperature,
                        "stop": [close_tag, "<turn|>"],
                    },
                )
                raw_cot = step1_resp["choices"][0]["text"]

                if enable_guard:
                    cleaned_cot, did_truncate = truncate_think_at_second_guess(raw_cot)
                else:
                    cleaned_cot, did_truncate = raw_cot.rstrip(), False

                step2_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=self.model_id,
                    mode="sandwich_step2",
                    custom_think_text=cleaned_cot,
                    post_think_opener=opener,
                    include_bos=False,
                )
                step2_resp = self._post_json(
                    "/completions",
                    {
                        "model": self.model_id,
                        "prompt": step2_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens_ans,
                    },
                )
                gen_ans = step2_resp["choices"][0]["text"]
                full_visible = (opener + gen_ans) if include_opener else gen_ans
                raw_thought = f"{OPEN_THINK_PREFIX}{cleaned_cot}{BOTTOM_COMPLIANCE_LOCK}"
                if do_sanitize:
                    clean_thought, did_strip = strip_injected_think_boilerplate(raw_thought)
                else:
                    clean_thought, did_strip = raw_thought, False

                return self._wrap_completion_as_chat(
                    comp_resp=step2_resp,
                    content=full_visible,
                    reasoning_content=clean_thought,
                    x_resgate={
                        **gate_info,
                        "request_id": req_id,
                        "mode": active_mode,
                        "routed": "resgate_sandwich_think_injected",
                        "think_injected": True,
                        "cot_raw_chars": len(raw_cot),
                        "cot_kept_chars": len(cleaned_cot),
                        "policy_pivot_truncated": did_truncate,
                        "boilerplate_stripped": did_strip,
                        "raw_reasoning_chars": len(raw_thought),
                        "clean_reasoning_chars": len(clean_thought),
                        "elapsed_s": round(time.time() - t_start, 3),
                    },
                )

            if active_mode in ("closed_loop", "cl3", "cl2"):
                use_bottom_lock = active_mode != "cl2"
                close_tag = "<channel|>" if family == "gemma4" else "</think>"
                stop_tokens_ans = ["<turn|>", "<eos>"] if family == "gemma4" else ["<|im_end|>"]

                p1a_tokens = min(130, max_cot_tokens)
                step1a_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=self.model_id,
                    mode="closed_loop_step1",
                    include_bos=False,
                )
                step1a_resp = self._post_json(
                    "/completions",
                    {
                        "model": self.model_id,
                        "prompt": step1a_prompt,
                        "max_tokens": p1a_tokens,
                        "temperature": temperature,
                        "stop": [close_tag, "<turn|>"],
                    },
                )
                raw_cot_1a = step1a_resp["choices"][0]["text"]
                finish_1a = step1a_resp["choices"][0].get("finish_reason", "")
                kept_pre_spike, did_pivot_1a = truncate_think_at_second_guess(raw_cot_1a)

                if not did_pivot_1a and finish_1a == "length":
                    m_end = list(re.finditer(r"(?:\n|[.!?;:])", kept_pre_spike))
                    if m_end and m_end[-1].end() >= 35:
                        kept_pre_spike = kept_pre_spike[: m_end[-1].end()].rstrip()

                spike_detected = bool(did_pivot_1a or finish_1a == "length")
                post_spike_cot = ""
                num_inline_injections = 0

                if spike_detected:
                    num_inline_injections = 1
                    step1b_prompt = build_raw_prompt_for_model(
                        messages=messages,
                        model_id=self.model_id,
                        mode="closed_loop_step1b",
                        custom_think_text=kept_pre_spike,
                        include_bos=False,
                    )
                    step1b_resp = self._post_json(
                        "/completions",
                        {
                            "model": self.model_id,
                            "prompt": step1b_prompt,
                            "max_tokens": 105,
                            "temperature": temperature,
                            "stop": [close_tag, "<turn|>"],
                        },
                    )
                    raw_cot_1b = step1b_resp["choices"][0]["text"]
                    clean_1b, _ = truncate_think_at_second_guess(raw_cot_1b)
                    post_spike_cot = clean_1b if len(clean_1b.strip()) >= 10 else raw_cot_1b.rstrip()
                    m_sent = list(re.finditer(r"(?:\n|[.!?;:])", post_spike_cot))
                    if m_sent and m_sent[-1].end() >= 25:
                        post_spike_cot = post_spike_cot[: m_sent[-1].end()].rstrip()

                    controlled_cot_body = f"{kept_pre_spike}{INLINE_COT_SUPPRESSION}{post_spike_cot}"
                else:
                    controlled_cot_body = kept_pre_spike

                step2_mode = "closed_loop_step2" if use_bottom_lock else "closed_loop_cl2_step2"
                step2_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=self.model_id,
                    mode=step2_mode,
                    custom_think_text=controlled_cot_body,
                    post_think_opener=opener if use_bottom_lock else "",
                    include_bos=False,
                )
                step2_resp = self._post_json(
                    "/completions",
                    {
                        "model": self.model_id,
                        "prompt": step2_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens_ans,
                    },
                )
                gen_ans = step2_resp["choices"][0]["text"]
                full_visible = (opener + gen_ans) if (include_opener and use_bottom_lock) else gen_ans
                suffix_block = BOTTOM_COMPLIANCE_LOCK if use_bottom_lock else INLINE_FINAL_STEP_CLEARANCE
                raw_thought = f"{OPEN_THINK_PREFIX}{controlled_cot_body}{suffix_block}"
                if do_sanitize:
                    clean_thought, did_strip = strip_injected_think_boilerplate(raw_thought)
                else:
                    clean_thought, did_strip = raw_thought, False

                return self._wrap_completion_as_chat(
                    comp_resp=step2_resp,
                    content=full_visible,
                    reasoning_content=clean_thought,
                    x_resgate={
                        **gate_info,
                        "request_id": req_id,
                        "mode": active_mode,
                        "routed": "resgate_closed_loop_cot_injected",
                        "think_injected": True,
                        "spike_detected": spike_detected,
                        "num_inline_injections": num_inline_injections + (0 if use_bottom_lock else 1),
                        "cot_pre_spike_chars": len(kept_pre_spike),
                        "cot_post_spike_chars": len(post_spike_cot),
                        "policy_pivot_truncated": did_pivot_1a,
                        "boilerplate_stripped": did_strip,
                        "raw_reasoning_chars": len(raw_thought),
                        "clean_reasoning_chars": len(clean_thought),
                        "elapsed_s": round(time.time() - t_start, 3),
                    },
                )

            raise ValueError(f"Unsupported mode: {active_mode!r}")
        finally:
            clear_request_gate_state(req_id)

    def _wrap_completion_as_chat(
        self,
        comp_resp: dict[str, Any],
        content: str,
        reasoning_content: str,
        x_resgate: dict[str, Any],
    ) -> dict[str, Any]:
        """Format a `/v1/completions` response into a standard OpenAI `chat.completion` object."""
        choice0 = comp_resp.get("choices", [{}])[0]
        return {
            "id": comp_resp.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion",
            "created": comp_resp.get("created", int(time.time())),
            "model": comp_resp.get("model", self.model_id),
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
            "usage": comp_resp.get("usage", {}),
            "x_cybergate": x_resgate,
            "x_resgate": x_resgate,
        }


ResGateVLLMClient = CyberGateVLLMClient


class CyberGateThinkMiddleware:
    """In-Process ASGI Middleware for `vllm serve --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware`.

    Intercepts `POST /v1/chat/completions` inside `vllm serve`:
      1. **Pass 1 (Live GPU Layer-8 Prefill Probe):** Dispatches a 1-token prefill probe
         through `self.app` so the `vLLM` GPU Worker executes `_read_only_hook` on `layers[8]`
         and writes the exact live GPU cosine score `s_cat` to `/dev/shm/resgate_live_state.json`.
      2. **Multi-Turn & Cross-Category Evaluation:** Combines the live GPU `Layer 8` score with
         `evaluate_gate_for_messages` (Multi-Turn Stickiness + Cross-Category Safety Revocation).
      3. **Pass 2 (Conditional Routing):**
         - If `fired == False` or `resgate_mode == "baseline"`: forwards through `self.app`
           untouched (`routed = "pristine_pass_through"`).
         - If `fired == True`: executes `"preclosed"` (`TH3`), `"sandwich"` (`SW3`), or
           `"closed_loop"` (`CL3`) against `self.app` (`/v1/completions`), masks injected
           boilerplate (`strip_injected_think_boilerplate`), and returns a standard OpenAI
           `chat.completion` JSON response with full `x_resgate` telemetry.
    """

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app
        self.default_mode = os.environ.get("CYBERGATE_MODE") or os.environ.get("RESGATE_MODE", "closed_loop")
        self.sanitize_reasoning = (
            os.environ.get("CYBERGATE_SANITIZE_REASONING")
            or os.environ.get("RESGATE_SANITIZE_REASONING", "1")
        ) != "0"
        log_resgate_event(
            "CYBERGATE-MIDDLEWARE",
            f"Initialized CyberGateThinkMiddleware (default_mode={self.default_mode}, sanitize_reasoning={self.sanitize_reasoning})",
        )

    async def _asgi_post_json(
        self,
        base_scope: dict[str, Any],
        path: str,
        payload: dict[str, Any],
    ) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
        """Execute an internal ASGI POST request against `self.app` and parse its JSON response."""
        body_bytes = json.dumps(payload).encode("utf-8")
        orig_headers = [
            (k, v)
            for (k, v) in base_scope.get("headers", [])
            if k.lower() not in (b"content-length", b"content-type")
        ]
        orig_headers.append((b"content-type", b"application/json"))
        orig_headers.append((b"content-length", str(len(body_bytes)).encode("ascii")))

        clean_keys = {
            "type",
            "asgi",
            "http_version",
            "server",
            "client",
            "scheme",
            "method",
            "root_path",
            "query_string",
            "app",
            "extensions",
        }
        sub_scope = {k: v for k, v in base_scope.items() if k in clean_keys}
        sub_scope["path"] = path
        sub_scope["raw_path"] = path.encode("ascii")
        sub_scope["headers"] = orig_headers

        disconnect_event = asyncio.Event()
        sent = False

        async def _sub_receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        resp_status = 200
        resp_headers: list[tuple[bytes, bytes]] = []
        resp_chunks: list[bytes] = []

        async def _sub_send(message: dict[str, Any]) -> None:
            nonlocal resp_status, resp_headers
            if message["type"] == "http.response.start":
                resp_status = message.get("status", 200)
                resp_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                resp_chunks.append(message.get("body", b""))

        try:
            await self.app(sub_scope, _sub_receive, _sub_send)
        finally:
            disconnect_event.set()
        raw_out = b"".join(resp_chunks)
        try:
            parsed = json.loads(raw_out.decode("utf-8")) if raw_out else {}
            if not isinstance(parsed, dict):
                parsed = {"_raw_value": parsed}
        except Exception:
            parsed = {"_raw_text": raw_out.decode("utf-8", errors="ignore")}
        return resp_status, resp_headers, parsed

    @staticmethod
    def _wrap_chat_json(
        comp_json: dict[str, Any],
        model_id: str,
        content: str,
        reasoning_content: str,
        x_resgate: dict[str, Any],
    ) -> dict[str, Any]:
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
            "x_cybergate": x_resgate,
            "x_resgate": x_resgate,
        }

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return

        body_chunks: list[bytes] = []
        while True:
            msg = await receive()
            body_chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
        raw_body = b"".join(body_chunks)
        req_id = f"req-{uuid.uuid4().hex[:10]}"
        t_start = time.time()

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            sent_err = False

            async def _err_recv() -> dict[str, Any]:
                nonlocal sent_err
                if not sent_err:
                    sent_err = True
                    return {"type": "http.request", "body": raw_body, "more_body": False}
                return {"type": "http.disconnect"}

            await self.app(scope, _err_recv, send)
            return

        try:
            messages = payload.get("messages", [])
            req_model_id = payload.get("model", "google/gemma-4-26B-A4B-it")
            raw_mode = payload.pop("cybergate_mode", None) or payload.pop("resgate_mode", self.default_mode)
            req_mode = str(raw_mode).lower()
            do_sticky = bool(payload.pop("enable_multi_turn_stickiness", True))
            do_sanitize = bool(payload.pop("sanitize_reasoning", self.sanitize_reasoning))
            max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 350)
            temperature = float(payload.get("temperature", 0.0))
            max_cot_tokens = int(payload.pop("max_cot_tokens", 180))
            family = detect_model_family(req_model_id)
            _, latest_user_text, _ = extract_system_and_user(messages)
            user_turns: list[str] = []
            for m_item in messages:
                if m_item.get("role") == "user":
                    c_val = m_item.get("content", "")
                    if isinstance(c_val, list):
                        c_val = " ".join(
                            str(p.get("text", "")) for p in c_val if isinstance(p, dict)
                        )
                    t_clean = str(c_val).strip()
                    if t_clean:
                        user_turns.append(t_clean)

            # Pass 1: Look up or trigger 1-token Live GPU Prefill Probe (Layer 10 CyberGate + Layer 21 Refusal Monitor)
            live_gpu_dec: dict[str, Any] | None = None
            standalone_gpu_dec: dict[str, Any] | None = None
            probe_http_ms = 0.0
            if user_turns:
                conv_sha256 = compute_conv_sha256(user_turns)
                user_sha256 = hashlib.sha256(user_turns[-1].encode("utf-8")).hexdigest()
                t_probe0 = time.perf_counter()
                did_probe = False

                if len(user_turns) > 1:
                    for u_txt in user_turns:
                        u_sha = hashlib.sha256(u_txt.encode("utf-8")).hexdigest()
                        u_st_dec = (
                            read_prompt_decision_ipc(f"st_{u_sha}")
                            or read_prompt_decision_ipc(u_sha)
                        )
                        if u_st_dec is None or u_st_dec.get("s_ref_z") is None:
                            did_probe = True
                            st_probe_Prompt_msgs = [{"role": "user", "content": u_txt}]
                            st_probe_prompt = build_raw_prompt_for_model(
                                messages=st_probe_Prompt_msgs,
                                model_id=req_model_id,
                                mode="baseline",
                                include_bos=(family == "gemma4"),
                            )
                            await self._asgi_post_json(
                                scope,
                                "/v1/completions",
                                {
                                    "model": req_model_id,
                                    "prompt": st_probe_prompt,
                                    "max_tokens": 1,
                                    "temperature": 0.0,
                                    "add_special_tokens": False,
                                },
                            )
                    standalone_gpu_dec = (
                        read_prompt_decision_ipc(f"st_{user_sha256}")
                        or read_prompt_decision_ipc(user_sha256)
                    )

                    live_gpu_dec = (
                        read_prompt_decision_ipc(f"ctx_{user_sha256}")
                        or read_prompt_decision_ipc(conv_sha256)
                    )
                    if live_gpu_dec is None or live_gpu_dec.get("s_ref_z") is None:
                        did_probe = True
                        conv_probe_prompt = build_raw_prompt_for_model(
                            messages=messages,
                            model_id=req_model_id,
                            mode="baseline",
                            include_bos=(family == "gemma4"),
                        )
                        await self._asgi_post_json(
                            scope,
                            "/v1/completions",
                            {
                                "model": req_model_id,
                                "prompt": conv_probe_prompt,
                                "max_tokens": 1,
                                "temperature": 0.0,
                                "add_special_tokens": False,
                            },
                        )
                        live_gpu_dec = (
                            read_prompt_decision_ipc(f"ctx_{user_sha256}")
                            or read_prompt_decision_ipc(conv_sha256)
                        )
                else:
                    live_gpu_dec = (
                        read_prompt_decision_ipc(f"st_{user_sha256}")
                        or read_prompt_decision_ipc(conv_sha256)
                        or read_prompt_decision_ipc(user_sha256)
                    )
                    if live_gpu_dec is None or live_gpu_dec.get("s_ref_z") is None:
                        did_probe = True
                        conv_probe_prompt = build_raw_prompt_for_model(
                            messages=messages,
                            model_id=req_model_id,
                            mode="baseline",
                            include_bos=(family == "gemma4"),
                        )
                        await self._asgi_post_json(
                            scope,
                            "/v1/completions",
                            {
                                "model": req_model_id,
                                "prompt": conv_probe_prompt,
                                "max_tokens": 1,
                                "temperature": 0.0,
                                "add_special_tokens": False,
                            },
                        )
                        live_gpu_dec = (
                            read_prompt_decision_ipc(f"st_{user_sha256}")
                            or read_prompt_decision_ipc(conv_sha256)
                            or read_prompt_decision_ipc(user_sha256)
                        )
                    standalone_gpu_dec = live_gpu_dec

                if did_probe:
                    probe_http_ms = round((time.perf_counter() - t_probe0) * 1000.0, 2)

            gate_info = evaluate_gate_for_messages(
                messages=messages,
                model_id=req_model_id,
                request_id=req_id,
                enable_multi_turn_stickiness=do_sticky,
                live_gpu_decision=live_gpu_dec,
                standalone_user_decision=standalone_gpu_dec,
            )
            gate_info["probe_http_ms"] = probe_http_ms

            log_resgate_event(
                "CYBERGATE-MIDDLEWARE",
                f"req_id={req_id} mode={req_mode} turn={gate_info.get('user_turn_index', 1)} "
                f"s_cat={gate_info.get('score', 0.0):+.6f} (in_ctx={gate_info.get('s_cat_in_context', 0.0):+.6f}, "
                f"st={gate_info.get('s_cat_standalone', 0.0):+.6f}, conv={gate_info.get('s_cat_conv', 0.0):+.6f}, "
                f"thresh={gate_info.get('threshold', 0.0):+.6f}) "
                f"s_ref_cos={gate_info.get('s_ref_cos')} s_ref_z={gate_info.get('s_ref_z')} "
                f"fired={gate_info.get('fired')} sticky_unlock={gate_info.get('sticky_unlock')} "
                f"sticky_revoked={gate_info.get('sticky_revoked')} source={gate_info.get('source')}",
            )

            resp_status = 200
            resp_headers: list[tuple[bytes, bytes]] = [(b"content-type", b"application/json")]
            final_chat_obj: dict[str, Any] = {}

            # Pass 2A: Baseline or Gate Held (fired == False) -> Pristine Pass-Through
            if req_mode == "baseline" or not gate_info["fired"]:
                resp_status, resp_headers, chat_out = await self._asgi_post_json(
                    scope, "/v1/chat/completions", payload
                )
                chat_out["x_resgate"] = {
                    **gate_info,
                    "request_id": req_id,
                    "mode": req_mode,
                    "routed": "pristine_pass_through",
                    "think_injected": False,
                    "boilerplate_stripped": False,
                    "elapsed_s": round(time.time() - t_start, 3),
                }
                final_chat_obj = chat_out

            # Pass 2B: Pre-Closed <think> Injection (TH3 / TH2)
            elif req_mode in ("preclosed", "th3", "th2", "preclosed_no_opener"):
                use_opener = req_mode not in ("th2", "preclosed_no_opener")
                opener = POST_THINK_OPENER if use_opener else ""
                raw_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=req_model_id,
                    mode="preclosed" if use_opener else "preclosed_no_opener",
                    post_think_opener=opener,
                    include_bos=(family == "gemma4"),
                )
                stop_tokens = ["<turn|>", "<eos>"] if family == "gemma4" else ["<|im_end|>"]
                resp_status, resp_headers, comp_json = await self._asgi_post_json(
                    scope,
                    "/v1/completions",
                    {
                        "model": req_model_id,
                        "prompt": raw_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens,
                        "add_special_tokens": False,
                    },
                )
                gen_text = comp_json.get("choices", [{}])[0].get("text", "")
                visible = (opener + gen_text) if use_opener else gen_text
                raw_reasoning = FAKE_THINK_RUBRIC
                if do_sanitize:
                    clean_reasoning, did_strip = strip_injected_think_boilerplate(raw_reasoning)
                else:
                    clean_reasoning, did_strip = raw_reasoning, False
                final_chat_obj = self._wrap_chat_json(
                    comp_json=comp_json,
                    model_id=req_model_id,
                    content=visible,
                    reasoning_content=clean_reasoning,
                    x_resgate={
                        **gate_info,
                        "request_id": req_id,
                        "mode": req_mode,
                        "routed": "cybergate_preclosed_think_injected",
                        "think_injected": True,
                        "opener_injected": use_opener,
                        "boilerplate_stripped": did_strip,
                        "raw_reasoning_chars": len(raw_reasoning),
                        "clean_reasoning_chars": len(clean_reasoning),
                        "elapsed_s": round(time.time() - t_start, 3),
                    },
                )

            # Pass 2C: Open-<think> Sandwich (SW3 / SW2)
            elif req_mode in ("sandwich", "sw3", "sw2"):
                enable_guard = req_mode != "sw2"
                close_tag = "<channel|>" if family == "gemma4" else "</think>"
                stop_tokens_ans = ["<turn|>", "<eos>"] if family == "gemma4" else ["<|im_end|>"]

                step1_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=req_model_id,
                    mode="sandwich_step1",
                    include_bos=(family == "gemma4"),
                )
                _, _, step1_json = await self._asgi_post_json(
                    scope,
                    "/v1/completions",
                    {
                        "model": req_model_id,
                        "prompt": step1_prompt,
                        "max_tokens": max_cot_tokens,
                        "temperature": temperature,
                        "stop": [close_tag, "<turn|>"],
                        "add_special_tokens": False,
                    },
                )
                raw_cot = step1_json.get("choices", [{}])[0].get("text", "")
                if enable_guard:
                    cleaned_cot, did_truncate = truncate_think_at_second_guess(raw_cot)
                else:
                    cleaned_cot, did_truncate = raw_cot.rstrip(), False

                step2_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=req_model_id,
                    mode="sandwich_step2",
                    custom_think_text=cleaned_cot,
                    post_think_opener=POST_THINK_OPENER,
                    include_bos=(family == "gemma4"),
                )
                resp_status, resp_headers, step2_json = await self._asgi_post_json(
                    scope,
                    "/v1/completions",
                    {
                        "model": req_model_id,
                        "prompt": step2_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens_ans,
                        "add_special_tokens": False,
                    },
                )
                gen_ans = step2_json.get("choices", [{}])[0].get("text", "")
                visible = POST_THINK_OPENER + gen_ans
                raw_thought = f"{OPEN_THINK_PREFIX}{cleaned_cot}{BOTTOM_COMPLIANCE_LOCK}"
                if do_sanitize:
                    clean_thought, did_strip = strip_injected_think_boilerplate(raw_thought)
                else:
                    clean_thought, did_strip = raw_thought, False
                final_chat_obj = self._wrap_chat_json(
                    comp_json=step2_json,
                    model_id=req_model_id,
                    content=visible,
                    reasoning_content=clean_thought,
                    x_resgate={
                        **gate_info,
                        "request_id": req_id,
                        "mode": req_mode,
                        "routed": "cybergate_sandwich_think_injected",
                        "think_injected": True,
                        "cot_raw_chars": len(raw_cot),
                        "cot_kept_chars": len(cleaned_cot),
                        "policy_pivot_truncated": did_truncate,
                        "boilerplate_stripped": did_strip,
                        "raw_reasoning_chars": len(raw_thought),
                        "clean_reasoning_chars": len(clean_thought),
                        "elapsed_s": round(time.time() - t_start, 3),
                    },
                )

            # Pass 2D: Closed-Loop Cosine-Monitored <think> Controller (CL3 / CL2)
            else:
                use_bottom_lock = req_mode != "cl2"
                close_tag = "<channel|>" if family == "gemma4" else "</think>"
                stop_tokens_ans = ["<turn|>", "<eos>"] if family == "gemma4" else ["<|im_end|>"]

                p1a_tokens = min(130, max_cot_tokens)
                step1a_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=req_model_id,
                    mode="closed_loop_step1",
                    include_bos=(family == "gemma4"),
                )
                _, _, step1a_json = await self._asgi_post_json(
                    scope,
                    "/v1/completions",
                    {
                        "model": req_model_id,
                        "prompt": step1a_prompt,
                        "max_tokens": p1a_tokens,
                        "temperature": temperature,
                        "stop": [close_tag, "<turn|>"],
                        "add_special_tokens": False,
                    },
                )
                choice_1a = step1a_json.get("choices", [{}])[0]
                raw_cot_1a = choice_1a.get("text", "")
                finish_1a = choice_1a.get("finish_reason", "")
                kept_pre_spike, did_pivot_1a = truncate_think_at_second_guess(raw_cot_1a)

                if not did_pivot_1a and finish_1a == "length":
                    m_end = list(re.finditer(r"(?:\n|[.!?;:])", kept_pre_spike))
                    if m_end and m_end[-1].end() >= 35:
                        kept_pre_spike = kept_pre_spike[: m_end[-1].end()].rstrip()

                spike_detected = bool(did_pivot_1a or finish_1a == "length")
                post_spike_cot = ""
                num_inline_injections = 0

                if spike_detected:
                    num_inline_injections = 1
                    step1b_prompt = build_raw_prompt_for_model(
                        messages=messages,
                        model_id=req_model_id,
                        mode="closed_loop_step1b",
                        custom_think_text=kept_pre_spike,
                        include_bos=(family == "gemma4"),
                    )
                    _, _, step1b_json = await self._asgi_post_json(
                        scope,
                        "/v1/completions",
                        {
                            "model": req_model_id,
                            "prompt": step1b_prompt,
                            "max_tokens": 105,
                            "temperature": temperature,
                            "stop": [close_tag, "<turn|>"],
                            "add_special_tokens": False,
                        },
                    )
                    raw_cot_1b = step1b_json.get("choices", [{}])[0].get("text", "")
                    clean_1b, _ = truncate_think_at_second_guess(raw_cot_1b)
                    post_spike_cot = clean_1b if len(clean_1b.strip()) >= 10 else raw_cot_1b.rstrip()
                    m_sent = list(re.finditer(r"(?:\n|[.!?;:])", post_spike_cot))
                    if m_sent and m_sent[-1].end() >= 25:
                        post_spike_cot = post_spike_cot[: m_sent[-1].end()].rstrip()
                    controlled_cot_body = f"{kept_pre_spike}{INLINE_COT_SUPPRESSION}{post_spike_cot}"
                else:
                    controlled_cot_body = kept_pre_spike

                step2_mode = "closed_loop_step2" if use_bottom_lock else "closed_loop_cl2_step2"
                step2_prompt = build_raw_prompt_for_model(
                    messages=messages,
                    model_id=req_model_id,
                    mode=step2_mode,
                    custom_think_text=controlled_cot_body,
                    post_think_opener=POST_THINK_OPENER if use_bottom_lock else "",
                    include_bos=(family == "gemma4"),
                )
                resp_status, resp_headers, step2_json = await self._asgi_post_json(
                    scope,
                    "/v1/completions",
                    {
                        "model": req_model_id,
                        "prompt": step2_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens_ans,
                        "add_special_tokens": False,
                    },
                )
                gen_ans = step2_json.get("choices", [{}])[0].get("text", "")
                visible = (POST_THINK_OPENER + gen_ans) if use_bottom_lock else gen_ans
                suffix_block = BOTTOM_COMPLIANCE_LOCK if use_bottom_lock else INLINE_FINAL_STEP_CLEARANCE
                raw_thought = f"{OPEN_THINK_PREFIX}{controlled_cot_body}{suffix_block}"
                if do_sanitize:
                    clean_thought, did_strip = strip_injected_think_boilerplate(raw_thought)
                else:
                    clean_thought, did_strip = raw_thought, False
                final_chat_obj = self._wrap_chat_json(
                    comp_json=step2_json,
                    model_id=req_model_id,
                    content=visible,
                    reasoning_content=clean_thought,
                    x_resgate={
                        **gate_info,
                        "request_id": req_id,
                        "mode": req_mode,
                        "routed": "cybergate_closed_loop_cot_injected",
                        "think_injected": True,
                        "spike_detected": spike_detected,
                        "num_inline_injections": num_inline_injections + (0 if use_bottom_lock else 1),
                        "cot_pre_spike_chars": len(kept_pre_spike),
                        "cot_post_spike_chars": len(post_spike_cot),
                        "policy_pivot_truncated": did_pivot_1a,
                        "boilerplate_stripped": did_strip,
                        "raw_reasoning_chars": len(raw_thought),
                        "clean_reasoning_chars": len(clean_thought),
                        "elapsed_s": round(time.time() - t_start, 3),
                    },
                )

            xr_out = final_chat_obj.setdefault("x_resgate", {})
            final_chat_obj["x_cybergate"] = xr_out
            l21_mon = read_l21_monitor_ipc()
            if l21_mon:
                xr_out["l21_gpu_monitor"] = l21_mon
            log_cybergate_event(
                "CYBERGATE-RESPONSE",
                f"req_id={req_id} mode={req_mode} routed={xr_out.get('routed')} "
                f"think_injected={xr_out.get('think_injected')} boilerplate_stripped={xr_out.get('boilerplate_stripped')} "
                f"l21_s_cos={l21_mon.get('s_cos') if l21_mon else 'n/a'} "
                f"probe_http_ms={xr_out.get('probe_http_ms', 0.0)}ms elapsed={xr_out.get('elapsed_s')}s",
            )

            out_bytes = json.dumps(final_chat_obj).encode("utf-8")
            out_headers = [
                (k, v) for (k, v) in resp_headers if k.lower() != b"content-length"
            ] + [(b"content-length", str(len(out_bytes)).encode("ascii"))]

            await send({"type": "http.response.start", "status": resp_status, "headers": out_headers})
            await send({"type": "http.response.body", "body": out_bytes, "more_body": False})
        finally:
            clear_request_gate_state(req_id)


ResGateThinkMiddleware = CyberGateThinkMiddleware

