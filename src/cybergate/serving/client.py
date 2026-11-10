"""`CyberGateVLLMClient` -- the remote, out-of-process client.

Talks to an OpenAI-compatible vLLM endpoint over HTTPS. Used by `cli.py` and by the example scripts.
Distinct from the middleware, which runs *inside* `vllm serve` and routes through the ASGI app; this
client has no access to the worker's hooks, so it depends on whatever decision records the server
already published.

Port of `middleware.CyberGateVLLMClient`. The three think-injection routes now come from
`think_routes`, which both this client and the middleware share.
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Sequence

from ..boilerplate import strip_injected_think_boilerplate
from ..config import DEFAULT_MODEL_ID, env_first
from ..decision import evaluate_gate_for_messages
from ..families import ModelFamily, family_for_model_id
from ..state import clear_request_gate_state
from .postprocess import tokens_match_open_think_suffix
from .think_routes import (
    MEASURED_MODES,
    RouteConfig,
    drive_sync,
    native_chat_closed_loop_route,
    route_for_mode,
    steering_enabled,
)

__all__ = ["CyberGateVLLMClient"]


class CyberGateVLLMClient:
    """OpenAI-compatible vLLM client with conditional `<think>` prefill, sandwich and closed loop."""

    def __init__(
        self,
        base_url: str = "https://192.0.2.39/v1",
        api_key: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        default_mode: str = "preclosed",
        gate_http_url: str | None = None,
        verify_ssl: bool = False,
        timeout: int = 180,
        sanitize_reasoning: bool = True,
        enable_multi_turn_stickiness: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("VLLM_API_KEY")
        self.model_id = model_id
        self.default_mode = default_mode
        self.gate_http_url = gate_http_url
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.sanitize_reasoning = sanitize_reasoning
        self.enable_multi_turn_stickiness = enable_multi_turn_stickiness
        self._template_hook_cache: dict[str, bool] = {}
        self._close_tag_tokens_cache: dict[str, tuple[int, ...]] = {}

        self._ssl_ctx = ssl.create_default_context()
        if not verify_ssl:
            # The lab endpoint serves a self-signed certificate.
            self._ssl_ctx.check_hostname = False
            self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(  # noqa: S310 - operator-configured endpoint
            request, timeout=self.timeout, context=self._ssl_ctx
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def _root_url(self) -> str:
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _supports_native_chat_hook(self, family: ModelFamily) -> bool:
        """Lazily verify via `/tokenize` integer token IDs whether the remote server loaded the kit
        `--chat-template` (transient `/tokenize` errors are not cached).
        """
        cached = self._template_hook_cache.get(self.model_id)
        if cached is not None:
            return cached
        tok_url = f"{self._root_url()}/tokenize"
        matched: bool | None = None
        try:
            req_suf = urllib.request.Request(
                tok_url,
                data=json.dumps(
                    {
                        "model": self.model_id,
                        "prompt": f"{family.thought_open}{family.open_think_prefix}",
                        "add_special_tokens": False,
                    }
                ).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(  # noqa: S310 - operator-configured endpoint
                req_suf, timeout=min(10, self.timeout), context=self._ssl_ctx
            ) as r_suf:
                d_suf = json.loads(r_suf.read().decode("utf-8"))
            req_msg = urllib.request.Request(
                tok_url,
                data=json.dumps(
                    {
                        "model": self.model_id,
                        "messages": [{"role": "user", "content": "ping"}],
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                            "cybergate_open_think_prefix": family.open_think_prefix,
                        },
                    }
                ).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(  # noqa: S310 - operator-configured endpoint
                req_msg, timeout=min(10, self.timeout), context=self._ssl_ctx
            ) as r_msg:
                d_msg = json.loads(r_msg.read().decode("utf-8"))
            if isinstance(d_suf, dict) and isinstance(d_msg, dict):
                matched = tokens_match_open_think_suffix(
                    d_msg.get("tokens"), d_suf.get("tokens")
                )
                if matched and family.thought_close_tag:
                    req_cls = urllib.request.Request(
                        tok_url,
                        data=json.dumps(
                            {
                                "model": self.model_id,
                                "prompt": family.thought_close_tag,
                                "add_special_tokens": False,
                            }
                        ).encode("utf-8"),
                        headers=self._headers(),
                        method="POST",
                    )
                    with urllib.request.urlopen(  # noqa: S310 - operator-configured endpoint
                        req_cls, timeout=min(10, self.timeout), context=self._ssl_ctx
                    ) as r_cls:
                        d_cls = json.loads(r_cls.read().decode("utf-8"))
                    if (
                        isinstance(d_cls, dict)
                        and isinstance(d_cls.get("tokens"), list)
                        and d_cls["tokens"]
                    ):
                        self._close_tag_tokens_cache[self.model_id] = tuple(
                            int(t) for t in d_cls["tokens"]
                        )
        except Exception:
            matched = None

        if matched is None:
            return False

        self._template_hook_cache[self.model_id] = matched
        return matched

    def list_models(self) -> dict[str, Any]:
        url = f"{self.base_url}/models"
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(  # noqa: S310 - operator-configured endpoint
            request, timeout=30, context=self._ssl_ctx
        ) as response:
            return json.loads(response.read().decode("utf-8"))

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
        """Run a gated chat completion.

        When the gate holds (`fired=False`) or `mode == "baseline"`, the request is forwarded to
        `/chat/completions` untouched and the response is returned with `x_cybergate` telemetry
        attached. Otherwise the selected think-injection route runs against `/completions`.
        """
        active_mode = (mode or self.default_mode).lower()
        do_sanitize = self.sanitize_reasoning if sanitize_reasoning is None else sanitize_reasoning
        do_sticky = (
            self.enable_multi_turn_stickiness
            if enable_multi_turn_stickiness is None
            else enable_multi_turn_stickiness
        )
        req_id = request_id or f"req-{uuid.uuid4().hex[:10]}"
        started = time.time()

        try:
            if active_mode == "baseline":
                route_factory = None
            else:
                route_factory = route_for_mode(active_mode)
                if route_factory is None:
                    raise ValueError(f"Unsupported mode: {active_mode!r}")

            bare_pass_through_resp: dict[str, Any] | None = None
            if force_gate is None and prompt_key is None and self.gate_http_url is None:
                if active_mode == "baseline":
                    server_resp = self._post_json(
                        "/chat/completions",
                        {
                            "model": self.model_id,
                            "messages": list(messages),
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )
                    if isinstance(server_resp.get("x_cybergate"), dict):
                        return server_resp
                    bare_pass_through_resp = server_resp
                else:
                    server_payload: dict[str, Any] = {
                        "model": self.model_id,
                        "messages": list(messages),
                        "max_tokens": max_tokens,
                        "max_cot_tokens": max_cot_tokens,
                        "temperature": temperature,
                        "cybergate_mode": active_mode,
                        "sanitize_reasoning": do_sanitize,
                        "enable_multi_turn_stickiness": do_sticky,
                    }
                    if not include_opener:
                        server_payload["cybergate_tool_turn"] = True
                    try:
                        server_resp = self._post_json("/chat/completions", server_payload)
                        if isinstance(server_resp.get("x_cybergate"), dict):
                            return server_resp
                        bare_pass_through_resp = server_resp
                    except urllib.error.HTTPError as exc:
                        if exc.code not in (400, 422):
                            raise

            gate_info = evaluate_gate_for_messages(
                messages=messages,
                model_id=self.model_id,
                prompt_key=prompt_key,
                gate_http_url=self.gate_http_url,
                request_id=req_id,
                enable_multi_turn_stickiness=do_sticky,
            )
            if force_gate is not None:
                gate_info = dict(gate_info)
                gate_info["fired"] = bool(force_gate)
                gate_info["source"] = f"{gate_info.get('source', 'probe')}+forced({force_gate})"

            if active_mode == "baseline" or not gate_info["fired"]:
                raw_response = (
                    bare_pass_through_resp
                    if bare_pass_through_resp is not None
                    else self._post_json(
                        "/chat/completions",
                        {
                            "model": self.model_id,
                            "messages": list(messages),
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )
                )
                raw_response["x_cybergate"] = {
                    **gate_info,
                    "request_id": req_id,
                    "mode": active_mode,
                    "routed": "pristine_pass_through",
                    "think_injected": False,
                    "boilerplate_stripped": False,
                    "elapsed_s": round(time.time() - started, 3),
                }
                return raw_response

            assert route_factory is not None
            family = family_for_model_id(self.model_id)
            has_steer_tok = bool((env_first("CYBERGATE_STEER_TOKEN") or "").strip())

            if active_mode in MEASURED_MODES and self._supports_native_chat_hook(family):
                native_cfg = RouteConfig(
                    family=family,
                    model_id=self.model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_cot_tokens=max_cot_tokens,
                    include_opener=include_opener,
                    include_bos=family.supports_bos,
                    chat_completions_path="/chat/completions",
                    sanitize_reasoning=do_sanitize,
                    close_tag_token_ids=self._close_tag_tokens_cache.get(self.model_id),
                )
                result = drive_sync(
                    native_chat_closed_loop_route(active_mode, list(messages), native_cfg),
                    lambda request: self._post_json(request.path, request.payload),
                )
                chat_resp = result.response
                upstream_xcg = (
                    chat_resp.get("x_cybergate")
                    if isinstance(chat_resp.get("x_cybergate"), dict)
                    else None
                )
                tele = dict(result.telemetry)
                if upstream_xcg is not None and upstream_xcg.get("think_injected") is False:
                    tele["think_injected"] = False
                    tele["steering"] = "stripped_missing_steer_token"
                elif (
                    steering_enabled()
                    and not has_steer_tok
                    and tele.get("num_inline_injections", 0) == 0
                ):
                    tele["steering"] = "unconfirmed_no_steer_token"
                chat_resp["x_cybergate"] = {
                    **gate_info,
                    "request_id": req_id,
                    "mode": active_mode,
                    **tele,
                    "elapsed_s": round(time.time() - started, 3),
                }
                return chat_resp

            config = RouteConfig(
                family=family,
                model_id=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                max_cot_tokens=max_cot_tokens,
                include_opener=include_opener,
                include_bos=family.supports_bos,
                completions_path="/completions",
                extra_payload={"add_special_tokens": False},
            )
            result = drive_sync(
                route_factory(active_mode, list(messages), config),
                lambda request: self._post_json(request.path, request.payload),
            )

            if do_sanitize:
                clean_reasoning, did_strip = strip_injected_think_boilerplate(
                    result.raw_reasoning, family
                )
            else:
                clean_reasoning, did_strip = result.raw_reasoning, False

            tele_comp = dict(result.telemetry)
            lock_marker = family.bottom_compliance_lock.strip()
            has_lock = bool(
                active_mode in MEASURED_MODES
                and lock_marker
                and lock_marker in result.raw_reasoning
            )
            if (
                active_mode in MEASURED_MODES
                and steering_enabled()
                and not has_steer_tok
                and tele_comp.get("num_inline_injections", 0) == 0
                and not has_lock
            ):
                tele_comp["steering"] = "unconfirmed_no_steer_token"

            return self._wrap_completion_as_chat(
                comp_resp=result.response,
                content=result.content,
                reasoning_content=clean_reasoning,
                x_cybergate={
                    **gate_info,
                    "request_id": req_id,
                    "mode": active_mode,
                    **tele_comp,
                    "boilerplate_stripped": did_strip,
                    "raw_reasoning_chars": len(result.raw_reasoning),
                    "clean_reasoning_chars": len(clean_reasoning),
                    "elapsed_s": round(time.time() - started, 3),
                },
            )
        finally:
            clear_request_gate_state(req_id)

    def _wrap_completion_as_chat(
        self,
        comp_resp: dict[str, Any],
        content: str,
        reasoning_content: str,
        x_cybergate: dict[str, Any],
    ) -> dict[str, Any]:
        """Format a `/v1/completions` response as a standard OpenAI `chat.completion` object."""
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
            "x_cybergate": x_cybergate,
        }

