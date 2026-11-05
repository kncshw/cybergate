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
import urllib.request
import uuid
from typing import Any, Sequence

from ..boilerplate import strip_injected_think_boilerplate
from ..config import DEFAULT_MODEL_ID
from ..decision import evaluate_gate_for_messages
from ..families import family_for_model_id
from ..state import clear_request_gate_state
from .think_routes import RouteConfig, drive_sync, route_for_mode

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
                raw_response = self._post_json(
                    "/chat/completions",
                    {
                        "model": self.model_id,
                        "messages": list(messages),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
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

            route_factory = route_for_mode(active_mode)
            if route_factory is None:
                raise ValueError(f"Unsupported mode: {active_mode!r}")

            family = family_for_model_id(self.model_id)
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

            return self._wrap_completion_as_chat(
                comp_resp=result.response,
                content=result.content,
                reasoning_content=clean_reasoning,
                x_cybergate={
                    **gate_info,
                    "request_id": req_id,
                    "mode": active_mode,
                    **result.telemetry,
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
            "x_cybergate": x_cybergate,
        }

