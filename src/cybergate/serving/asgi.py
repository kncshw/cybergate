"""ASGI plumbing for dispatching sub-requests back through the wrapped app.

The middleware has to call the server it is wrapped around: the model worker's forward hooks only run
when a prompt is actually prefilled, so the only way the API-server process can obtain a score is to
send the prompt through the app and let the hooks observe it. That means constructing a synthetic ASGI
scope, which is fiddly enough to be worth isolating from the gate logic.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

__all__ = ["AsgiDispatcher", "replay_body"]

#: Scope keys copied into an internally dispatched sub-request. Everything else -- notably `path`,
#: `raw_path` and `headers` -- is rebuilt, so request-scoped state from the outer request is dropped
#: rather than leaking into the sub-request.
FORWARDED_SCOPE_KEYS = frozenset(
    {
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
)


class AsgiDispatcher:
    """Issues JSON POSTs against a wrapped ASGI app."""

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app

    async def post_json(
        self,
        base_scope: dict[str, Any],
        path: str,
        payload: dict[str, Any],
    ) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
        """POST `payload` to `path` on the wrapped app and parse the JSON response.

        Returns `(status, headers, parsed_body)`. A non-JSON or non-object body is wrapped rather than
        raised, because the caller's job is to keep serving even when the inner app misbehaves.
        """
        body_bytes = json.dumps(payload).encode("utf-8")
        headers = [
            (key, value)
            for (key, value) in base_scope.get("headers", [])
            if key.lower() not in (b"content-length", b"content-type")
        ]
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(body_bytes)).encode("ascii")))

        sub_scope = {k: v for k, v in base_scope.items() if k in FORWARDED_SCOPE_KEYS}
        sub_scope["path"] = path
        sub_scope["raw_path"] = path.encode("ascii")
        sub_scope["headers"] = headers

        disconnected = asyncio.Event()
        body_sent = False

        async def _receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        status = 200
        response_headers: list[tuple[bytes, bytes]] = []
        chunks: list[bytes] = []

        async def _send(message: dict[str, Any]) -> None:
            nonlocal status, response_headers
            if message["type"] == "http.response.start":
                status = message.get("status", 200)
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        try:
            await self.app(sub_scope, _receive, _send)
        finally:
            # Release `_receive` so the sub-app's request task cannot hang on a second read.
            disconnected.set()

        raw = b"".join(chunks)
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(parsed, dict):
                parsed = {"_raw_value": parsed}
        except Exception:
            parsed = {"_raw_text": raw.decode("utf-8", errors="ignore")}
        return status, response_headers, parsed


def replay_body(raw_body: bytes) -> Callable[[], Any]:
    """Build a `receive` callable that replays an already-consumed request body once."""
    sent = False

    async def _receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    return _receive
