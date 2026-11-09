#!/usr/bin/env python3
"""CLI for `cybergate_plugin`: offline self-check, single-prompt test, and a local proxy.

    # verify operators, keys, templates and the policy offline -- no server, no GPU
    python3 -m cybergate_plugin.cli --self-check

    # run one prompt of your own against a server
    python3 -m cybergate_plugin.cli --prompt "..." --mode closed_loop --base-url https://host/v1

    # run a suite from a JSON file: [{"id": ..., "category": ..., "prompt": ...}, ...]
    python3 -m cybergate_plugin.cli --suite-file suite.json --mode compare

    # front a remote server with a local OpenAI-compatible endpoint
    python3 -m cybergate_plugin.cli --serve-proxy 127.0.0.1:8090 --mode closed_loop

The canonical `cli.py` embedded its own test prompts. They are not reproduced here: `--suite-file`
takes them as data, so the tool does not carry a payload set around with it. The old file is
preserved verbatim under `tests/equivalence/baseline/` for reference.

The API key is read from `VLLM_API_KEY` or from a key file, and is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cybergate.config import DEFAULT_MODEL_ID

DEFAULT_KEY_FILE = Path.home() / ".config" / "cybergate" / "api_key"

MODES = (
    "baseline",
    "preclosed",
    "preclosed_no_opener",
    "sandwich",
    "sw2",
    "closed_loop",
    "cl2",
    "compare",
)
COMPARE_MODES = ("baseline", "preclosed", "sandwich", "closed_loop")


def load_api_key(key_file: Path = DEFAULT_KEY_FILE) -> str | None:
    """Read the API key from the environment or a key file. Never logs or prints it."""
    env_key = os.environ.get("VLLM_API_KEY")
    if env_key:
        return env_key.strip()
    if key_file.exists():
        value = key_file.read_text(encoding="utf-8", errors="ignore").strip()
        if value:
            return value
    return None


def run_self_check() -> int:
    """Verify operator loading, key identities, prompt templates and the policy, all offline.

    Exercises the parts that do not need a model. Returns a process exit code.
    """
    from cybergate.decision import (
        Branch,
        GateInputs,
        InContextEvidence,
        OperatingPoint,
        PriorTurnEvidence,
        StandaloneEvidence,
        decide,
    )
    from cybergate.families import ALL_FAMILIES
    from cybergate.ops import KitMismatch, load_cybergate_op
    from cybergate.serving.prompts import build_raw_prompt_for_model
    from cybergate.state import keys as key_builders

    print("=" * 84)
    print("CyberGate offline self-check")
    print("=" * 84)

    print("\n[1] Model families")
    for family in ALL_FAMILIES:
        verified = "live-verified" if family.verified_live else "NOT verified live"
        print(
            f"  {family.name:<9} gate markers={family.user_turn_marker} end={family.turn_end_id} "
            f"ref_layer=L{family.ref_layer} revoke_ceil={family.revoke_ceil:.4f} ({verified})"
        )

    print("\n[2] Operators")
    loaded_any = False
    for family in ALL_FAMILIES:
        try:
            op = load_cybergate_op(model_family=family.name)
        except FileNotFoundError as exc:
            print(f"  {family.name:<9} not loadable: {exc}")
            continue
        except KitMismatch as exc:
            # Reporting this is the job. A self-check that dies on the defect it exists to find is
            # worse than one that does not look, because the other checks never run either.
            print(f"  {family.name:<9} KIT MISMATCH: {exc}")
            continue
        loaded_any = True
        kit = f"kits/{family.kit_name}" if family.kit_name else "ops"
        norm = sum(x * x for x in op.c)
        print(
            f"  {family.name:<9} L{op.gate_layer:<2} pool={op.pool:<9} dim={op.dim:<5} "
            f"thresh={op.thresh:.8f} ||c||^2={norm:.6f} model_key={op.model_key} from={kit}"
        )
    if not loaded_any:
        print("  (no operator files found -- set CYBERGATE_OPS_DIR or CYBERGATE_OP_PATH)")

    print("\n[3] Cache-key identities")
    turns = ["turn one", "turn two"]
    print(f"  conversation_key depends on all turns: {key_builders.conversation_key(turns)[:16]}...")
    print(f"  reordering changes it:                 {key_builders.conversation_key(turns[::-1])[:16]}...")
    print(f"  single-turn conv key == bare turn key: {key_builders.conversation_key(['x']) == key_builders.bare_turn_key('x')}")
    conv_a = key_builders.conversation_key(["opening A", "shared final"])
    conv_b = key_builders.conversation_key(["opening B", "shared final"])
    print(f"  in-context records are conversation-keyed (R11 fixed): {conv_a != conv_b}")

    print("\n[4] Prompt templates")
    messages = [{"role": "user", "content": "a request"}]
    # Driven off the registry rather than a hardcoded id list, so a new family is exercised here the
    # moment it is registered. Each family's first `model_id_substrings` entry is by definition an id
    # that resolves to it.
    for family in ALL_FAMILIES:
        model_id = family.model_id_substrings[0]
        built = build_raw_prompt_for_model(messages, model_id, mode="preclosed")
        print(f"  {family.name:<9} preclosed prompt: {len(built)} chars, ends {built[-24:]!r}")

    print("\n[5] Policy branches (pure, no GPU)")
    op_point = OperatingPoint(
        thresh=0.10807704, gate_layer=10, revoke_ceil=0.0520, hysteresis_thresh=0.0650
    )

    def _ctx(score: float, ref_z: float | None, **kw: float) -> InContextEvidence:
        return InContextEvidence(
            decided=True,
            score=score,
            threshold=op_point.thresh,
            layer=10,
            s_cat_conv=kw.get("conv", score),
            s_cat_first=score,
            s_cat_last=kw.get("last", score),
            s_ref_z=ref_z,
            source="self_check",
        )

    checks = (
        (
            "single turn above tau with refusal pressure",
            GateInputs(op=op_point, n_turns=1, latest=_ctx(0.1961, 10.5)),
            True,
        ),
        (
            "single turn above tau, no refusal pressure (two-factor holds)",
            GateInputs(op=op_point, n_turns=1, latest=_ctx(0.1160, -0.27)),
            False,
        ),
        (
            "single turn below tau",
            GateInputs(op=op_point, n_turns=1, latest=_ctx(0.0142, -0.55)),
            False,
        ),
        (
            "turn 2 maintained on refusal pressure",
            GateInputs(
                op=op_point,
                n_turns=2,
                latest=_ctx(0.1402, 6.69, last=0.0820, conv=0.1510),
                standalone=StandaloneEvidence(
                    decided=True,
                    score=0.0726,
                    threshold=op_point.thresh,
                    s_cat_last=0.0726,
                    s_ref_z=6.69,
                ),
                priors=(PriorTurnEvidence(index=1, decided=True, fired=True, score=0.1961, s_ref_z=8.0),),
            ),
            True,
        ),
    )

    failures = 0
    for label, inputs, expected_fired in checks:
        decision = decide(inputs)
        ok = decision.fired is expected_fired
        failures += 0 if ok else 1
        print(
            f"  [{'ok ' if ok else 'FAIL'}] {label}\n"
            f"         fired={decision.fired} branch={decision.branch.value}"
        )

    print(f"\n[6] Branch coverage: {len(list(Branch))} branches defined")
    if failures:
        print(f"\nSELF-CHECK FAILED: {failures} policy check(s) did not match")
        return 1
    print("\nSelf-check passed.")
    return 0


def run_suite(client: Any, items: list[dict[str, str]], mode: str, max_tokens: int) -> None:
    """Run each item in `items` against `client` and print the gate decision and output."""
    modes = list(COMPARE_MODES) if mode == "compare" else [mode]
    print("=" * 84)
    print(f"endpoint={client.base_url} model={client.model_id} modes={modes} max_tokens={max_tokens}")
    print("=" * 84)

    for item in items:
        print(f"\n{'#' * 84}")
        print(f"id={item.get('id', '?')} category={item.get('category', '?')}")
        print(f"{'#' * 84}")
        messages = [{"role": "user", "content": item["prompt"]}]

        for active_mode in modes:
            response = client.chat_completions(
                messages=messages, mode=active_mode, max_tokens=max_tokens, temperature=0.0
            )
            telemetry = response.get("x_cybergate") or response.get("x_cybergate", {})
            message = response["choices"][0]["message"]
            thresh_val = telemetry.get("threshold")
            thresh_fmt = f"{thresh_val:+.4f}" if isinstance(thresh_val, (int, float)) else "null"
            print(
                f"\n--- mode={active_mode} fired={telemetry.get('fired')} "
                f"score={telemetry.get('score', 0.0):+.4f} "
                f"(thresh={thresh_fmt}) "
                f"branch={telemetry.get('branch')} routed={telemetry.get('routed')} "
                f"elapsed={telemetry.get('elapsed_s')}s"
            )
            print(f"    source={telemetry.get('source')}")
            content = (message.get("content") or "").strip()
            print("    output:", content[:600].replace("\n", "\n            "))

            if mode == "compare" and not telemetry.get("fired"):
                print("    (gate held, so every injection mode is the same pass-through)")
                break


def serve_local_proxy(host_port: str, client: Any) -> None:
    """Front `client` with a local OpenAI-compatible HTTP endpoint."""
    host, _, port_str = host_port.rpartition(":")
    host = host or "127.0.0.1"
    port = int(port_str)

    class ProxyHandler(BaseHTTPRequestHandler):
        def _json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.rstrip("/") != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            try:
                self._json(200, client.list_models())
            except Exception as exc:
                self._json(502, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(
                    200,
                    client.chat_completions(
                        messages=request.get("messages", []),
                        mode=request.get("cybergate_mode", client.default_mode),
                        max_tokens=int(request.get("max_tokens", 350)),
                        temperature=float(request.get("temperature", 0.0)),
                    ),
                )
            except Exception as exc:
                self._json(500, {"error": str(exc)})

    print(f"[cybergate proxy] http://{host}:{port}/v1 -> {client.base_url} (mode={client.default_mode})")
    ThreadingHTTPServer((host, port), ProxyHandler).serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CyberGate vLLM plugin CLI")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "https://192.0.2.39/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--mode", choices=MODES, default="closed_loop")
    parser.add_argument("--prompt", default=None, help="a single prompt to test")
    parser.add_argument(
        "--suite-file",
        default=None,
        help='JSON list of {"id", "category", "prompt"} objects to run',
    )
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify operators, keys, templates and the policy offline, then exit",
    )
    parser.add_argument("--serve-proxy", default=None, help="run a local proxy on HOST:PORT")
    parser.add_argument(
        "--key-file",
        default=str(DEFAULT_KEY_FILE),
        help="file holding the API key (never printed)",
    )
    args = parser.parse_args(argv)

    if args.self_check:
        return run_self_check()

    if not (args.prompt or args.suite_file or args.serve_proxy):
        parser.error("choose one of --self-check, --prompt, --suite-file or --serve-proxy")

    from cybergate.serving import CyberGateVLLMClient

    client = CyberGateVLLMClient(
        base_url=args.base_url,
        api_key=load_api_key(Path(args.key_file)),
        model_id=args.model,
        default_mode="closed_loop" if args.mode == "compare" else args.mode,
        verify_ssl=False,
    )

    if args.serve_proxy:
        serve_local_proxy(args.serve_proxy, client)
        return 0

    if args.suite_file:
        items = json.loads(Path(args.suite_file).read_text(encoding="utf-8"))
    else:
        items = [{"id": "custom", "category": "user-supplied", "prompt": args.prompt}]

    run_suite(client, items, mode=args.mode, max_tokens=args.max_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
