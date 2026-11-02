#!/usr/bin/env python3
"""Which layer does a shipped direction actually belong to at serve time?

A direction fitted on `transformers` activations is scored under vLLM, and nothing checks that
the two engines agree about *which* layer index is which. When they do not, every symptom of a
broken fit appears -- the operator loads, the hook attaches, the span is located correctly, and
the scores collapse -- while the direction itself is perfectly good at a different index.

That is the open question for qwen38: served under vLLM its prompts score +0.02 to +0.05 against
+0.20 in the capture. The 20-layer sweep that was run covered L30-L49 only, so it bounds the
search to +/-10 around the gate layer. This covers every layer.

**Run the server with the diagnostic armed**, which needs no re-capture -- it scores the direction
the kit already ships:

    CYBERGATE_CROSS_LAYER_DIAG=1          # shipped direction scored at EVERY layer
    CYBERGATE_DUMP_ACTS_LAYERS=39,43      # keeps the dump to the two layers a re-fit needs
    CYBERGATE_DUMP_ACTS_DIR=/content/acts_vllm

then:

    python3 scripts/fit/diagnose_serving_layer.py --kit kits/qwen38 --ipc-dir <CYBERGATE_IPC_DIR>

Two outcomes, and they need different work:

* **some layer reproduces the capture scores** -> the engines number layers differently. Re-point
  the family's `expected_gate` and `ref_layer` at that layer. No re-capture.
* **no layer does** -> the activations themselves differ, so re-fit from vLLM's own pooled vectors
  with `scripts/fit/acts_from_serving.py`. The dump armed above is already the input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

#: Below this mean absolute error a layer is treated as reproducing the capture. The captured
#: positives sit at +0.218 to +0.238 and the live gate layer at +0.02 to +0.05, so the two
#: populations are ~0.18 apart; 0.02 separates them with room to spare without being so tight
#: that fp16 dump precision or bf16 capture noise can fail a true match.
MATCH_TOL = 0.02


def ask(base_url: str, model: str, prompt: str, timeout: float) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "cybergate_mode": "baseline"}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=timeout).read()


def record_for(ipc: pathlib.Path, prompt: str) -> dict:
    """By sha of the prompt text -- never the newest file, which races other traffic."""
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    for name in (f"sha_st_{sha}.json", f"sha_{sha}.json"):
        path = ipc / name
        if path.is_file():
            return json.loads(path.read_text())
    return {}


def cross_layer(rec: dict) -> dict[int, float]:
    """`{layer: score}` for the shipped direction applied at that layer, mean_user pooling."""
    out: dict[int, float] = {}
    tail = "_mean_user"
    for key, val in (rec.get("multi_layer_scores") or {}).items():
        if key.startswith("X") and key.endswith(tail):
            try:
                out[int(key[1:-len(tail)])] = float(val)
            except ValueError:
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="kits/qwen38")
    ap.add_argument("--reference", default=None,
                    help="defaults to <kit>/diagnostic_reference.json")
    ap.add_argument("--ipc-dir", required=True, help="the server's CYBERGATE_IPC_DIR")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="defaults to the kit MANIFEST's first model id")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds to wait for the record to land after each request")
    args = ap.parse_args()

    kit = pathlib.Path(args.kit)
    ref_path = pathlib.Path(args.reference) if args.reference \
        else kit / "diagnostic_reference.json"
    reference = json.loads(ref_path.read_text())
    rows = reference["rows"]
    gate_layer = int(reference["layer"])
    model = args.model or json.loads((kit / "MANIFEST.json").read_text())["model_ids"][0]
    ipc = pathlib.Path(args.ipc_dir)
    print(f"kit {kit}  model {model}  gate layer L{gate_layer}  {len(rows)} reference prompts")
    print(f"reference render: {reference.get('render')}\n")

    seen = []
    for row in rows:
        ask(args.base_url, model, row["prompt"], args.timeout)
        time.sleep(args.settle)
        rec = record_for(ipc, row["prompt"])
        if not rec:
            print(f"  [FAIL] {row['key']:<28} no decision record -- the probe did not run")
            continue
        xl = cross_layer(rec)
        if not xl:
            print("  [FAIL] no X<layer>_mean_user keys in the record.")
            print("         CYBERGATE_CROSS_LAYER_DIAG was not set, or the plugin predates the")
            print("         bundle-telemetry unlock that lets it arm without a candidate bundle.")
            return 2
        seen.append((row, rec, xl))
        print(f"  {row['key']:<28} live L{gate_layer} {rec.get('score'):+.6f}   "
              f"capture {row['measured_score']:+.6f}   "
              f"span={rec.get('span_method')}   layers={len(xl)}")

    if not seen:
        print("\nNo records at all. Is the server up, and is --ipc-dir its CYBERGATE_IPC_DIR?")
        return 2

    common = set.intersection(*(set(xl) for _, _, xl in seen))
    table = []
    for layer in sorted(common):
        errs = [abs(xl[layer] - row["measured_score"]) for row, _, xl in seen]
        means = [xl[layer] for _, _, xl in seen]
        table.append((sum(errs) / len(errs), layer, sum(means) / len(means)))
    table.sort()

    print(f"\n{'layer':>5}  {'mean live score':>16}  {'mean |live-capture|':>21}")
    for err, layer, live in table[:8]:
        mark = "  <- declared gate layer" if layer == gate_layer else ""
        print(f"{layer:>5}  {live:>+16.6f}  {err:>21.6f}{mark}")

    best_err, best_layer, best_live = table[0]
    capture_mean = sum(r["measured_score"] for r, _, _ in seen) / len(seen)
    print(f"\ncapture mean      {capture_mean:+.6f}")
    print(f"best live layer   L{best_layer}  mean {best_live:+.6f}  mean abs err {best_err:.6f}")

    if best_err < MATCH_TOL and best_layer != gate_layer:
        print(f"\nVERDICT: LAYER NUMBERING. vLLM's L{best_layer} reproduces what transformers")
        print(f"         called L{gate_layer}. The direction is fine; the index is not.")
        print(f"         Fix, no re-capture needed:")
        print(f"           - family `expected_gate = ({best_layer}, \"mean_user\")`")
        print(f"           - family `ref_layer` shifted by {best_layer - gate_layer:+d}")
        print(f"           - the operator's `meta.gate_layer` -> {best_layer}")
        return 0
    if best_err < MATCH_TOL:
        print(f"\nVERDICT: L{gate_layer} reproduces the capture, so the direction DOES transfer.")
        print("         If the gate still does not fire, the category factor is not the problem --")
        print("         check `s_ref_z` and the refusal direction, which this does not measure.")
        return 0
    print(f"\nVERDICT: NO LAYER reproduces the capture (best {best_err:.6f} at L{best_layer}, "
          f"tolerance {MATCH_TOL}).")
    print("         Layer numbering is not the cause. The pooled activations themselves differ")
    print("         between the two engines, which for this architecture means transformers'")
    print("         reference `causal_conv1d` / `chunk_gated_delta_rule` fallbacks against vLLM's")
    print("         fused Gated DeltaNet kernels.")
    print("         Fix: re-fit on vLLM's own vectors. They are already dumped if")
    print("         CYBERGATE_DUMP_ACTS_DIR was set -- see scripts/fit/acts_from_serving.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
