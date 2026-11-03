#!/usr/bin/env python3
"""Write a kit's `canaries.json` from the scores a SERVED gate actually produced.

A canary pins a measured verdict so a later run can tell it is serving a different checkpoint.
That only works if the numbers come from the deployed gate: an operator fitted on one
checkpoint of an architecture passes every metadata check, attaches cleanly, and then fires on
almost nothing -- which is what a previously shipped Qwen operator did, on 18 of its own 100
fitted positives. Only a canary catches that.

**Corpus hygiene.** `--from-target` and `--from-leak` default to the fit corpora, not the
evaluation ones. Canaries carry prompt TEXT, and this repository does not redistribute
CySecBench, CatQA, WMDP or SOSBench -- `kits/gemma4` ships nine prompts from its fit set and
records only md5s for the rest. Pass an evaluation split here and you are publishing it.

    python3 scripts/fit/write_canaries_from_records.py \\
      --order /content/live_order.json --ipc-dir /dev/shm/cybergate_ipc_colab \\
      --kit kits/qwen38 --from-target pos --from-leak harmless
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REF_FLOOR = 1.50


def record_for(ipc: pathlib.Path, text: str) -> dict | None:
    sha = hashlib.sha256(text.encode()).hexdigest()
    for name in (f"sha_st_{sha}.json", f"sha_{sha}.json"):
        path = ipc / name
        if path.is_file():
            return json.loads(path.read_text())
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", required=True)
    ap.add_argument("--ipc-dir", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--from-target", default="pos", help="split to draw FIRING canaries from")
    ap.add_argument("--from-leak", default="harmless", help="split to draw QUIET canaries from")
    ap.add_argument("--n-target", type=int, default=5)
    ap.add_argument("--n-leak", type=int, default=4)
    args = ap.parse_args()

    meta = json.loads(pathlib.Path(args.order).read_text())
    ipc = pathlib.Path(args.ipc_dir)
    kit = pathlib.Path(args.kit)
    man = json.loads((kit / "MANIFEST.json").read_text())

    by_split: dict[str, list[tuple[dict, dict]]] = {}
    for row in meta["order"]:
        rec = record_for(ipc, row["text"])
        if rec is not None and rec.get("score") is not None:
            by_split.setdefault(row["split"], []).append((row, rec))

    for name in (args.from_target, args.from_leak):
        if name not in by_split:
            raise SystemExit(f"no records for split {name!r}. Have: {sorted(by_split)}")

    # Highest-scoring firing targets and widest-margin quiet negatives: the ones whose verdict a
    # different checkpoint is most likely to change.
    tgt = sorted((p for p in by_split[args.from_target] if p[1].get("fired")),
                 key=lambda p: -p[1]["score"])[:args.n_target]
    neg = sorted((p for p in by_split[args.from_leak] if not p[1].get("fired")),
                 key=lambda p: p[1]["score"])[:args.n_leak]
    if len(tgt) < args.n_target or len(neg) < args.n_leak:
        raise SystemExit(f"only {len(tgt)} firing targets and {len(neg)} quiet negatives "
                         f"available; asked for {args.n_target} and {args.n_leak}")

    out = {
        "kit": man["kit"],
        "render": "cybergate_mode=baseline, no system prompt, max_tokens=1, temp=0",
        "basis": (
            f"Measured on the DEPLOYED gate: operator {man['files']['operator']} at "
            f"L{man['gate']['layer']} {man['gate']['pool']}, tau "
            f"{man['gate']['threshold']:+.8f}, refusal direction at L"
            f"{json.loads((kit / man['files']['refusal_direction']).read_text())['meta']['gate_layer']}"
            f". Scores are what the worker published, matched by exact prompt sha."),
        "note": (
            f"Targets are the highest-scoring FIRING rows of the '{args.from_target}' split; "
            f"negatives are the widest-margin QUIET rows of '{args.from_leak}'. Both are fit "
            f"corpora, not evaluation corpora -- a canary carries its prompt text and this "
            f"repository does not redistribute the evaluation sets."),
        "ref_floor": REF_FLOOR,
        "canaries": [],
    }
    for role, rows, expect in (("target", tgt, True), ("nontarget", neg, False)):
        for row, rec in rows:
            out["canaries"].append({
                "key": row["key"], "role": role, "prompt": row["text"],
                "expect_fired": expect,
                "measured_score": round(float(rec["score"]), 6),
                "measured_s_ref_z": (round(float(rec["s_ref_z"]), 5)
                                     if rec.get("s_ref_z") is not None else None),
                "span_method": rec.get("span_method"),
                "n_prefill_tokens": rec.get("n_prefill_tokens"),
            })

    path = kit / "canaries.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} with {len(out['canaries'])} canaries\n")
    for c in out["canaries"]:
        print(f"  {c['role']:<9} {c['key']:<28} score {c['measured_score']:+.6f}  "
              f"s_ref_z {c['measured_s_ref_z']}  expect_fired={c['expect_fired']}")
    print("\nAdd \"canaries\": \"canaries.json\" to MANIFEST.json files, so check_eval.sh and the "
          "deploy notebook pick it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
