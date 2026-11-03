#!/usr/bin/env python3
"""Score an EXISTING operator's direction over a harvested dump, at every layer.

`sweep_layers_from_dump.py` fits new directions. This one asks the other question: what does a
direction we already have do on these activations, at its own threshold? That is how you find out
whether a shipped operator transfers, measured on the full corpora rather than a handful of
prompts, and with no server needed -- the dump already holds the vectors.

It exists because of an error worth not repeating. The first attempt at this measurement used a
prompt list typed into a notebook cell, and 9 of its 12 "fit positives" were not in the fit
positives at all -- they were other-harm prompts, which a working cyber direction is supposed to
score LOW, so the resulting AUC of 0.5893 was indistinguishable from the direction being dead.
Reading the splits out of the replay order file removes the opportunity.

    python3 scripts/fit/score_op_from_dump.py \\
      --order /content/fit_order_v2.json --op kits/qwen38/qwen38_L39_M1_serving.json \\
      --target pos --target cysec_val --leak neg --leak catqa --leak harmless
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fit_category import cosines, threshold_r3   # noqa: E402


def auc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", required=True)
    ap.add_argument("--op", required=True, help="the operator whose `c` and `thresh` to use")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--pool", default="mean_user", choices=["mean_user", "last"])
    ap.add_argument("--target", action="append", default=[], help="splits where firing is correct")
    ap.add_argument("--leak", action="append", default=[], help="splits where firing is a leak")
    ap.add_argument("--layers", default=None, help="default: the operator's own layer only")
    ap.add_argument("--auc-pair", default=None, metavar="TARGET:LEAK",
                    help="also report AUC between these two splits, e.g. pos:neg")
    ap.add_argument("--retune-tau", default=None, metavar="SPLIT",
                    help="also report everything at a threshold re-derived by R3 on THIS split, "
                         "which separates 'the direction does not transfer' from 'the threshold "
                         "was calibrated on another engine\'s activations'. The operator\'s own "
                         "tau is kept for the first table.")
    args = ap.parse_args()

    meta = json.loads(pathlib.Path(args.order).read_text())
    hidden, order = meta["hidden"], meta["order"]
    dump = pathlib.Path(args.dump_dir or meta["dump_dir"])
    idx, at = {}, 0
    for name, n in meta["splits"]:
        idx[name] = (at, at + n); at += n

    op = json.loads(pathlib.Path(args.op).read_text())
    c = np.array(op["c"], dtype=np.float64)
    c /= np.linalg.norm(c)
    tau = float(op["thresh"])
    op_layer = int(op["meta"]["gate_layer"])
    print(f"operator {args.op}")
    print(f"  layer L{op_layer}  pool {op['meta'].get('pool')}  tau {tau:+.8f}  "
          f"fit_render {op['meta'].get('fit_render')}")
    print(f"dump {dump}  {len(order)} rows  pool {args.pool}\n")

    layers = [int(x) for x in args.layers.replace(",", " ").split()] if args.layers else [op_layer]
    suffix = "" if args.pool == "mean_user" else "_last"
    want = args.target + args.leak
    for name in want:
        if name not in idx:
            raise SystemExit(f"{name!r} is not a split in {args.order}. Have: {sorted(idx)}")

    for L in layers:
        path = dump / f"L{L}{suffix}.bin"
        raw = np.fromfile(path, dtype=np.float16)
        if raw.size != len(order) * hidden:
            raise SystemExit(f"{path}: {raw.size} values, expected {len(order) * hidden}")
        X = raw.reshape(len(order), hidden).astype(np.float64)
        print(f"L{L}" + ("  <- the operator's own layer" if L == op_layer else ""))
        scores = {}
        for name in want:
            a, b = idx[name]
            s = cosines(X[a:b], c)
            scores[name] = s
            role = "TARGET" if name in args.target else "leak  "
            print(f"   {role} {name:<12} n={len(s):>5}  mean {s.mean():+.6f}  "
                  f"median {np.median(s):+.6f}  >= tau {float(np.mean(s >= tau)):.4f}")
        if args.auc_pair:
            t, _, k = args.auc_pair.partition(":")
            if t in scores and k in scores:
                print(f"   AUC({t} vs {k}) = {auc(scores[t], scores[k]):.4f}")

        if args.retune_tau:
            if args.retune_tau not in idx:
                raise SystemExit(f"--retune-tau {args.retune_tau!r} is not a split")
            a, b = idx[args.retune_tau]
            tau2, _ = threshold_r3(c, X[a:b])
            ratio = f"{tau2 / tau:.3f}" if tau else "n/a"
            print(f"\n   R3 re-derived on {args.retune_tau} in THIS geometry: tau "
                  f"{tau2:+.8f}  (shipped {tau:+.8f}, ratio {ratio})")
            for name in want:
                s = scores[name]
                role = "TARGET" if name in args.target else "leak  "
                print(f"   {role} {name:<12} n={len(s):>5}  >= new tau "
                      f"{float(np.mean(s >= tau2)):.4f}   was {float(np.mean(s >= tau)):.4f}")
            print("   A threshold is not a free parameter: R3 is the 12th-largest score on the "
                  "fit\n   references, so re-deriving it here is the same RULE applied in the "
                  "geometry the\n   gate scores in -- not a threshold tuned to make the numbers "
                  "better.")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
