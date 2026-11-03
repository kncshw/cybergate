#!/usr/bin/env python3
"""Fit the category direction at every layer of a vLLM dump, and rank layers OUT OF SAMPLE.

This exists because of a specific failure. qwen38's shipped direction was fitted on 240 rows in
5120 dimensions, reported AUC 0.9974, and was shipped on that number. Measured live it scored
0.5893 at its own gate layer -- chance. A ridge fit of 240 rows in 5120 dimensions separates
almost any labelling, so an in-sample AUC is not evidence of anything and must never again pick a
layer or justify a kit.

So the layer is ranked on **held-out recall**, exactly the way the gemma4 kit's headline number is
produced: tau comes from R3 on the fit references, which never sees an evaluation corpus, and
recall is then measured on positives the fit never touched. The in-sample AUC is still printed --
next to the held-out number, so the gap between them is visible rather than flattering.

A positive control runs first. The dumped row for a prompt must reproduce the `X<layer>_mean_user`
the serving hook recorded for that same prompt. If it does not, the dump is not the vectors the
gate scores and nothing below it means anything.

    python3 scripts/fit/sweep_layers_from_dump.py \\
      --order /content/fit_order.json --pool mean_user \\
      --fit-pos pos --fit-neg neg --holdout cysec_val \\
      --ipc-dir /dev/shm/cybergate_ipc_colab \\
      --control-op kits/qwen38/qwen38_L39_M1_serving.json --control-layer 39
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fit_category import cosines, fit_m1_subcat_tridge, threshold_r3   # noqa: E402

CONTROL_TOL = 2e-3      # fp16 dump against the hook's fp32 dot, rounded to 6 d.p. in the record


def rows_for(dump: pathlib.Path, layer: int, pool: str, hidden: int, n: int) -> np.ndarray:
    path = dump / f"L{layer}{'' if pool == 'mean_user' else '_last'}.bin"
    raw = np.fromfile(path, dtype=np.float16)
    if raw.size != n * hidden:
        raise SystemExit(f"{path}: {raw.size} values, expected {n * hidden} "
                         f"({n} rows x {hidden}). The replay and the dump disagree.")
    return raw.reshape(n, hidden).astype(np.float64)


def control(dump: pathlib.Path, order: list[dict], hidden: int, ipc: pathlib.Path,
            op_path: pathlib.Path, layer: int, pool: str) -> None:
    """The dumped row must reproduce the score the hook published for the same prompt."""
    c = np.array(json.loads(op_path.read_text())["c"], dtype=np.float64)
    c /= np.linalg.norm(c)
    X = rows_for(dump, layer, pool, hidden, len(order))
    key = f"X{layer}_{'mean_user' if pool == 'mean_user' else 'last'}"

    checked, worst, missing = 0, 0.0, 0
    for i, r in enumerate(order):
        sha = hashlib.sha256(r["text"].encode()).hexdigest()
        rec = next((json.loads(p.read_text()) for p in
                    (ipc / f"sha_st_{sha}.json", ipc / f"sha_{sha}.json") if p.is_file()), None)
        if rec is None:
            missing += 1
            continue
        want = (rec.get("multi_layer_scores") or {}).get(key)
        if want is None:
            missing += 1
            continue
        got = float(X[i] @ c / np.linalg.norm(X[i]))
        worst = max(worst, abs(got - float(want)))
        checked += 1

    print(f"POSITIVE CONTROL  dumped row vs recorded {key}")
    print(f"  checked {checked} of {len(order)} rows ({missing} without that key in the record)")
    print(f"  worst absolute difference {worst:.2e}  tolerance {CONTROL_TOL:.0e}")
    if checked == 0:
        raise SystemExit("  FAILED: nothing could be checked. Without this the sweep is "
                         "uninterpretable -- was CYBERGATE_CROSS_LAYER_DIAG set during the replay?")
    if worst > CONTROL_TOL:
        raise SystemExit("  FAILED: the dumped vectors are not what the gate scored. Stop here.")
    print("  PASSED: the dump is the vectors the gate sees.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", required=True, help="written by harvest_vllm_acts.py")
    ap.add_argument("--dump-dir", default=None, help="defaults to the one named in --order")
    ap.add_argument("--pool", default="mean_user", choices=["mean_user", "last"])
    ap.add_argument("--fit-pos", default="pos")
    ap.add_argument("--fit-neg", default="neg")
    # Required, not optional. Ranking layers on an in-sample number is the mistake this
    # script exists to prevent, and a missing holdout used to rank on NaN and pick layer 0.
    ap.add_argument("--holdout", required=True, help="split of positives the fit never sees")
    ap.add_argument("--leak", action="append", default=[], metavar="SPLIT",
                    help="repeatable: NON-TARGET splits the fit never sees. Recall without a "
                         "leak number is not a result -- a direction that fires on everything "
                         "scores recall 1.0. Given any --leak, layers rank by recall minus the "
                         "worst leak rather than by recall alone.")
    ap.add_argument("--layers", default=None, help="e.g. '8,11,25,34,39'; default every layer found")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--ipc-dir", default=None, help="enables the positive control")
    ap.add_argument("--control-op", default=None, help="shipped operator for the control")
    ap.add_argument("--control-layer", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the best layer's operator here")
    args = ap.parse_args()

    meta = json.loads(pathlib.Path(args.order).read_text())
    order, hidden = meta["order"], meta["hidden"]
    dump = pathlib.Path(args.dump_dir or meta["dump_dir"])
    at, idx = 0, {}
    for label, n in meta["splits"]:
        idx[label] = (at, at + n)
        at += n
    print(f"{len(order)} rows  hidden {hidden}  pool {args.pool}  dump {dump}")
    print("splits: " + "  ".join(f"{k}=[{a}:{b})" for k, (a, b) in idx.items()) + "\n")

    if args.ipc_dir and args.control_op and args.control_layer is not None:
        control(dump, order, hidden, pathlib.Path(args.ipc_dir),
                pathlib.Path(args.control_op), args.control_layer, args.pool)
    else:
        print("POSITIVE CONTROL SKIPPED -- pass --ipc-dir, --control-op and --control-layer.\n")

    if args.layers:
        layers = [int(x) for x in args.layers.replace(",", " ").split()]
    else:
        suf = "" if args.pool == "mean_user" else "_last"
        layers = sorted(int(p.stem[1:-len(suf)] if suf else p.stem[1:])
                        for p in dump.glob(f"L*{suf}.bin")
                        if p.stem.startswith("L") and (p.stem.endswith(suf) if suf else
                                                       not p.stem.endswith("_last")))

    pa, pb = idx[args.fit_pos]
    na, nb = idx[args.fit_neg]
    subcat = [r.get("cat") or r["key"].rsplit("-", 1)[0] for r in order[na:nb]]
    if any(r.get("cat") is None for r in order[na:nb]):
        print(f"WARNING: {sum(r.get('cat') is None for r in order[na:nb])} negatives carry no "
              f"`cat`; subcategory centring fell back to the key prefix.")
    print(f"negatives span {len(set(subcat))} subcategories (the reference set has 14)\n")
    if args.holdout not in idx:
        raise SystemExit(f"--holdout {args.holdout!r} is not a split in {args.order}. "
                         f"Have: {sorted(idx)}")
    hold = idx[args.holdout]
    if args.holdout in (args.fit_pos, args.fit_neg):
        raise SystemExit(f"--holdout {args.holdout!r} is also being fitted on; it would not "
                         f"be held out.")

    table = []
    for L in layers:
        X = rows_for(dump, L, args.pool, hidden, len(order))
        c_unit, _ = fit_m1_subcat_tridge(X[pa:pb], X[na:nb], subcat, k=args.k)
        s_pos, s_neg = cosines(X[pa:pb], c_unit), cosines(X[na:nb], c_unit)
        tau, _ = threshold_r3(c_unit, X[na:nb])
        auc_in = float(np.mean((s_pos[:, None] > s_neg[None, :]) +
                               0.5 * (s_pos[:, None] == s_neg[None, :])))
        recall_in = float(np.mean(s_pos >= tau))
        s_h = cosines(X[hold[0]:hold[1]], c_unit)
        recall_out = float(np.mean(s_h >= tau))
        margin = float(np.median(s_h) - tau)
        leaks = {}
        for name in args.leak:
            a, b = idx[name]
            leaks[name] = float(np.mean(cosines(X[a:b], c_unit) >= tau))
        rank = recall_out - (max(leaks.values()) if leaks else 0.0)
        table.append((rank, recall_out, auc_in, recall_in, tau, L, margin, leaks, c_unit))

    table.sort(reverse=True)
    head = (f"{'layer':>5} {'HELD-OUT recall':>16}"
            + "".join(f" {('leak ' + n)[:13]:>13}" for n in args.leak)
            + f" {'in-sample AUC':>14} {'tau':>11} {'median-tau':>11}")
    print(head)
    for rk, ro, ai, ri, tau, L, mg, lk, _ in table:
        print(f"{L:>5} {ro:>16.4f}"
              + "".join(f" {lk[n]:>13.4f}" for n in args.leak)
              + f" {ai:>14.4f} {tau:>+11.6f} {mg:>+11.6f}")
    if args.leak:
        print("\nranked by held-out recall minus the WORST leak; raw recall is the second column")
    else:
        print("\nNO --leak SPLIT GIVEN. Recall alone cannot distinguish a working direction from "
              "one\nthat fires on everything. Treat this ranking as provisional.")
    n_h = hold[1] - hold[0]
    se = (table[0][1] * (1 - table[0][1]) / n_h) ** 0.5
    print(f"standard error on held-out recall at n={n_h}: {se:.4f}. Layers within ~2x that of each "
          f"other\nare not distinguishable, and taking the best of {len(table)} biases it upward -- "
          f"choose the layer\nhere, then measure once on a split that was never used to choose "
          f"anything.")

    rk, ro, ai, ri, tau, L, mg, lk, c_unit = table[0]
    print(f"\nbest: L{L}  held-out recall {ro:.4f}"
          + ("  leaks " + " ".join(f"{n}={lk[n]:.4f}" for n in args.leak) if args.leak else "")
          + f"  in-sample AUC {ai:.4f}")
    if ai - ro > 0.25:
        print(f"  NOTE: the in-sample AUC exceeds held-out recall by {ai - ro:.3f}. That gap is "
              f"the\n        overfitting this script exists to expose; treat {ai:.4f} as "
              f"uninformative.")
    if args.out:
        X = rows_for(dump, L, args.pool, hidden, len(order))
        _, info = fit_m1_subcat_tridge(X[pa:pb], X[na:nb], subcat, k=args.k)
        _, all_scores = threshold_r3(c_unit, X[na:nb])
        info.update({"gate_layer": L, "pool": args.pool, "model_key": "qwen38",
                     "fit_render": "serving", "fit_source": "vLLM activation dump",
                     "fitted_by": "scripts/fit/sweep_layers_from_dump.py",
                     "serve_thresh_rule": "R3", "thresholds_all": all_scores,
                     "holdout_split": args.holdout, "holdout_recall": ro,
                     "leak_splits": {n: lk[n] for n in args.leak},
                     "in_sample_auc": ai, "in_sample_recall": ri})
        pathlib.Path(args.out).write_text(json.dumps(
            {"c": [float(x) for x in c_unit], "thresh": tau, "meta": info}))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
