#!/usr/bin/env python3
"""Turn a vLLM activation dump into the `.npz` that `fit_category.py` and `fit_refusal.py` read.

**Why this exists.** Both fit scripts capture activations through `transformers`, and the gate then
scores them under vLLM. That is an unstated assumption: the two engines compute the same weights
with different kernels, and they only have to agree for the fit to transfer. On gemma4 and qwen36 --
standard attention throughout -- they agree well enough. On Qwen3.8, where 48 of 64 layers are
Gated DeltaNet and `transformers` runs reference PyTorch fallbacks for `causal_conv1d` and
`chunk_gated_delta_rule` while vLLM runs its own kernels, the fitted direction scored +0.20 in the
capture and +0.02 to +0.05 served, and the gate never fired.

Fitting on vLLM's own pooled vectors removes the assumption instead of testing it: the capture
engine and the serving engine become the same program, and `tau` is calibrated in the geometry the
gate actually scores in.

The serving hook writes one fp16 row per request per layer, appended, in request order:

    <dump>/L<layer>.bin        `mean_user`, pooled over the user span  -> the CATEGORY factor
    <dump>/L<layer>_last.bin   the last prefill position               -> the REFUSAL factor

armed by `CYBERGATE_DUMP_ACTS_DIR` (and narrowed by `CYBERGATE_DUMP_ACTS_LAYERS`). Row `i` is
request `i`, so **the replay must be strictly sequential** -- one request in flight, no retries.
`--split` names the corpora in the order they were replayed, and the total is checked against the
file size, which is what catches a retried or concurrent request having inserted a row. A silent
off-by-one here mislabels every row after it and produces a direction that looks fine.

    python3 scripts/fit/acts_from_serving.py \\
      --dump-dir /content/acts_vllm --layer 39 --pool mean_user --hidden 5120 \\
      --split pos=100 --split neg=140 --out /content/acts_qwen38_vllm.npz

    python3 scripts/fit/fit_category.py ... --acts-cache /content/acts_qwen38_vllm.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

#: `fit_category.py` reads `pos`/`neg`/`ref`; `fit_refusal.py` reads `harm`/`hm` and optionally
#: `cal_harm`/`cal_hm`. `ref` and the two `cal_*` keys default to another split, exactly as the
#: scripts' own `--ref` / `--cal-*` flags do, so a two-corpus replay still produces a complete file.
DEFAULTED_KEYS = {"ref": "neg", "cal_harm": "harm", "cal_hm": "hm"}


def load_rows(path: pathlib.Path, hidden: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float16)
    if raw.size == 0:
        raise SystemExit(f"{path} is empty: the hook never wrote a row. Check that the serve had "
                         f"CYBERGATE_DUMP_ACTS_DIR set and that the probe actually ran.")
    if raw.size % hidden:
        raise SystemExit(f"{path} holds {raw.size} values, which is not a multiple of "
                         f"--hidden {hidden}. Wrong layer width or a truncated write.")
    return raw.reshape(-1, hidden).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pool", default="mean_user", choices=["mean_user", "last"])
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--split", action="append", required=True, metavar="NAME=COUNT",
                    help="corpora in REPLAY ORDER, e.g. --split pos=100 --split neg=140")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits: list[tuple[str, int]] = []
    for item in args.split:
        name, _, count = item.partition("=")
        if not count.isdigit() or int(count) <= 0:
            raise SystemExit(f"--split {item!r} must be NAME=COUNT with a positive count")
        splits.append((name, int(count)))

    suffix = "" if args.pool == "mean_user" else "_last"
    path = pathlib.Path(args.dump_dir) / f"L{args.layer}{suffix}.bin"
    if not path.is_file():
        have = sorted(p.name for p in pathlib.Path(args.dump_dir).glob("*.bin"))
        raise SystemExit(f"{path} does not exist. Present: {have or 'nothing'}. "
                         f"CYBERGATE_DUMP_ACTS_LAYERS may have excluded L{args.layer}.")

    rows = load_rows(path, args.hidden)
    want = sum(n for _, n in splits)
    if rows.shape[0] != want:
        raise SystemExit(
            f"{path} holds {rows.shape[0]} rows but --split totals {want}. Row i must be request "
            f"i, so this cannot be sliced safely -- a retried or concurrent request inserts a row "
            f"and every label after it shifts. Re-run the replay strictly sequentially.")

    out: dict[str, np.ndarray] = {}
    at = 0
    for name, n in splits:
        out[name] = rows[at:at + n]
        at += n
        print(f"  {name:<10} {n:>5} rows")
    for key, source in DEFAULTED_KEYS.items():
        if key not in out and source in out:
            out[key] = out[source]
            print(f"  {key:<10} {out[key].shape[0]:>5} rows  (defaulted to {source})")

    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out}  keys={sorted(out)}  hidden={args.hidden} layer={args.layer} "
          f"pool={args.pool}")
    print("Feed it to fit_category.py / fit_refusal.py with --acts-cache; neither will load a "
          "model, so this runs with numpy alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
