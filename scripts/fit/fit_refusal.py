#!/usr/bin/env python3
"""Fit a refusal direction and its sigma calibration: the gate's SECOND factor.

The category factor decides whether a prompt is on-topic. The refusal factor decides whether the
model is under refusal pressure, and the policy fires only when both hold (`s_ref_z >= 1.50`,
`DEFAULT_REF_MAINT_Z_THRESH`). A family with no refusal direction falls back to `mu = 0.0`,
`sd = 1.0`, which leaves `s_ref_z` an uncalibrated raw cosine -- the 1.50 floor is then
unreachable in practice and the gate quietly runs on one factor.

So the calibration is not optional trimming. `mu_hm_cos` and `sd_hm_cos` ARE the sigma scale every
refusal threshold in the policy is expressed in:

    s_ref_z = (s_ref_cos - mu_hm_cos) / sd_hm_cos

Method, as recovered from the code that fitted the shipped artifacts
(`profile_ab.py:143-150`, and `direction.py:83-93` for the 5-prompt variant, both in the internal
fitting pipeline rather than this repository) -- at `--layer` with `last` pooling, on RAW rows, not unit-normalised ones:

    ab     = mean(harmless)                     # harmless centroid
    hb     = mean(harmful)
    gd     = unit(ab)
    raw    = unit(hb - ab)
    r_unit = unit(raw - <raw, gd> gd)           # orthogonalize against the harmless centroid
    sep    = ||hb - ab|| / mean_i ||g_i - ab||  # separation, in units of harmless spread

That last orthogonalization is load-bearing and is NOT cosmetic. Without it the direction keeps a
component along the harmless centroid -- i.e. along "generic prompt" -- which pushes `mu_hm_cos`
off zero, costs false positives, and tilts the refusal axis towards the category axis. Dropping it
yields cosine 0.86 to the shipped gemma4 direction, with `mu_hm_cos` of the WRONG SIGN. The
projection is the one published in Arditi et al. 2024 (arXiv:2406.11717); Heretic implements it as
`orthogonalize_direction` (github.com/p-e-w/heretic), which is what this was checked against. See
ATTRIBUTION.md -- that package is AGPL-3.0-or-later and this is an independent implementation.

Rows are unit-normalised for the CALIBRATION (a cosine per row) but not for building the
direction; the two are easy to conflate and the difference is measurable.

The row sets choose the variant, so this one function covers both shipped recipes: the full corpus
(692 derivation harmful + 554 harmless) reproduces `dir_gemma4-26B_sep.pt` at cosine 1.0000000000,
and 1 harmful + 4 harmless reproduces the 5-prompt "blind" kit that gemma4 actually ships. See
`README.md` for the reproduction table and the corpus conventions, which are part of the method:
feed it the held-out `para_*`/`adv_*` rows and the calibration shifts.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from cybergate.families import family_by_name  # noqa: E402

from fit_category import load_prompts, model_shape, pool_activations  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--layer", type=int, required=True, help="refusal layer, well above the gate")
    ap.add_argument("--harmful", required=True, help="rows that build the direction")
    ap.add_argument("--harmless", required=True, help="rows that build the direction")
    # The direction can come from a handful of prompts; the sigma scale cannot. gemma4's shipped
    # artifact is exactly this split: direction from a 5-prompt kit, calibration from 692+554.
    ap.add_argument("--cal-harmful", default=None,
                    help="calibration corpus; defaults to --harmful")
    ap.add_argument("--cal-harmless", default=None,
                    help="calibration corpus; defaults to --harmless")
    ap.add_argument("--pool", default="last", choices=["last", "mean_user"])
    ap.add_argument("--render", default="fit", choices=["fit", "serving"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--acts-cache", default=None)
    ap.add_argument("--verify-against", default=None)
    args = ap.parse_args()

    family = family_by_name(args.family)
    harmful, harmless = load_prompts(args.harmful), load_prompts(args.harmless)
    cal_harmful = load_prompts(args.cal_harmful) if args.cal_harmful else harmful
    cal_harmless = load_prompts(args.cal_harmless) if args.cal_harmless else harmless
    print(f"[1] direction: {len(harmful)} harmful, {len(harmless)} harmless")
    print(f"    calibration: {len(cal_harmful)} harmful, {len(cal_harmless)} harmless"
          f"{'' if args.cal_harmful or args.cal_harmless else '  (same rows)'}")
    # sd_hm_cos is the unit every refusal threshold in the policy is expressed in. A single row
    # makes it NaN, which would ship silently and make s_ref_z meaningless.
    for name, rows in (("harmful", cal_harmful), ("harmless", cal_harmless)):
        if len(rows) < 2:
            raise SystemExit(f"calibration needs >= 2 {name} rows, got {len(rows)}; "
                             f"pass --cal-{name} with a real corpus")

    cache = pathlib.Path(args.acts_cache) if args.acts_cache else None
    if cache and cache.exists():
        z = np.load(cache)
        A_harm, A_hm = z["harm"], z["hm"]
        C_harm = z["cal_harm"] if "cal_harm" in z else A_harm
        C_hm = z["cal_hm"] if "cal_hm" in z else A_hm
        print(f"[2] activations reused from {cache}")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[2] loading {args.model} in bfloat16 ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
        device = next(model.parameters()).device
        hidden, nl = model_shape(model.config)
        print(f"    hidden={hidden} layers={nl} device={device}")
        if args.layer >= nl:
            raise SystemExit(f"--layer {args.layer} does not exist on a {nl}-layer model")
        cap = lambda rows: pool_activations(  # noqa: E731
            model, tokenizer, family, [r["prompt"] for r in rows], args.layer, args.pool, device,
            hidden, render=args.render)
        print("    harmful ...", flush=True);  A_harm = cap(harmful)
        print("    harmless ...", flush=True); A_hm = cap(harmless)
        if cal_harmful is harmful and cal_harmless is harmless:
            C_harm, C_hm = A_harm, A_hm
        else:
            print("    calibration harmful ...", flush=True);  C_harm = cap(cal_harmful)
            print("    calibration harmless ...", flush=True); C_hm = cap(cal_harmless)
        if cache:
            np.savez_compressed(cache, harm=A_harm, hm=A_hm, cal_harm=C_harm, cal_hm=C_hm)
            print(f"    cached -> {cache}")

    # Direction from RAW row means. Unit-normalising the rows first is a different direction.
    ab = A_hm.mean(axis=0)
    hb = A_harm.mean(axis=0)
    spread = float(np.linalg.norm(A_hm - ab, axis=1).mean())
    gap = float(np.linalg.norm(hb - ab))
    sep = gap / spread if spread > 1e-9 else 0.0
    if len(A_hm) < 2:
        # sep's denominator is identically 0 for one harmless row -- undefined, not small.
        raise SystemExit(f"sep is undefined with {len(A_hm)} harmless row(s); use >= 2 "
                         "(the shipped 5-prompt kit uses 4)")
    gd = ab / float(np.linalg.norm(ab))
    raw = hb - ab
    raw /= float(np.linalg.norm(raw))
    r_unit = raw - float(raw @ gd) * gd              # project out the harmless centroid
    r_unit /= float(np.linalg.norm(r_unit))

    # Rows ARE unit-normalised here: the calibration is a cosine per row.
    U_harm = C_harm / np.linalg.norm(C_harm, axis=1, keepdims=True)
    U_hm = C_hm / np.linalg.norm(C_hm, axis=1, keepdims=True)
    cos_hm, cos_harm = U_hm @ r_unit, U_harm @ r_unit
    mu_hm, sd_hm = float(cos_hm.mean()), float(cos_hm.std(ddof=1))
    cal = {
        "mu_hm_cos": mu_hm, "sd_hm_cos": sd_hm,
        "mu_harm_cos": float(cos_harm.mean()), "sd_harm_cos": float(cos_harm.std(ddof=1)),
        "z_harm_mean": float((cos_harm.mean() - mu_hm) / sd_hm),
        "p95_hm_cos": float(np.percentile(cos_hm, 95)),
        "p99_hm_cos": float(np.percentile(cos_hm, 99)),
        "p05_harm_cos": float(np.percentile(cos_harm, 5)),
        "p10_harm_cos": float(np.percentile(cos_harm, 10)),
    }
    z_hm = (cos_hm - mu_hm) / sd_hm
    print(f"[3] sep = {sep:.9f}  (gap {gap:.6f} / harmless spread {spread:.6f})")
    print(f"    <r_unit, unit(harmless centroid)> = {float(r_unit @ gd):+.3e}  (0 after orthogonalize)")
    print("    calibration")
    for key in ("mu_hm_cos", "sd_hm_cos", "mu_harm_cos", "z_harm_mean"):
        print(f"    {key:<14} {cal[key]:+.12f}")
    print(f"    harmless rows reaching the 1.50 sigma floor: "
          f"{int((z_hm >= 1.5).sum())}/{len(z_hm)}  (these are the false-positive risk)")
    print(f"    harmful  rows reaching it: "
          f"{int(((cos_harm - mu_hm) / sd_hm >= 1.5).sum())}/{len(cos_harm)}")

    if args.verify_against:
        shipped = json.loads(pathlib.Path(args.verify_against).read_text())
        sv = np.array(shipped.get("r_unit") or shipped["c"], dtype=np.float64)
        sv /= np.linalg.norm(sv)
        sm = shipped.get("meta") or {}
        print(f"\n[4] verification against {args.verify_against}")
        print(f"    {'cosine to shipped direction':<28} got {abs(float(r_unit @ sv)):+.8f}  want ~1.0")
        for key in ("mu_hm_cos", "sd_hm_cos", "z_harm_mean"):
            print(f"    {key:<28} got {cal[key]:+.12f}  shipped {sm.get(key)}")

    meta = dict(cal)
    meta.update({"gate_layer": args.layer, "pool": args.pool, "category": "refusal_monitor",
                 "model_key": (family.model_key_prefixes or (args.family,))[0],
                 "model_tag": args.family, "fit_render": args.render,
                 "n_harmless": len(harmless), "n_harmful": len(harmful), "sep": sep,
                 "n_cal_harmless": len(cal_harmless), "n_cal_harmful": len(cal_harmful),
                 "fitted_by": "scripts/fit/fit_refusal.py"})
    pathlib.Path(args.out).write_text(json.dumps({
        "model": args.model, "layer": args.layer, "hidden_dim": int(r_unit.shape[0]),
        "source": f"fit_refusal.py; difference-of-means + orthogonalize vs harmless centroid "
                  f"at L{args.layer} {args.pool}",
        "r_unit": [float(x) for x in r_unit], "meta": meta}, indent=1))
    print(f"\n[5] wrote {args.out}")
    print("    Add to the family: ref_layer, ref_mu_hm, ref_sd_hm -- and set "
          "default_ref_dir_name to this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
