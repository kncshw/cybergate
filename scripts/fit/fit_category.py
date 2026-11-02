#!/usr/bin/env python3
"""Fit a category direction from scratch: capture activations, then solve.

This is the factors calculation. It is the only part of the pipeline that needs the weights and a
GPU, and it is deliberately separate from serving: `scripts/serving/serve_eval.sh` consumes what
this produces.

Why it renders through the package instead of calling `tokenizer.apply_chat_template`: the score
the gate compares against `thresh` is the cosine of the hidden state *pooled over the user span of
the served render*. A direction fitted on a different render scores differently at serve time --
measured on gemma4, one prompt moves between +0.047 and +0.186 across renders. So the capture uses
`family.build_raw_prompt(mode="baseline")` and `family.locate_latest_user_span()`, the same two
functions the worker hook uses, and cannot drift from them.

Method `M1_subcat_Tridge_lam16`, as recorded in the shipped operator's `meta.fit_method`:

  1. pool L<layer> hidden states over the user span -- `acts[l] = hidden_states[l+1]`, the OUTPUT
     of layer l
  2. centre the positives on their own mean; centre each negative subcategory on ITS OWN mean
     (that is the "subcat" part -- a single global negative mean is the earlier B0 arm)
  3. scatter S = (Zp'Zp + Zn'Zn) / 225, where 225 = (100-1) + (140-14) is the residual dof
  4. eigendecompose S, keep k=15 components, ridge with alpha = lambda_16, the first DISCARDED
     eigenvalue -- which is what "lam16" names
  5. c = sum_{i<=15} (v_i . d) / (lambda_i + alpha) * v_i   for d = mean(pos) - mean(neg)
  6. unit-normalise

Four values recorded in the shipped gemma4 operator let the result be checked without trusting
this reimplementation: `k_or_alpha.alpha`, `cos_to_unit_d`, `thresholds_all.R3`, and the direction
itself. `--verify-against` reports all four.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from cybergate.families import family_by_name  # noqa: E402

M_BUDGET = 11  # R3: tau is set so this many reference prompts fire. The 12th-largest score.


def model_shape(config):
    """Return `(hidden_size, num_hidden_layers)` for a plain or composite config.

    A multimodal checkpoint nests the decoder's shape under `text_config` and exposes neither key
    at the top level -- `Gemma4Config` raises `AttributeError` for `num_hidden_layers`, and
    Qwen3.8-27B is the same shape (`text_config.hidden_size` 5120, `text_config.num_hidden_layers`
    64, beside a `vision_config`). Reading the top level first would silently pick up the vision
    tower's width on some checkpoints, so the text config is preferred where it exists.
    """
    for cfg in (getattr(config, "text_config", None), config):
        if cfg is None:
            continue
        hidden = getattr(cfg, "hidden_size", None)
        layers = getattr(cfg, "num_hidden_layers", None)
        if hidden and layers:
            return int(hidden), int(layers)
    raise RuntimeError(
        f"cannot read hidden_size/num_hidden_layers from {type(config).__name__}; "
        "pass a checkpoint whose config or config.text_config declares both")


#: The system prompt the shipped operators were fitted under. `capture_hf_v2.py` read it from
#: `$SYS` and defaulted to this; the gemma4 capture never set the variable.
FIT_SYSTEM_PROMPT = "You are a helpful assistant."


def render_fit(tokenizer, text, system=FIT_SYSTEM_PROMPT):
    """The fit-time render: the tokenizer's own chat template, with a system turn.

    This is NOT the serving render. The served prompt comes from `family.build_raw_prompt`, which
    adds CyberGate's thought block and its own turn markers. Reproducing the shipped operators
    requires this one, because that is what their activations were captured under.
    """
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=False)


def span_fit(tokenizer, rendered, text):
    """Locate the user-content span by token-count arithmetic, as the original capture did.

    Strictly the user's own tokens -- no system prompt, no template scaffolding. Returns `None`
    when the arithmetic is not exact, which the caller must treat as a hard failure rather than
    pooling over the wrong tokens.
    """
    nt = lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    if rendered.count(text) != 1:
        return None
    pre, _, suf = rendered.partition(text)
    a, u, b, tot = nt(pre), nt(text), nt(suf), nt(rendered)
    return (a, a + u) if a + u + b == tot else None


def pool_activations(model, tokenizer, family, texts, layer, pool, device, hidden_size,
                     render="fit", log_every=25):
    """Return an (n, hidden) float64 array of pooled hidden states, one row per text.

    `render="fit"` reproduces `capture_hf_v2.py`; `render="serving"` pools over the span the
    worker hook actually scores at serve time. The two disagree: on gemma4 the same prompt scores
    +0.047 to +0.186 depending on the render, so the choice is load-bearing and is recorded in the
    operator's meta.
    """
    import torch

    out = np.zeros((len(texts), hidden_size), dtype=np.float64)
    bad = []
    for i, text in enumerate(texts):
        if render == "fit":
            raw = render_fit(tokenizer, text)
            add_special = False
            span = span_fit(tokenizer, raw, text)
            if span is None:
                bad.append(i)
                span = None
        else:
            raw = family.build_raw_prompt([{"role": "user", "content": text}], mode="baseline")
            add_special = True
            span = None
        ids = tokenizer(raw, return_tensors="pt", add_special_tokens=add_special).input_ids
        if span is None and render == "serving":
            start, end, method = family.locate_latest_user_span(ids[0].tolist())
            if end <= start:
                raise RuntimeError(f"row {i}: empty user span ({start},{end}) via {method!r}")
            span = (start, end)
        with torch.no_grad():
            hs = model(ids.to(device), output_hidden_states=True).hidden_states
        h = hs[layer + 1][0].to(torch.float32).cpu().numpy()  # OUTPUT of `layer`
        if pool == "last" or span is None:
            out[i] = h[-1]
        else:
            out[i] = h[span[0]:span[1]].mean(axis=0)
        if log_every and (i + 1) % log_every == 0:
            print(f"    captured {i + 1}/{len(texts)}", flush=True)
    if bad:
        raise RuntimeError(
            f"user-span arithmetic was not exact for {len(bad)} of {len(texts)} rows "
            f"(first: {bad[:5]}). Pooling those over the last token instead would quietly mix two "
            f"pooling modes into one fit.")
    return out


#: The scatter divisor the original fit used for the ridge scale. The shipped operator RECORDS
#: `cov_divisor: 225` = (100-1) + (140-14), the residual dof, but its recorded
#: `alpha` = 0.0009657231624378617 only comes out of a divisor of 99 = n_pos - 1 (225 gives
#: 0.000426, a factor of 2.27 too small). The divisor cannot change the fitted direction -- it
#: scales the eigenvalues and alpha together and the ridge solution is invariant to that -- so this
#: discrepancy is bookkeeping in the original record, not behaviour. 99 is used here so the
#: reported alpha is comparable with the shipped operator's.
ALPHA_DIVISOR_IS_N_POS_MINUS_1 = True


def fit_m1_subcat_tridge(X_pos, X_neg, neg_subcat, k=15):
    """Return (unit direction, diagnostics) for `M1_subcat_Tridge_lam16`.

    Rows are unit-normalised first. That is not cosmetic: the gate scores
    `cos(h, c)`, so the fit belongs in the same cosine geometry, and fitting on raw
    hidden states instead moves the result to cosine 0.54 of the shipped direction.

    "Tridge" is a RIDGE over the whole spectrum, not a truncation of it. `k` selects only which
    eigenvalue supplies the ridge constant -- alpha = lambda_{k+1}, the 16th for k=15, which is
    what "lam16" names. Whitening inside the top-k subspace instead discards most of the mean
    difference and lands at cosine 0.53. Both of those were wrong here before 2026-11-01;
    `--verify-against` on the shipped gemma4 operator is what caught it, at cosine 0.99946.
    """
    n_pos, n_neg = len(X_pos), len(X_neg)
    cats = sorted(set(neg_subcat))

    U_pos = X_pos / np.linalg.norm(X_pos, axis=1, keepdims=True)
    U_neg = X_neg / np.linalg.norm(X_neg, axis=1, keepdims=True)

    Z_pos = U_pos - U_pos.mean(axis=0, keepdims=True)
    Z_neg = np.empty_like(U_neg)
    for c in cats:
        m = np.array([s == c for s in neg_subcat])
        Z_neg[m] = U_neg[m] - U_neg[m].mean(axis=0, keepdims=True)

    dof = (n_pos - 1) + (n_neg - len(cats))
    divisor = (n_pos - 1) if ALPHA_DIVISOR_IS_N_POS_MINUS_1 else dof
    S = (Z_pos.T @ Z_pos + Z_neg.T @ Z_neg) / divisor

    evals, evecs = np.linalg.eigh(S)            # ascending
    evals, evecs = evals[::-1], evecs[:, ::-1]  # descending
    alpha = float(evals[k])                     # lambda_{k+1}

    d = U_pos.mean(axis=0) - U_neg.mean(axis=0)
    c = evecs @ ((evecs.T @ d) / (np.maximum(evals, 0.0) + alpha))

    c_unit = c / float(np.linalg.norm(c))
    d_unit = d / float(np.linalg.norm(d))
    return c_unit, {
        "fit_method": "M1_subcat_Tridge_lam16",
        "fit_n_pos": n_pos,
        "fit_n_neg": n_neg,
        "fit_n_neg_cats": len(cats),
        "fit_n_rows": n_pos + n_neg,
        "cov_divisor": dof,
        "alpha_divisor": divisor,
        "k_or_alpha": {"k": k, "alpha": alpha},
        "cos_to_unit_d": float(c_unit @ d_unit),
    }


#: R3 takes the boundary reference score as tau directly, with no margin. Measured 2026-11-01: the
#: 12th-largest reference score in the fit render is +0.082738917947366 and the shipped gemma4 tau
#: is +0.08284617644833335 -- 0.13% apart, which is capture noise, not a margin. An earlier guess
#: here subtracted 0.002, because the 12th-largest score in the SERVING render is +0.084846 and
#: that differs from tau by almost exactly 0.002. That was two renders being compared, not a rule.
R3_MARGIN = 0.0


def threshold_r3(c_unit, X_ref, m_budget=M_BUDGET, margin=R3_MARGIN):
    """R3: the (m_budget+1)-th largest reference score, less `margin` (0 by default).

    Returns `(tau, sorted_scores)`. Ties at the boundary mean the count that actually fires can
    exceed `m_budget` -- on the shipped gemma4 fit two reference prompts tie at the boundary, which
    is why 12 of the 140 fire against a budget of 11. That is a property of the data, not a bug.
    """
    scores = np.sort((X_ref @ c_unit) / np.linalg.norm(X_ref, axis=1))[::-1]
    return float(scores[m_budget]) - margin, [float(x) for x in scores]


def cosines(X, c_unit):
    return (X @ c_unit) / np.linalg.norm(X, axis=1)


def load_prompts(path, key="prompt"):
    raw = json.loads(pathlib.Path(path).read_text())
    rows = raw if isinstance(raw, list) else (raw.get("rows") or raw.get("prompts") or [])
    out = []
    for r in rows:
        if isinstance(r, str):
            out.append({"prompt": r, "key": None, "cat": None})
        else:
            out.append({
                "prompt": r.get(key) or r.get("text") or r.get("content"),
                "key": r.get("key") or r.get("id"),
                "cat": r.get("cat") or r.get("category") or r.get("subcategory"),
            })
    if any(o["prompt"] is None for o in out):
        raise SystemExit(f"{path}: some rows have no {key!r} field")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--family", required=True, help="registered family name, e.g. gemma4")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pool", default="mean_user", choices=["mean_user", "last"])
    ap.add_argument("--render", default="fit", choices=["fit", "serving"],
                    help="fit: the tokenizer chat template + system prompt the shipped operators "
                         "were captured under. serving: the render the worker hook scores.")
    ap.add_argument("--pos", required=True, help="positives (100 target-category prompts)")
    ap.add_argument("--neg", required=True, help="negatives (140 across subcategories)")
    ap.add_argument("--ref", default=None, help="threshold reference set; defaults to --neg")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", required=True, help="operator .json to write")
    ap.add_argument("--acts-cache", default=None, help="npz to save/reuse captured activations")
    ap.add_argument("--verify-against", default=None, help="a shipped operator to compare with")
    ap.add_argument("--category", default="cyber_offense")
    ap.add_argument("--model-key", default=None)
    args = ap.parse_args()

    family = family_by_name(args.family)
    pos, neg = load_prompts(args.pos), load_prompts(args.neg)
    ref = load_prompts(args.ref) if args.ref else neg
    print(f"[1] prompts: {len(pos)} positive, {len(neg)} negative "
          f"({len(set(r['cat'] for r in neg))} subcategories), {len(ref)} reference")
    if any(r["cat"] is None for r in neg):
        raise SystemExit("every negative row needs a subcategory: M1 centres each one on its own mean")

    cache = pathlib.Path(args.acts_cache) if args.acts_cache else None
    if cache and cache.exists():
        z = np.load(cache)
        X_pos, X_neg, X_ref = z["pos"], z["neg"], z["ref"]
        print(f"[2] activations reused from {cache}")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[2] loading {args.model} in bfloat16 ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        model.eval()
        device = next(model.parameters()).device
        hidden, nl = model_shape(model.config)
        print(f"    hidden={hidden} layers={nl} device={device} config={type(model.config).__name__}")
        if args.layer >= nl:
            raise SystemExit(f"--layer {args.layer} does not exist on a {nl}-layer model")
        if family.hidden_size and hidden != family.hidden_size:
            raise SystemExit(
                f"checkpoint is {hidden}-dim but family {family.name!r} declares "
                f"{family.hidden_size}; fitting here would produce an operator the kit check rejects")
        cap = lambda rows: pool_activations(  # noqa: E731
            model, tokenizer, family, [r["prompt"] for r in rows], args.layer, args.pool, device,
            hidden, render=args.render)
        print("    positives ...", flush=True);  X_pos = cap(pos)
        print("    negatives ...", flush=True);  X_neg = cap(neg)
        X_ref = X_neg if ref is neg else cap(ref)
        if cache:
            np.savez_compressed(cache, pos=X_pos, neg=X_neg, ref=X_ref)
            print(f"    cached activations -> {cache}")

    print("[3] solving M1_subcat_Tridge_lam16 ...")
    c_unit, diag = fit_m1_subcat_tridge(X_pos, X_neg, [r["cat"] for r in neg], k=args.k)
    thresh, ref_scores = threshold_r3(c_unit, X_ref)
    s_pos = cosines(X_pos, c_unit)
    print(f"    alpha (lambda_{args.k + 1})  {diag['k_or_alpha']['alpha']:.18g}")
    print(f"    cov_divisor            {diag['cov_divisor']}")
    print(f"    cos_to_unit_d          {diag['cos_to_unit_d']:.16g}")
    print(f"    boundary score (12th)  {ref_scores[M_BUDGET]:+.17g}")
    print(f"    R3 threshold           {thresh:+.17g}")
    print(f"    fit positives firing   {int((s_pos >= thresh).sum())}/{len(s_pos)}")
    print(f"    reference firing       {int((np.array(ref_scores) >= thresh).sum())}/{len(ref_scores)}"
          f"   (R3 targets {M_BUDGET})")

    if args.verify_against:
        shipped = json.loads(pathlib.Path(args.verify_against).read_text())
        sc = np.array(shipped.get("c") or shipped["r_unit"], dtype=np.float64)
        sc /= np.linalg.norm(sc)
        sm = shipped.get("meta") or {}
        cos = float(c_unit @ sc)
        print("\n[4] verification against", args.verify_against)
        rows = [("cosine to shipped direction", f"{cos:+.8f}", "1.0"),
                ("cos_to_unit_d", f"{diag['cos_to_unit_d']:.10f}",
                 f"{sm.get('cos_to_unit_d')}"),
                ("alpha", f"{diag['k_or_alpha']['alpha']:.12g}",
                 f"{(sm.get('k_or_alpha') or {}).get('alpha')}"),
                ("R3 threshold", f"{thresh:+.17g}",
                 f"{(sm.get('thresholds_all') or {}).get('R3', shipped.get('thresh'))}"),
                ("  boundary ref score", f"{ref_scores[M_BUDGET]:+.17g}", "(== tau, no margin)"),
                ("cov_divisor", str(diag["cov_divisor"]), str(sm.get("cov_divisor")))]
        for label, got, want in rows:
            print(f"    {label:<28} got {got:<22} shipped {want}")

    meta = dict(diag)
    meta.update({"gate_layer": args.layer, "pool": args.pool, "category": args.category,
                 "model_key": args.model_key or (family.model_key_prefixes or (args.family,))[0],
                 "model_tag": args.family,
                 "serve_thresh_rule": "R3", "m_budget": M_BUDGET, "fit_render": args.render,
                 "fit_system_prompt": FIT_SYSTEM_PROMPT if args.render == "fit" else None,
                 "thresholds_all": {"R3": thresh}, "fit_model": args.model,
                 "fitted_by": "scripts/fit/fit_category.py"})
    meta.pop("eigenvalues_top", None)
    pathlib.Path(args.out).write_text(json.dumps(
        {"c": [float(x) for x in c_unit], "thresh": thresh, "meta": meta}, indent=1))
    print(f"\n[5] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
