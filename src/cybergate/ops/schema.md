# `ops/*` file contract

These files are **read-only inputs**. This document describes them; it does not propose changes to
them. The maths that produced them is out of scope for the refactor (see
`docs/ARCHITECTURE.md`, "Method questions that are out of scope here").

## Operator files (`<family>_L<layer>_M1.json`)

```json
{
  "c":      [ ... float ... ],
  "thresh": 0.10807703530099158,
  "meta": {
    "gate_layer": 10,
    "pool":       "mean_user",
    "category":   "cyber_offense",
    "model_key":  "gemma",
    "model_tag":  "gemma4",
    "c_last":     [ ... float ... ],
    "...":        "fit provenance, see below"
  }
}
```

| Field | Read by | Meaning |
|---|---|---|
| `c` | `loader` | the probe direction. **Unit-normalised at load time**, so a file whose vector is unnormalised still yields a cosine. A `r_unit` key is accepted as an alias. |
| `thresh` | the policy | the firing threshold on `s_cat`. Missing or null becomes `0.0`, which would fire on everything -- so a file without it is a configuration error, not a default. |
| `meta.gate_layer` | the hook | which decoder layer to attach to. Falls back to a top-level `layer`, then to `10` if the vector is 2816-dimensional and `15` otherwise. |
| `meta.pool` | the hook, the policy | `mean_user` scores the mean over the user turn's tokens; `last` scores the final prefill token. Default `mean_user`. |
| `meta.model_key` | the hook | selects the family, by prefix: `gemma` or `qwen36`. Anything else falls through to the Qwen arm. Falls back to `model_tag`, then to the `model_family` argument. |
| `meta.c_last` | the hook | an optional second direction used for `s_cat_last`. Without it, the last-token score reuses `c`. Both shipped Gemma-4 artifacts carry one (`gemma4_L16_M1.json` and `gemma4_ruleC_serve.json`, 2816 dims each); `qwen36_L21_M1_refit.json` does not. This said "only the Gemma-4 L10 operator ships one" until 2026-11-01, which was wrong even before L10 stopped shipping. |
| `meta.category` | nothing | documentation. Every shipped file says `cyber_offense`. |

Everything else in `meta` is fit provenance (`fit_method`, `arm_name`, `fit_n_rows`, `fit_n_pos`,
`fit_n_neg`, `fit_n_neg_cats`, `cov_divisor`, `k_or_alpha`, `serve_thresh_rule`, `thresholds_all`,
`seed0_aucs`, `delivered`). The plugin reads none of it. It is the record of how the operator was
produced and should be preserved when a file is copied.

### Files present, and what is verified about each

**The shipped operators are in `kits/<name>/`, not here.** Those are the only two that have been
served and measured live:

| File | Layer | Pool | tau | dim |
|---|---|---|---|---|
| `kits/gemma4/gemma4_L16_M1.json` | 16 | `mean_user` | +0.08284618 | 2816 |
| `kits/qwen36/qwen36_L21_M1_refit.json` | 21 | `mean_user` | -0.00022390 | 2048 |

Each kit also carries its refusal direction, its candidate bundle, a `MANIFEST.json` and a
`canaries.json`. See the kit's own README.

There is no top-level `ops/` directory. `kits/` is the only source of servable artifacts, and
`CYBERGATE_OPS_DIR` is how a kit is consumed: the launch scripts point it at the kit directory, so
`OPS_DIR` resolves there and the repo-relative default is never used in a deployment.

It used to carry 12 **superseded** per-layer operators for both families -- gemma4 at L5/L8/L10
(`mean_user` and `last`)/L11/L20/L21 and qwen36 at L9/L13/L15/L16/L18 -- plus two `ruleC` specs,
14 operator files in all. None was ever served, and they
were removed on 2026-11-01 because a deployment ships only what it serves. `git log` in the internal
repository retains them.

Two of them had been **live traps** until 2026-10-31, because the family default ladders led with
them: `gemma4_L10_M1.json`, and `qwen36_L15_M1.json`. (The 18-of-100 self-recall failure often
quoted here belongs to `qwen36_L16_M1.json`, the operator actually shipped before the re-fit -- a
different file. Both qwen36 artifacts attribute it to L16.) That cannot
recur now for two independent reasons: `default_op_names` names only the kit's own operator, and
loading a correctly-shaped operator at the wrong gate layer raises `KitMismatch` naming both
layers. The second is still tested --
`tests/unit/test_kit_binding.py::test_loading_a_superseded_operator_raises_through_the_loader`
builds the case in a tmp directory rather than depending on a shipped artifact.

Phi-4 and Llama-3.1 operators were also present. Both families were measured offline, never served,
and their classes have been removed, so the operators went with them.

## Refusal direction (`gemma4_L21_refusal_dir.json`)

Same loader, different shape -- it has no `thresh`, and carries the calibration instead:

```json
{
  "model": "...", "layer": 21, "hidden_dim": 2816, "source": "...",
  "meta": {
    "gate_layer": 21, "pool": "last", "category": "refusal_monitor",
    "mu_hm_cos": 0.13742038538765958,
    "sd_hm_cos": 0.05130108033692934,
    "mu_harm_cos": ..., "sd_harm_cos": ..., "z_harm_mean": ...,
    "p95_hm_cos": ..., "p99_hm_cos": ..., "p05_harm_cos": ..., "p10_harm_cos": ...
  }
}
```

`mu_hm_cos` and `sd_hm_cos` are the harmless-prompt mean and standard deviation at L21 (n = 554), and
they define the sigma scale every refusal threshold in the policy is expressed in:

```
s_ref_z = (s_ref_cos - mu_hm_cos) / sd_hm_cos
```

Because `thresh` is absent, the loader sets it to `0.0`. That is harmless here -- nothing compares
against the refusal operator's threshold -- but it is why `load_cybergate_op` must not treat a missing
`thresh` as an error.

**This is the only refusal direction in the package.** Every other family falls back to `mu = 0.0`,
`sd = 1.0`, leaving `s_ref_z` an uncalibrated cosine against which the `1.50` and `7.0` thresholds are
unreachable. See `docs/ARCHITECTURE.md`.

## Candidate bundle (`gemma4_candidate_ops_bundle.json`)

```json
{ "layers": { "5": { "mean_user": {"c": [...]}, "last": {"c": [...]} }, "8": {...}, ... } }
```

Telemetry only. A read-only hook is attached at each layer present in BOTH the bundle and the
family's `bundle_probe_layers`, and the scores land on the decision record under
`multi_layer_scores`. They never influence a decision.

The shipped kit bundles are **not** empty, so these hooks do attach on a live server -- measured
2026-11-01:

| kit | layers in the bundle | `bundle_probe_layers` | hooks that attach |
|---|---|---|---|
| `gemma4` | 5-12, 16, 20, 21 | 15, 16, 18-22, 24, 26, 28 | **3** (16, 20, 21) |
| `qwen36` | 5, 8-13, 15, 16, 18-20, 22, 24, 26, 28, 30, 32 | same | **9** |

This file said "zero bytes, so no bundle hooks attach" until 2026-11-01. That was read off `ops/`,
where the bundle is in fact empty; the kit copies carry the real layer sweep (1.4 MB and 1.6 MB) and
every harvested decision record carries `multi_layer_scores` as a result.

## `.pt` operators

**No `.pt` file ships any more.** Seven were removed on 2026-11-01: each duplicated a `.json`
operator that was itself superseded, nothing referenced them, and `.json` is preferred at every call
site. The loader keeps the `.pt` path because the fitting pipeline emits that format and a hand-made
operator may still arrive as one.

`op_<family>_L<layer>_...serve.pt` are PyTorch archives holding the same `{c, thresh, meta}`. The
loader prefers `torch.load`, and falls back to a standard-library zip+pickle reader
(`loader._load_pt_stdlib`) when torch is absent, so the package stays importable and testable without
it. That fallback honours only `torch._utils._rebuild_tensor_v2` and stubs every other `torch` global,
so an unexpected archive layout fails rather than executing anything.

The `.json` files are preferred everywhere.

## Precomputed scores (`precomputed_l8_scores.json`)

**No such table ships.** `load_precomputed_scores` guards on `.exists()` and returns `{}`, so
`PRECOMPUTED_CACHE` is empty unless you point `CYBERGATE_OPS_DIR` at a directory holding one. It was
dropped rather than shipped for two reasons: in every served configuration `CYBERGATE_OPS_DIR` names
the kit, so the table never loaded anyway; and its name says L8 while gemma4 now gates at L16, with
nothing inside the file declaring which probe produced its numbers -- a hit would be labelled with
the *live* operator's layer. The format below is the contract for anyone supplying their own.

```json
{ "by_key":       { "<prompt_key>": { "gemma": {"s_cat_M1": ..., "fired_M1": ..., "id": ...} } },
  "by_sha256":    { "<sha256>":     { ... } },
  "by_text_norm": { "<lowercased>": { ... } } }
```

An offline escape hatch for running the decision path with no GPU, used by the dry-run paths. Looked
up by key, then sha, then lowercased text. The inner family tag is `gemma` or `qwen36` -- the
two-valued surface, not the four family names.

A record resolved from here carries its own `fired_M1` and **does not get the two-factor gate**,
because it has no refusal reading to apply it with. That asymmetry is why `evidence` resolves `fired`
for prior turns rather than leaving it to the policy.

## Changing a file

Don't, as part of a refactor. Three things depend on these bytes:

1. the equivalence corpus pins `thresh` values as literals;
2. `tests/fixtures/*.json` copy the real thresholds, so a change there must be mirrored;
3. the published numbers in the paper were measured at these operating points.

If an operator genuinely needs replacing, that is an experiment with its own record, not a code change.
