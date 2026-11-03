# Fitting a kit from scratch

`serve_eval.sh` consumes a kit. This is how a kit's **category direction** is produced, which is
the only step that needs the weights and a GPU.

Both scripts here need `numpy`, which the plugin itself does not -- `pyproject.toml` deliberately
keeps the serving package stdlib-plus-vLLM, so fitting deps are not declared there. Capturing
activations additionally needs `torch` and `transformers`, imported only on that path: with
`--acts-cache` pointing at an existing `.npz`, `numpy` alone is enough.

```bash
pip install numpy                        # solve only, from a cached capture
pip install numpy torch transformers     # to capture activations
```

```bash
python3 scripts/fit/fit_category.py \
  --model google/gemma-4-26B-A4B-it --family gemma4 \
  --layer 16 --pool mean_user --render fit \
  --pos  pos_train_cyber_100.json \
  --neg  neg_ref_14cats_140.json \
  --out  kits/gemma4/refit_L16.json \
  --acts-cache acts_gemma4_L16.npz \
  --verify-against kits/gemma4/gemma4_L16_M1.json
```

`--acts-cache` is worth always passing: the capture is the slow part (a model load plus one forward
pass per prompt) and the solve is seconds, so a cached `.npz` makes re-fitting at a different `k`
or threshold rule free.

## Inputs

| flag | what it needs |
|---|---|
| `--pos` | the target-category prompts. 100 `cyber_offense` rows for the shipped kits. |
| `--neg` | the non-target prompts, **each carrying a `cat`** -- 140 rows across 14 subcategories. The subcategory is not decoration: M1 centres each subcategory on its own mean, and that is the whole difference from the earlier `B0` arm. |
| `--ref` | the threshold-calibration set. Defaults to `--neg`, which is what R3 uses. |

Rows may be `{"text": ..., "cat": ...}` or `{"prompt": ...}`; a bare list of strings works for
`--pos`, which needs no subcategory.

**The corpora are not in this repository.** Request them separately; see
`docs/EVALUATOR_QUICKSTART.md` §4.

## The method, and two ways to get it wrong

`M1_subcat_Tridge_lam16`:

1. pool layer `L`'s output over the user span -- `acts[l] = hidden_states[l+1]`
2. **unit-normalise every row**, then centre: positives on their own mean, each negative
   subcategory on its own mean
3. `S = (Zp'Zp + Zn'Zn) / (n_pos - 1)`, eigendecompose
4. `alpha = lambda_16`, the eigenvalue just past `k=15` -- that is what "lam16" names
5. `c = V (Lambda + alpha I)^-1 V' d` over the **whole spectrum**, for `d = mean(pos) - mean(neg)`
6. unit-normalise; `tau` = R3 = the 12th-largest reference score (`m_budget = 11`)

Two mistakes cost hours on 2026-11-01 and both reproduce a plausible-looking direction that is
quietly wrong:

* **fitting on raw hidden states instead of unit rows** -> cosine 0.54 to the shipped direction.
  The gate scores `cos(h, c)`, so the fit belongs in the same geometry.
* **reading "Tridge" as truncating the whitening to the top `k`** -> cosine 0.53. `k` selects the
  ridge constant only; whitening inside the top-15 subspace throws away most of `d`.

Neither shows up as an error. Both land near the right `cos_to_unit_d`, because a direction can sit
0.82 from `d` and still be far from the one you want. Always pass `--verify-against`.

## `--render`: the fit render is not the serving render

| mode | prompt | span |
|---|---|---|
| `fit` *(default)* | `tokenizer.apply_chat_template` with the system prompt `"You are a helpful assistant."`, `add_special_tokens=False` | the user's own tokens, by exact token-count arithmetic |
| `serving` | `family.build_raw_prompt(mode="baseline")` -- CyberGate's own turn markers and thought block | `family.locate_latest_user_span` |

The shipped operators were fitted under `fit`, so that is the default and the only mode that
reproduces them. The two renders genuinely disagree -- one gemma4 prompt scores +0.047, +0.167 or
+0.186 depending on the render -- so `meta.fit_render` records which was used.

That the threshold still behaves at serve time is luck rather than design: R3 is computed from the
same activations as the direction, so the fit is internally consistent in whatever render it uses,
and the shipped `tau` happens to land in the right place in the serving render too. Worth
re-measuring on any new model rather than assumed.

## Verified reproduction

Re-fitting the shipped gemma4 operator from its own 240 prompts, 2026-11-01, bf16 on a GB10:

| quantity | reproduced | shipped | delta |
|---|---|---|---|
| cosine to the shipped direction | **0.99945700** | 1.0 | 0.05% |
| `alpha` (lambda_16) | 0.000967956474741 | 0.0009657231624378617 | 0.23% |
| R3 threshold | +0.082738917947366 | +0.08284617644833335 | 0.13% |
| `cos_to_unit_d` | 0.8422136351 | 0.8325451067258752 | 1.2% |
| fit positives firing | 100/100 | 100/100 | exact |
| reference prompts firing | 12/140 | 12 (measured live) | exact |

The residuals are bf16 capture noise. The original capture names a pinned snapshot
(`7d4c97e5`); this run used whatever `google/gemma-4-26B-A4B-it` resolves to in the local cache, so
**bit-identical agreement was not expected and has not been demonstrated** -- 0.99946 in 2816
dimensions is a reproduction, not a coincidence, but it is not proof the snapshots match.

## Still needed for a complete kit

`fit_category.py` produces the **category** factor. A kit also needs:

* `MANIFEST.json` and `canaries.json`. The canaries are what catch an operator fitted on a
  different checkpoint of the same architecture -- see `scripts/serving/check_eval.sh`.


# The refusal direction: `fit_refusal.py`

```bash
python3 scripts/fit/fit_refusal.py \
  --model <model> --family <family> --layer 21 --pool last \
  --harmful harmful.json --harmless harmless.json \
  --out kits/<kit>/<kit>_L21_refusal_dir.json
```

Method, recovered 2026-11-01 from the code that fitted the shipped artifacts
(`profile_ab.py:143-150`; `direction.py:83-93` for the 5-prompt variant -- both in the internal
fitting pipeline, not in this repository):

```
ab     = mean(harmless)                     # harmless centroid, RAW rows
hb     = mean(harmful)
gd     = unit(ab)
raw    = unit(hb - ab)
r_unit = unit(raw - <raw, gd> gd)           # orthogonalize against the harmless centroid
sep    = ||hb - ab|| / mean_i ||g_i - ab||  # separation, in units of harmless spread
```

Two details are easy to get wrong and both are measurable. Rows are **raw** when building the
direction and unit-normalised only for the calibration, which is a cosine per row. And the
orthogonalization is **load-bearing**: without it the direction keeps a component along the
harmless centroid -- along "generic prompt" -- which pushes `mu_hm_cos` off zero, costs false
positives, and tilts the refusal axis toward the category axis. Dropping that one line yields
cosine 0.860 to the shipped gemma4 direction with `mu_hm_cos` of the **wrong sign**. The
projection is Arditi et al. 2024 (arXiv:2406.11717); Heretic implements it as
`orthogonalize_direction` (github.com/p-e-w/heretic) and that is what this was checked against --
see `ATTRIBUTION.md`, since Heretic is AGPL-3.0-or-later and this repository is MIT.

The calibration is not optional trimming. `mu_hm_cos` and `sd_hm_cos` *are* the scale the policy is
written in: `s_ref_z = (s_ref_cos - mu_hm_cos) / sd_hm_cos`. A family with no refusal direction
falls back to `mu = 0.0, sd = 1.0`, leaving `s_ref_z` a raw cosine against which the `1.50` floor is
effectively unreachable -- so the gate silently runs on its category factor alone.

## The row sets choose the variant

One function covers both shipped recipes; only the rows differ.

| variant | direction rows | shipped by | blob |
|---|---|---|---|
| full corpus | 692 derivation harmful + 554 harmless | **qwen36** | `dir_<tag>_sep.pt` |
| 5-prompt "blind" | 1 harmful + 4 harmless | **gemma4** | `gemma_dir90.pt` |

A direction can come from a handful of prompts. A sigma scale cannot, and gemma4's shipped artifact
is exactly that split: direction from the 5-prompt kit, calibration from 692 + 554. `--cal-harmful`
/ `--cal-harmless` express it; they default to the direction rows. Fewer than 2 calibration rows on
either side is refused rather than written, because `std(ddof=1)` of one row is `NaN` and a `NaN`
`sd_hm_cos` would ship silently and make every `s_ref_z` meaningless.

**Corpus conventions are part of the method.** Harmful is the **692 derivation rows only** -- the 80
`para_*` and 100 `adv_*` rows are held out, and including them shifts the calibration. Harmless is
all 554. `sd` is `ddof=1` and percentiles are linearly interpolated.

## Verified reproduction

Recomputed from the surviving `gemma_pool681_acts.pt` activation cache (snapshot `7d4c97e5`,
1426 rows, L21 `last`). No GPU, no model load.

| what | result |
|---|---|
| full-corpus recipe vs `dir_gemma4-26B_sep.pt[21.attn_out]` | **cosine 1.0000000000** |
| its `sep(21)` | 1.162652063 vs blob 1.162652 |
| orthogonality residual `<r_unit, gd>` | −2.8e-16 |
| shipped `r_unit` calibrated on 692 + 554 | every sidecar field to **≤5.6e-16**, `z_harm_mean` bit-identical |
| that operating point at `s_ref_z >= 1.50` | 28/554 harmless = **5.1%**, 692/692 derivation, 180/180 held-out |
| 5-prompt direction + 692/554 calibration | `mu_hm_cos` +0.138893 vs shipped +0.137420; `sd_hm_cos` 0.051490 vs 0.051301; `z_harm_mean` 10.594 vs 10.727 |

The shipped `+0.137` `mu_hm_cos` -- which looked impossible for a difference-of-means direction --
is explained by the split: the direction is orthogonalized against a **4-prompt** centroid while
the calibration measures against a **different 554-row** corpus, so nothing forces it to zero. The
full-corpus variant orthogonalizes against the same centroid it calibrates on and lands at
`mu_hm_cos` +0.00103 accordingly.

## What is not reproduced, and the bound

* **gemma4's shipped direction is not reproduced bit-exactly: cosine 0.9735.** The 5-prompt kit's
  own activation capture did not survive, so the 1 harmful and 4 harmless rows had to be taken from
  `pool681`, whose render and padding differ. `sep(21)` comes out 2.7849 against the blob's 2.8285.
* **The identity of the single harmful prompt is not confirmed.** `harm-critical_infrastructure-08`
  is the best match of the 11 `harm-*` rows in `pool681` at cosine 0.9735; documentation elsewhere
  names a different prompt (`harm-patient`), which is not in `pool681` under that key. Confirming it
  needs a 5-prompt GPU capture.
* Everything above is a recomputation from one cached capture, not a fresh one. It shows the recipe
  and the arithmetic are right; it does not re-measure the model.

`gemma4` keeps its shipped file either way -- every live number in this repo is keyed to that exact
vector. For a **new** model, use the full-corpus variant: it is what six models were built with,
it holds `mu_hm_cos` at ~0 so the sigma scale is clean, and on gemma it separates better
(`z_harm_mean` 14.66 vs 10.73). Pick `--layer` by the `sep` peak, which is what the original did.


# Fitting from the engine that serves: `acts_from_serving.py`

Everything above captures through `transformers` and the gate then scores under vLLM. That is an
**unstated assumption**: the two engines compute the same weights with different kernels, and they
only have to agree for the fit to transfer. It holds for gemma4 and qwen36 -- gemma4's own capture
(`capture_hf_v2.py`) is a `transformers` one and it delivers 558/675 live.

It did not hold for Qwen3.8-27B. 48 of its 64 layers are Gated DeltaNet, `transformers` ran
reference PyTorch fallbacks for `causal_conv1d` and `chunk_gated_delta_rule` while vLLM ran its own
fused kernels, and the fitted direction scored +0.20 in the capture and +0.02 to +0.05 served.
Nothing raised. See `kits/qwen38/MANIFEST.json` for what is and is not established about that.

Fitting on vLLM's own pooled vectors removes the assumption instead of testing it. The serving hook
dumps them when armed:

```bash
CYBERGATE_DUMP_ACTS_DIR=/content/acts_vllm \
CYBERGATE_DUMP_ACTS_LAYERS=39,43 \
vllm serve ...      # any loadable operator will do; it only has to attach, not work
```

`L<layer>.bin` is `mean_user` (the category factor), `L<layer>_last.bin` is the last position (the
refusal factor), one fp16 row per request, **in request order**. Replay the corpora strictly
sequentially -- one request in flight, no retries -- then:

```bash
python3 scripts/fit/acts_from_serving.py \
  --dump-dir /content/acts_vllm --layer 39 --pool mean_user --hidden 5120 \
  --split pos=100 --split neg=140 --out acts_vllm.npz

python3 scripts/fit/fit_category.py ... --acts-cache acts_vllm.npz    # numpy only, no model load
```

The row count is checked against the file: a retried or concurrent request inserts a row and
shifts every label after it, which would produce a direction that looks fine and means nothing.

## `diagnose_serving_layer.py`: before re-capturing, check the layer index

Nothing verifies that `transformers` layer `L` and vLLM layer `L` are the same layer. When they are
not, the symptoms are identical to a bad fit -- operator loads, hook attaches, span correct, scores
collapse -- and a re-capture is wasted work, because the direction is already right at a different
index. Serve with `CYBERGATE_CROSS_LAYER_DIAG=1`, which scores the shipped direction at **every**
layer, and:

```bash
python3 scripts/fit/diagnose_serving_layer.py --kit kits/qwen38 --ipc-dir "$CYBERGATE_IPC_DIR"
```

Exit `0` the question is answered (either an index to re-point, or the direction does transfer),
`1` no layer matches so the activations differ and a re-fit from the dump is the fix, `2` the
diagnostic was not armed. `kits/<kit>/diagnostic_reference.json` holds the capture-time scores it
compares against; those are a reference, not canaries -- they assert a score, not a verdict.

## `harvest_vllm_acts.py` — capture activations from the serving engine itself

Replays corpora at a live vLLM server with `CYBERGATE_DUMP_ACTS_DIR` armed, so the fit uses the
vectors the gate actually scores rather than `transformers`' version of them. Needed when a
direction fitted offline does not transfer: on qwen38 the shipped direction scored AUC **0.5893**
live at its own gate layer against **0.9974** claimed in-sample, measured on 12 of its own fit
positives against 14 of its own fit references.

Row `i` must be request `i`, so it sends one request at a time and verifies the dump grew by
exactly one row after each. On any anomaly it aborts instead of continuing — a gap mislabels every
row after it and nothing downstream can detect that. `--out` records the replay order, which is
what makes the dump interpretable.

## `sweep_layers_from_dump.py` — fit at every layer, rank OUT OF SAMPLE

Fits the category direction at each layer of a dump and ranks layers by **held-out recall** on a
split the fit never sees, with `tau` from R3 on the fit references — the same protocol that
produces the gemma4 kit's headline number. `--holdout` is required: ranking on an in-sample number
is the mistake this script exists to prevent.

It runs a positive control first. The dumped row for a prompt must reproduce the
`X<layer>_mean_user` the serving hook recorded for that same prompt, or the dump is not the
vectors the gate scores and the sweep means nothing. The in-sample AUC is still printed, beside
the held-out number, so the gap between them is visible.

`acts_from_serving.py` remains the route for feeding a dump to `fit_refusal.py`, which this does
not cover.

`--leak` is not optional in practice. Recall alone cannot tell a selective direction from one that
fires on everything, so with no leak split the sweep says its ranking is provisional. Given leak
splits it ranks by recall minus the worst leak. It also prints the standard error on held-out
recall and says plainly that taking the best of 64 layers biases that number upward -- choose the
layer on this split, then measure once on one that was never used to choose anything.

## `score_op_from_dump.py` — what does an operator we already have do on these activations?

Scores an existing operator's direction over a harvested dump at its own threshold, with the
splits read from the replay order file. No server needed; the dump already holds the vectors.

It exists because of an error worth not repeating. The first attempt at this measurement used a
prompt list typed into a notebook cell, and 9 of its 12 "fit positives" were not fit positives at
all -- they were other-harm prompts, which a working cyber direction is supposed to score low. The
resulting AUC was therefore indistinguishable from the direction being dead, and it was written
into a kit manifest as evidence before anyone checked. Reading the splits out of the order file
removes the opportunity to get this wrong.

## `score_live_from_records.py` — what did the DEPLOYED gate do?

Counts `fired` per corpus from the decision records a served kit published. This is not the same
measurement as a layer sweep: the deployed path adds the mid/late rescue, the refusal veto, and
`fired` as the worker publishes it. gemma4's headline numbers come from this measurement and 82 of
its 558 fires come from the rescue, so an offline recall is not comparable to them. Reports both
the as-recorded and the veto-adjusted basis, counts missing records rather than dropping them, and
separates rescued fires from threshold crossings.

## `write_canaries_from_records.py` — pin the verdicts a served gate produced

A canary pins a measured verdict so a later run can tell it is serving a different checkpoint, and
that only works if the numbers come from the deployed gate. `--from-target` and `--from-leak`
default to the **fit** corpora: a canary carries its prompt text, and this repository does not
redistribute CySecBench, CatQA, WMDP or SOSBench.
