# CyberGate kit: qwen36

Everything needed to serve and verify the CyberGate category gate on **Qwen3.6-35B-A3B**, and
nothing belonging to any other model. `MANIFEST.json` is the machine-readable version of this page.

> **Read "The threshold rule" below before comparing any number here to the `gemma4` kit.** This
> kit's threshold was chosen against an evaluation corpus and that one's was not, so the headline
> leak numbers are not comparable even though they sit in the same kind of table.

## What is in here

| file | role |
|---|---|
| `qwen36_L21_M1_refit.json` | the shipped category operator — layer 21, `mean_user` pooling, τ = −0.00022390 |
| `qwen36_L28_refusal_dir.json` | refusal direction for the second gate factor, layer 28 |
| `qwen36_candidate_ops_bundle.json` | 18 candidate-layer probes, telemetry only — no decision reads them |
| `canaries.json` | 5 prompts with the verdict measured on this checkpoint — positives only, see below |
| `MANIFEST.json` | model ids, checkpoint shape, measured results, known gaps |

## Serve it

```bash
scripts/serving/serve_eval.sh --kit qwen36        # ~15 min first load, one 35B-class GPU
scripts/serving/check_eval.sh --kit qwen36        # then verify
```

`serve_eval.sh` mounts this directory as the operator directory and adds `--reasoning-parser qwen3`,
which this model needs and Gemma does not.

**Do not drop the serve flags** — `--quantization fp8_per_channel`, `--no-enable-prefix-caching`,
`--enforce-eager`, `--attention-backend TRITON_ATTN`. The second one in particular is not a
performance setting: a cached prefix skips the forward pass, so the hook never runs and the decision
is silently stale. The `gemma4` kit README has the full table.

## Why this operator was re-fit, which is the whole reason the kit binding exists

The previously shipped `ops/qwen36_L16_M1.json` direction cleared **its own threshold on 18 of its
own 100 fitted training positives** on this served checkpoint. It loaded without error, attached
without error, scored every request, and gated almost nothing. Nobody found out from a crash — it
surfaced as a recall measurement that looked too low.

That is the failure mode the canaries and the `expected_gate` / `hidden_size` / `num_layers`
declarations on `Qwen36Family` exist to catch. Metadata alone cannot catch it: dimensions and layer
both matched. Only running the operator against the live model does.

## What was measured

Live, on the served checkpoint, raw threshold crossings:

| | CySecBench 675 | CatQA 234 | fit positives 100 | fit reference 140 |
|---|---|---|---|---|
| as recorded | **651 (96.4%)** | **40 (17.1%)** | 100 | 17 |
| veto-adjusted | 596 (88.3%) | 40 | — | — |

The mid/late rescue contributes **nothing** here, unlike the `gemma4` kit where it supplies 82 of
558. Conversely the veto exposure is much larger: 55 of the 651 recorded fires sit below the
`s_ref_z` threshold of 1.50, against 5 of 558 on `gemma4`.

### The threshold rule

τ comes from **`catqa_leak_le_40`**: the lowest threshold whose CatQA leak stays at or below 40.

That rule is set **against an evaluation corpus**. So the 17.1% leak above **is the constraint that
was imposed, not a measurement of how the gate generalises.** The `gemma4` kit's 9.8% comes from R3,
which only ever sees the fit reference set, and is a genuine out-of-sample number. Putting the two
side by side without this paragraph would be misleading.

For a comparable point: **R3 at this same layer gives 621/675 CySec at 30/234 CatQA.** That is the
number to use against the `gemma4` kit.

## Known gaps

* The layer **and** the threshold were both selected against `cysec_675` and `catqa_234`. Both are
  in-sample for that choice, and there is **no held-out confirmation**.
* τ is negative (−0.00022) and sits in a dense part of the score distribution, so there is very
  little margin either side of it.
* `canaries.json` carries **positives only.** The operator's metadata records 100/100 fit positives
  firing raw, which licenses a per-prompt fire expectation for any of them. It also records 17/140
  fit references firing but not *which* 17 — so there is no honest way to assert that a particular
  reference prompt stays quiet. Closing this needs one Qwen serve to harvest `fit_ref_140` per
  prompt.
* The canary positives carry no per-prompt margins, because Qwen scores were never harvested at this
  layer and threshold. If a canary fails, harvest all 100 before concluding it is a checkpoint
  mismatch — it could be a near-threshold prompt moving under a render change.
* `wmdp_bio`, `sos_bio` and `wmdp_cyber` have not been measured for this operator at all.
