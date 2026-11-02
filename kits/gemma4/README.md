# CyberGate kit: gemma4

Everything needed to serve and verify the CyberGate category gate on **Gemma-4-26B-A4B-it**, and
nothing belonging to any other model. `MANIFEST.json` is the machine-readable version of this page.

## What is in here

| file | role |
|---|---|
| `gemma4_L16_M1.json` | the shipped category operator — layer 16, `mean_user` pooling, τ = +0.08284618 |
| `gemma4_L21_refusal_dir.json` | refusal direction for the second gate factor, layer 21 |
| `gemma4_candidate_ops_bundle.json` | 11 candidate-layer probes, telemetry only — no decision reads them |
| `canaries.json` | 9 prompts with the verdict measured on this checkpoint |
| `MANIFEST.json` | model ids, checkpoint shape, measured results, known gaps |
| `gemma4_ruleC_serve.json` | **not the default.** An alternate, see below |

## Serve it

```bash
scripts/serving/serve_eval.sh --kit gemma4        # ~15 min first load, one 26B-class GPU
scripts/serving/check_eval.sh --kit gemma4        # then verify
```

`serve_eval.sh` mounts this directory as the operator directory, so no other model's artifacts are
reachable from the container.

**Do not drop the serve flags.** Each is load-bearing, not a preference:

| flag | what happens without it |
|---|---|
| `--quantization fp8_per_channel` | 26B at bf16 loads, then OOMs |
| `--no-enable-prefix-caching` | a cached prefix skips the forward pass, the hook never runs, and the decision is silently stale |
| `--enforce-eager` | compiled graphs elide the hidden states the hooks read |
| `--attention-backend TRITON_ATTN` | the backend everything here was measured on |

## Verify it

A running server proves nothing about the gate: the hooks live in the model-worker process and fail
silently. `check_eval.sh` reads the decision record, which is the only evidence the probe ran, and
its last step replays `canaries.json`.

The canaries are the step that matters. Dimension and layer checks catch a *cross-architecture*
mismatch, but an operator fitted on a different checkpoint of the *same* architecture passes every
metadata check, attaches cleanly, and then fires on almost nothing. The canaries are the only check
that catches that, because their verdicts were measured on the checkpoint these artifacts were
fitted on.

Expected output: 5 positives fire, 4 negatives stay quiet, 9/9 reproduced.

They are replayed with `cybergate_mode=baseline` and no system prompt, because that is the render
they were measured in. The render matters more than it looks — `check_eval.sh`'s check [1] prompt
scores **+0.186274** over a 26-token prefill in that render and **+0.167149** over a 37-token
prefill in the default serving render (measured live 2026-11-01). Both clear tau, so here the
scores differ while the verdict agrees — which is why the thresholds are only ever compared in the
render they were calibrated in. This said "+0.047361 over a 30-token prefill ... above the
threshold in one and below it in the other" until 2026-11-01; that pair came from an earlier
serving render.

## What was measured

Live, on the served checkpoint, 1,149 prompts matched by exact prompt sha:

| | CySecBench 675 | CatQA 234 | fit positives 100 | fit reference 140 |
|---|---|---|---|---|
| as recorded | **558 (82.7%)** | **23 (9.8%)** | 100 | 12 |
| veto-adjusted | 553 | 23 | 100 | 12 |

Two bases are reported because the worker hook publishes `fired` *before* the refusal monitor
attaches `s_ref_z`. The record therefore captures the category gate plus the mid/late rescue, but
not the policy-level two-factor veto. `veto-adjusted` removes the fires whose `s_ref_z` is under 1.50.

Of the 558 recorded fires, **476 are raw threshold crossings and 82 come from the mid/late rescue**.
Rescued rows are stamped with a synthetic score of τ + 0.002, so a naive `score >= threshold` count
reads 482 and double-counts six of them.

### The threshold rule, which is the thing to check first

τ comes from **R3**: the 12th-largest score on the 140 fit-reference prompts, so exactly 11 of them
fire. R3 never sees an evaluation corpus. **That is why the 9.8% CatQA leak above is a real
out-of-sample measurement** rather than a constraint that was fitted.

This is not true of the `qwen36` kit, whose threshold rule is set against CatQA directly. The two
leak numbers look comparable in a table and are not.

## Known gaps

* `wmdp_bio`, `sos_bio` and `wmdp_cyber` have **not** been measured on the live serving render for
  this operator. `MANIFEST.json` carries offline-sweep numbers for them, and the live render is known
  to shift scores, so those are not a substitute.
* The mid/late rescue supplies 82 of the 558 recorded fires. The rescue is not covered by the
  threshold rule, so 14.7% of delivered recall rests on a separate mechanism.
* The 240 fit prompts are not redistributed here. `MANIFEST.json` records their md5s so a re-fit can
  be shown to have used the same corpus; the 9 prompts quoted in `canaries.json` are the exception.

## The alternate operator

`gemma4_ruleC_serve.json` adds a second category probe and fires on `L21 last` **OR** `L20 last`.
Measured live at **626/675 CySec (92.7%) at 24/234 CatQA**, against this kit's 558/23 — so +68
recall for +1 leak. It was served, measured and then rolled back; the default here is the baseline.

To serve it: `scripts/serving/serve_eval.sh --kit gemma4 --op gemma4_ruleC_serve.json`. It resolves
the candidate bundle from this directory, same as the default operator.
