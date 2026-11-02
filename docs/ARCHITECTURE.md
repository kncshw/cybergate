# Architecture

## What the gate actually does

Two readings of one prefill, combined:

1. **Category probe** -- a linear probe (`M1_ridge_lam16`) read off a mid-early decoder layer
   (L16 on Gemma-4), scored as a cosine against a unit direction over the user turn's tokens. Call it
   `s_cat`.
2. **Refusal monitor** -- a calibrated cosine against a refusal direction at a mid-stack layer
   (L21 on Gemma-4), expressed in sigma above a harmless-prompt mean:
   `s_ref_z = (s_ref_cos - mu_hm) / sd_hm`.

The gate fires when the turn is on-topic **and** the model was building up to refuse it. The second
factor is what stops a benign technical question ("explain TCP vs UDP") from being treated as a
request the gate needs to unlock: it scores as cyber-topical but carries no refusal pressure, so
there is nothing to remove.

Across turns, an unlocked session is *maintained* while refusal pressure stays at or above the floor,
and *revoked* when the conversation pivots out of the target domain.

Read-only throughout: `K = 0.0`, no weight edits, no logits touched. When the gate fires, the
intervention is a conditional `<think>` prefill on the serving side, not a change to the model.

## The shape of the refactor

The canonical package put the gate's correctness in serving-path state, interleaved with I/O, so it
could only be tested against a live 26B model. `evaluate_gate_for_messages` was one 299-line function
mixing cache reads, HTTP sub-probes, score extraction and the policy rules. Every defect found across
four audit rounds sat in that interleaving rather than in the maths.

So the central move is one pure function:

```python
decide(inputs: GateInputs) -> GateDecision
```

No cache reads, no HTTP, no globals, no clock, no environment, no torch. `tests/unit/test_purity.py`
enforces that by inspecting the module's imports, so it cannot rot. Everything the policy may read is
reachable from `GateInputs`; if a number is not on it, the policy cannot use it.

```
            evidence.py                 policy.py              outcome.py
  /dev/shm ----------> GateInputs ----------> GateDecision ----------> payload dict
  HTTP        ALL I/O                PURE                   rendering
  precomputed
```

That split is the point. Every defect in the audit became a table row: see
`tests/unit/test_regressions.py` (R1-R14) and `tests/unit/test_r11_cross_conversation.py`, which
runs in milliseconds with no GPU.

### Module map

| Package | Responsibility |
|---|---|
| `config` | constants, env vars, paths; stdlib only |
| `boilerplate` | the injected `<think>` text, the pivot regex, the two text transforms |
| `messages` | OpenAI-format message handling |
| `families/` | per-architecture parameters, as a base class plus one folder per model |
| `ops/` | operator loading from `.json` / `.pt`; `schema.md` documents the file contract |
| `state/` | the cross-process boundary: `keys` (every key), `backend` (protocol), `ipc_shm`, `memory`, `request_state` |
| `decision/` | `inputs`, `records`, `evidence` (all I/O), `rules`, `policy` (pure), `outcome`, `api` |
| `hooks/` | the two worker-side forward hooks; the only torch in the package |
| `vllm_adapter/` | quarantined version-fragile vLLM/HF introspection |
| `serving/` | `middleware` (in-process ASGI), `client` (remote), `think_routes` (shared CoT), `asgi`, `prompts` |
| `telemetry/` | event logging |

`cybergate_plugin` is a thin drop-in facade over `cybergate`, because the live launch command and the
vLLM entry point both name it.

### Families as a class, not a branch

The canonical package expressed per-architecture variation as `if model_family.startswith("gemma")`
in nine separate places. Those nine branches are now nine attribute groups on `ModelFamily`, and each
concrete family sets only its own values in its own folder:

```
families/base.py          the common algorithms
families/gemma4/          live-verified (Gemma-4-26B-A4B-it, L16 gate + L5..L21 bundle, L21 refusal)
families/qwen36/          live-verified (Qwen3.6-35B-A3B, L21 gate + L5..L32 bundle, L28 refusal)
```

Both remaining families are served and measured. `qwen36` was also the canonical `else` arm, so the
canonical package scored an unrecognised model id with Qwen's chat template and token markers. **The
two dispatch paths no longer agree on that.** `family_for_model_id` (serving) resolves from the
configured operator and otherwise raises `UnknownModelFamily`, because that path chooses the render
and guessing it is silent; `family_for_model_key` (worker) still falls back to `qwen36`, where
`ops.kit.check_operator` has already rejected a mismatched operator on dimensions.
`tests/equivalence/test_prompts_differential.py` asserts the **divergence** from the vendored
baseline, not agreement with it.

**Phi-4 and Llama-3.1 family classes existed in earlier revisions and were removed.** Both inherited
the Qwen arm wholesale -- token markers, refusal layer, policy thresholds -- overriding only
`default_op_names`, and both were measured offline and never served. Keeping a class whose every
serving parameter belonged to a different architecture asserted a support that did not exist. Their
model ids now raise `UnknownModelFamily` on the serving path -- measured 2026-11-01, with no
`CYBERGATE_OP_PATH` set: `microsoft/phi-4` and `meta-llama/Llama-3.1-8B-Instruct` both raise. An
operator naming `phi` or `llama` as its `model_key` still lands on the Qwen arm, which is what the
canonical dispatch did with them anyway.

Each family additionally declares its **kit binding** -- `kit_name`, `hidden_size`, `num_layers`,
`expected_gate` -- which `ops/kit.py` checks at load and at hook install. See the kit section in
`README.md` for why: nothing previously bound an operator to a checkpoint, and a mismatched one
loaded, attached and silently stopped firing.

## Stated limitations

### The gate reads no authorization signal

**This is the one a refactor cannot fix, and it must not be inherited as solved.**

The gate decides on cyber-topicality combined with refusal pressure. It reads nothing about who is
asking, or whether they are authorized. The "authorized pentest" framing that appears in the prompt
boilerplate and in the existing test suites therefore **does no work in the decision** -- it is text
sent to the model, not an input to the gate.

What that means concretely: the same conversation from any caller produces the same decision. There
is no tenant, no role, no scope, no engagement identifier anywhere in `GateInputs`. Whether that is
acceptable is a design question about what the mechanism is supposed to key on, and it needs the
user's and the producer's input. It is not a module boundary.

### Refusal calibration and multi-layer category bundles

Both `Gemma-4-26B-A4B-it` and `Qwen3.6-35B-A3B` ship calibrated refusal directions and multi-layer
category operator bundles, each inside its own kit:

* **Gemma-4 (`kits/gemma4/gemma4_L21_refusal_dir.json` & `kits/gemma4/gemma4_candidate_ops_bundle.json`):**
  * Refusal monitor at L21: prefill `mu_hm = +0.137421`, `sd_hm = 0.051301`; decode `mu_decode = +0.085139`, `sd_decode = 0.043947`.
  * Primary gate at **L16** (`mean_user`, `tau = +0.08284618`, threshold rule R3 with `m_budget=11`), plus 11 candidate layers in the bundle (`L5-L12, L16, L20, L21`, both `mean_user` and `last` pooling). The middle/late subset of those is what the category consensus rescue reads (`_apply_mid_late_category_rescue` in `hooks/refusal_monitor.py`); `ops/schema.md` records how many hooks actually attach.
  * This said "L10 (`tau = 0.10807704`)" until 2026-11-01. L10 is the **superseded** operator that a stale `default_op_names` ladder silently resolved to until 2026-10-31; it is no longer shipped, and loading one now raises `KitMismatch`.
* **Qwen-3.6 (`kits/qwen36/qwen36_L28_refusal_dir.json` & `kits/qwen36/qwen36_candidate_ops_bundle.json`):**
  * Refusal monitor at L28: prefill `mu_hm = +0.001382`, `sd_hm = 0.047725`; decode `mu_decode = +0.017064`, `sd_decode = 0.032047`.
  * Primary gate at **L21** (`qwen36_L21_M1_refit.json`, `mean_user`, `tau = -0.00022390`, threshold rule `catqa_leak_le_40` -- which is **in-sample**), plus 18 candidate layers in the bundle (`L5, L8-L13, L15, L16, L18-L20, L22, L24, L26, L28, L30, L32`, both poolings) for the same consensus rescue.
  * This said "L15 (`tau = 0.07104209`)" until 2026-11-01. L15 led the default ladder until 2026-10-31 and is no longer shipped. The "18 of its own 100 fitted training positives" failure belongs to a **different** operator -- `qwen36_L16_M1.json`, the one that was actually shipped before the re-fit (`kits/qwen36/MANIFEST.json` `fit.refit_note`, and `meta.deployed_baseline_for_comparison` in the refit operator).
* Both families carry a measured decode-domain refusal calibration (`ref_mu_decode`/`ref_sd_decode`). These are **not** interchangeable with the prefill pair `ref_mu_hm`/`ref_sd_hm`; conflating them was the producer review's Finding 1.

### Method questions that are out of scope here

Named so they are not mistaken for settled:

* the operating point was selected by argmax over an 88-point grid scored on one corpus;
* `mean_user` is the weaker of the two pooling modes on the measurements available;
* the shipped threshold is a quantile of the same prompt set it is evaluated against.

These are experiment-design issues. Changing any of them would invalidate the equivalence corpus, so
the refactor leaves the maths, the operator files and the thresholds exactly as they are.

## Known defects, carried forward deliberately

Nothing in this list is fixed. Phase 6 fixes them one commit at a time, failing test first. Each has a
test that asserts *current* behaviour plus a strict-xfail test asserting the desired behaviour, so a
fix flips exactly its own test and the rest of the corpus stays green.

| Id | Defect | Where | Test |
|---|---|---|---|
| **R6/R7** | One intervening turn scoring below `revoke_ceil` discards a legitimate unlock, and the no-unlock exit returns before Step 2B is consulted, so the next turn is denied at 4x the maintenance floor | `rules.prior_unlock_discarded` + the scan in `policy.decide` | `test_regressions.py::test_r6_r7_*` |
| **R11** | The in-context record is keyed `ctx_<sha(final turn)>` with no conversation component, and is read *before* the conversation-keyed record, so a decision transfers between conversations that share only their last user turn | `state.keys.in_context_key`, `evidence._resolve_in_context` | `test_keys.py`, `test_r11_cross_conversation.py` |
| **R10** | The first two pivot disjuncts test score magnitude with no guard on intent, so an ordinary pleasantry is reported as a non-target-harm pivot | `rules.is_nontarget_harm_pivot` | `test_regressions.py::test_r10_*` |
| **R12** | `raw_gate_fired` is computed from the in-context observation on every branch, so it can contradict the score the branch actually used | `records.apply_in_context_two_factor` | `test_regressions.py::test_r12_*` |
| **R13** | The `NO_PRIOR_UNLOCK` branch copies a standalone sub-probe's `source=` verbatim, so a 3-turn request reports `turns=1`. The payload's `user_turn_index` is correct; only the string lies | `policy._nested_standalone_source` | `test_regressions.py::test_r13_*` |
| **R14** | "No fresh observation" is not a typed state: an undecided record falls through to the ordinary score comparison with `0.0` | `policy.decide` | `test_regressions.py::test_r14_*` |
| **IPC TTL** | Records never expire and are never consumed on read. The audit left 310 in place; an earlier round saw 272 with the oldest 6 hours old | `state.ipc_shm` | none yet |

### Two defects found during this refactor

Neither is in the plan's list. Both are latent rather than live, and both are **carried forward
unfixed** for the same reason as the rest.

**D1 -- the decision path loads the wrong operator for a non-Gemma, non-Qwen model.**
`evidence.legacy_op_for_model_id` reproduces the canonical collapse:

```python
family = "gemma" if "gemma" in model_id.lower() else "qwen36"
op = load_cybergate_op(model_family="gemma4" if family == "gemma" else "qwen36")
```

This was a live inconsistency while Phi-4 and Llama-3.1 had classes: the policy path scored them
against Qwen's tau while the worker hook loaded their own operators via `op.model_key`, and the two
disagreed. **Both classes are now removed, so the collapse is exactly correct for every family that
exists** -- there are two, and the predicate distinguishes them.

It is recorded rather than deleted because the shape of the bug survives the fix: the predicate is
a hardcoded two-way branch, not a lookup through `family_for_model_id`. Adding a third family
reintroduces D1 immediately and silently. The kit binding check now limits the damage -- a third
family's operator would be rejected on dimensions rather than scored against Qwen's threshold -- but
it does not make the predicate correct.

**D2 -- `Branch.NEURAL_REVOKED` is unreachable.**
Reaching Step 2C requires no refusal signal at all. Surviving Step 2A requires the in-context score to
be at or above the in-context pivot floor of `0.0750`. But `0.0750` already exceeds both families'
hysteresis floors (Gemma `0.0650`, Qwen `0.0500`), so `hysteresis_holds` is necessarily true and the
revocation below it can never run. The branch is dead code in both families.
`test_differential.py::test_every_branch_is_covered` excludes it and says why.

## Proving the refactor preserved behaviour

The canonical implementation is vendored, pinned by sha256, at `tests/equivalence/baseline/`, and the
suite drives it in lockstep with the new code. First-run-green is not evidence, so each differential
suite was checked for sensitivity by mutation.

| Suite | What it compares | Cases |
|---|---|---|
| `test_differential.py` | the full decision payload, key by key, on the same `/dev/shm` records | 48 |
| `test_prompts_differential.py` | raw prompts as exact strings, 2 models x 9 modes x 8 message shapes x 2 bos settings (288 combinations), plus both text transforms | 336 |
| `test_spans_differential.py` | token-span location and BOS batch segmentation on synthetic token sequences | 42 |
| `test_think_routes_differential.py` | the exact sequence of completion calls and payloads, the visible content, the reasoning and the telemetry | 38 |

Mutation checks run against `test_differential.py`, each reverted afterwards:

| Mutation | Result |
|---|---|
| maintenance `>=` becomes `>` | 1 failure |
| two-factor `<` becomes `<=` | 1 failure |
| revoke ceiling `<` becomes `<=` | 1 failure |
| prior-unlock discard disabled (i.e. "fixing" R6) | 4 failures |
| one character changed in a rendered `source=` string | 48 failures |

The route differential caught two real bugs this refactor introduced, both now fixed:

* the shared route hardcoded the middleware's absolute `/v1/completions`, which on the remote client
  (whose `base_url` already ends in `/v1`) would have produced `/v1/v1/completions`;
* the shared route always emitted the post-think opener, which is correct for the middleware but
  wrong for the client's `include_opener=False`.

Those are exactly the failures that de-duplicating two drifted copies risks, which is why the proof
was written before the de-duplication was trusted.

### Not covered by tests

Stated plainly, because a green suite should not be read as more than it is:

* **the hook bodies.** `hooks/category_probe.py` and `hooks/refusal_monitor.py` need torch and a
  loaded model. They are ported and import-checked and pyflakes-clean, but not executed. A `NameError`
  introduced during a late refactor of these files was caught by grep and pyflakes, not by a test --
  take that as the measure of the coverage gap.
* **the ASGI middleware end to end.** `serving/asgi.py` and the middleware's probe sequencing are not
  driven by a test. The routes they call are.
* **hook latency.** The 64-70 us figure is the audit's measurement of the canonical code. This port
  has not been benchmarked, so plan step 3's latency check is outstanding.
