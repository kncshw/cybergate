"""`ModelFamily` -- the common code, as a class.

Everything that differs between Gemma-4 and Qwen-3.6 is a class attribute or an
overridable method here; the algorithms that consume them are written once. Each concrete family
lives in its own folder under `families/<name>/` and sets only its own values.

The canonical package expresses the same variation as `if model_family.startswith("gemma"): ... else:
...` in nine separate places (span location, batch segmentation, tokenizer discovery, refusal layer,
refusal calibration, default offsets, revoke ceiling, hysteresis floor, prompt template). Those nine
branches are the nine attribute groups below, which is why a family object is the right shape: adding
a family stops meaning "find every `startswith` and add an arm".

Standard library only. No torch, no vllm, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..boilerplate import (
    BOTTOM_COMPLIANCE_LOCK,
    FAKE_THINK_RUBRIC,
    INLINE_COT_SUPPRESSION,
    INLINE_FINAL_STEP_CLEARANCE,
    OPEN_THINK_PREFIX,
    POST_THINK_OPENER,
)
from ..messages import extract_system_and_user

__all__ = ["MidLateHits", "MidLateVeto", "ModelFamily", "PROMPT_MODES"]


@dataclass(frozen=True)
class MidLateHits:
    """Which middle/late candidate probes crossed which threshold, for one request.

    Computed once by the refusal-monitor hook from `ModelFamily.mid_late_probes` and handed to the
    family's own veto and rescue predicates. Shared because the counting is identical for every
    family; only what you conclude from the counts differs.
    """

    r3: tuple[str, ...] = ()
    strict: tuple[str, ...] = ()
    r2: tuple[str, ...] = ()
    last_r3: tuple[str, ...] = ()


@dataclass(frozen=True)
class MidLateVeto:
    """A family's decision to veto a fired gate on middle/late probe disagreement.

    `floor` is the lower bound the dampened score is clamped to. The dampening is how the veto is
    *implemented*, not decoration: `records.finalize_live_gpu` recomputes
    `raw_gate_fired = score >= threshold` from it, and `policy`'s prior-unlock scan reads it as
    `prior_score` on later turns.
    """

    floor: float
    source_suffix: str
    hits: tuple[str, ...] = ()
    #: Set only by a family that wants the reason recorded and logged. `None` stays quiet.
    reason: str | None = None
    #: Family-specific tail for the veto log line, e.g. which hit lists did not save the decision.
    #: Only emitted when `reason` is set.
    log_detail: str = ""

PROMPT_MODES = (
    "baseline",
    "preclosed",
    "preclosed_no_opener",
    "sandwich_step1",
    "sandwich_step2",
    "closed_loop_step1",
    "closed_loop_step1b",
    "closed_loop_step2",
    "closed_loop_cl2_step2",
)

# Modes that prepend the `<|think|>` system marker (Gemma-4 only; see `THINK_SYSTEM_PREFIX`).
_OPEN_THINK_MODES = ("sandwich_step1", "closed_loop_step1", "closed_loop_step1b")


class ModelFamily:
    """Per-architecture serving and probing parameters, plus the algorithms that use them.

    Subclasses are stateless; one shared instance per family is held in `families.registry`.
    """

    # --- identity ---------------------------------------------------------------------------------
    name: str = "base"
    #: Prefixes of the `model_key` recorded in an `ops/*.json` `meta` block. The canonical hook
    #: dispatches on `op.model_key.startswith(...)`, and the shipped ops carry `gemma` and
    #: `qwen36`.
    model_key_prefixes: tuple[str, ...] = ()
    #: Substrings of an OpenAI `model` field that select this family on the serving path.
    model_id_substrings: tuple[str, ...] = ()
    #: False when this family's serving parameters below are inherited rather than measured. See the
    #: family's own module docstring for what is and is not verified.
    verified_live: bool = False

    # --- 1. chat-template token markers (used for token-span location) ----------------------------
    #: Token ids that open a user turn, e.g. `<|turn>user\n` on Gemma-4.
    user_turn_marker: tuple[int, ...] = ()
    #: Token id that closes any turn, e.g. `<turn|>` on Gemma-4.
    turn_end_id: int = -1
    #: Label reported as `span_method` when the markers above matched.
    span_label: str = "exact_token_markers(unknown)"

    # --- 2. fixed-offset fallback when the markers are absent -------------------------------------
    default_pre: int = 15
    default_suf: int = 9

    # --- 2b. which prefill is the observation -----------------------------------------------------
    #: Suffix lengths that identify the **observation** prefill, for a family whose
    #: `is_baseline_prefill` tests exact membership. Empty means the family uses the base
    #: `suffix_len <= max_suffix` rule instead.
    #:
    #: Measure this from `build_raw_prompt(messages, mode="baseline")`, NOT from the checkpoint's
    #: `apply_chat_template`. `middleware._probe_observations` renders the observation itself with
    #: that call and dispatches it to `/v1/completions`, so the client's own chat render never
    #: reaches this predicate as an observation -- it arrives later as the answer-generation
    #: prefill, which must be dropped.
    #:
    #: Do not derive it from `default_suf`. That constant is the fixed-offset span fallback and
    #: nothing else; coupling the two means an edit to one silently changes the other, and because
    #: the exact markers normally match, the fallback never fires and never surfaces the mistake.
    #:
    #: **Both ways of getting it wrong are silent.** Too narrow and `publish_decision` drops the
    #: observation: no record, no dump row, no log line, while the server answers normally. Too
    #: wide and a later prefill overwrites the published decision and appends a second unlabelled
    #: fp16 row to any running dump. `notebooks/render_preflight.ipynb` measures both directions
    #: against a real tokenizer without a GPU.
    baseline_suffix_lens: tuple[int, ...] = ()

    # --- 3. packed-batch segmentation -------------------------------------------------------------
    #: Token id that begins a sequence, used to split a packed prefill batch when vLLM's
    #: `query_start_loc` is unavailable. `None` disables BOS segmentation for the family.
    bos_token_id: int | None = None

    #: Token ids that, at index 0, mark the start of a fresh prompt rather than a continuation
    #: chunk. Consulted only when `bos_token_id` is absent from position 0. Empty means a family
    #: relies on its BOS alone.
    fresh_prompt_marker: tuple[int, ...] = ()

    #: Token index where the user turn starts in a minimal single-turn prompt with no system turn.
    #: Only consulted by the multi-turn heuristic, and only for a family that uses a BOS token --
    #: see `looks_like_in_context_prefill`.
    single_turn_user_start: int = 4

    #: A lone user span starting later than this is taken to be a prefix-cached multi-turn prefill:
    #: the history was cached away, so the hook sees only the final turn but at a late offset.
    in_context_user_start_threshold: int = 16

    # --- 4. refusal monitor -----------------------------------------------------------------------
    ref_layer: int = 28
    #: Calibrated harmless-cosine mean and sd at `ref_layer`. Overridden by an `ops/*refusal_dir.json`
    #: `meta.mu_hm_cos` / `meta.sd_hm_cos` when one is loaded.
    #: Decode-domain refusal calibration: mean and sd of the layer-`ref_layer` cosine over
    #: **mid-sentence decode steps** on harmless prompts. `None` until measured for a family, and
    #: while it is `None` the closed-loop trigger stands down instead of comparing prefill-domain
    #: sigma against `tau_spike`. `ref_mu_hm`/`ref_sd_hm` above are the *prefill-end* calibration and
    #: are not interchangeable with these -- that conflation was the producer review's Finding 1.
    ref_mu_decode: float | None = None
    ref_sd_decode: float | None = None

    ref_mu_hm: float = 0.0
    ref_sd_hm: float = 1.0

    # --- 5. policy thresholds ---------------------------------------------------------------------
    #: Cross-domain non-target-harm revocation ceiling (`tau_revoke`).
    revoke_ceil: float = 0.0380
    #: In-context category score floor in Step 2A non-target-harm pivot check.
    in_context_pivot_floor: float = 0.0750
    #: Last-token category score floor in Step 2A non-target-harm pivot check.
    s_cat_last_pivot_floor: float = 0.0350
    #: Fallback hysteresis floor used only when live refusal telemetry is absent.
    hysteresis_thresh: float = 0.0500
    #: Fallback category threshold used when no operator file is present on disk.
    category_threshold: float = 0.0

    # --- 6. operator defaults and kit binding -----------------------------------------------------
    #: Preferred `ops/` filenames, most preferred first. The first that exists is used.
    default_op_names: tuple[str, ...] = ()
    default_ref_dir_name: str | None = None
    default_bundle_name: str | None = None

    #: Name of the `kits/<kit_name>/` directory holding this family's shipped artifacts, or `None`
    #: for a family that ships none. A kit directory *is* an ops directory -- the filenames keep
    #: their prefixes -- so pointing `CYBERGATE_OPS_DIR` at one resolves through the ladders above
    #: with no special case.
    kit_name: str | None = None

    #: Hidden dimension and decoder-layer count of the checkpoint this family's artifacts were
    #: fitted against. `0` means "not measured for this family" and stands the check down.
    #:
    #: These exist because nothing bound an operator to a checkpoint. A direction fitted on a
    #: different checkpoint of the same architecture loads without complaint and simply stops
    #: firing -- that is not hypothetical, it is how the shipped Qwen operator came to clear its own
    #: threshold on 18 of its own 100 fitted training positives. Dimension and layer count catch the
    #: cross-architecture case here; the same-architecture case needs a live canary, which is
    #: `kits/<name>/canaries.json` and is checked by `scripts/serving/check_eval.sh`.
    hidden_size: int = 0
    num_layers: int = 0

    #: Key this family's rows are stored under in a `precomputed_l8_scores.json`, if one is supplied
    #: via `CYBERGATE_OPS_DIR` (none ships). A family with no
    #: rows there simply misses every lookup and the resolver falls through to the next source, so a
    #: new family needs no entry -- but it must not silently borrow another family's.
    precomputed_tag: str = ""

    #: `(gate_layer, pool)` the shipped operator must declare. `None` stands the check down.
    #:
    #: Each family's module docstring already named a gate layer, in prose, and for both live
    #: families that prose named a layer we do not ship. Making the claim a checked field is the
    #: point: a stale default ladder now fails loudly instead of serving the wrong operator.
    expected_gate: tuple[int, str] | None = None

    # --- 7. worker tokenizer discovery ------------------------------------------------------------
    #: Globs searched inside the model-worker container for a `tokenizer.json`.
    tokenizer_globs: tuple[str, ...] = ()
    #: Literal directories searched after the globs.
    tokenizer_dirs: tuple[str, ...] = ()

    # --- 8. chat template ------------------------------------------------------------------------
    turn_open: str = ""
    turn_close: str = ""
    assistant_role: str = "assistant"
    supports_bos: bool = False
    bos_text: str = ""
    think_system_prefix: str = ""

    # --- 9. thought channel ----------------------------------------------------------------------
    thought_open: str = ""
    thought_close: str = ""
    #: The empty thought block emitted in `baseline` mode. Spelled out per family because Gemma-4
    #: uses a single newline (`<|channel>thought\n<channel|>`) while the `<think>` families use two
    #: (`<think>\n\n</think>\n\n`), so it is not `thought_open + thought_close`.
    empty_thought_block: str = ""
    #: The bare closing tag, used as an HTTP `stop` string while the model is generating CoT.
    thought_close_tag: str = ""
    stop_tokens: tuple[str, ...] = ()
    # --- Server-injected think scaffold ---------------------------------------------------------
    # Every family declares all six itself -- they are in `test_families.SHARED_ATTRIBUTES` -- because
    # a model's think tokens are a property of its own training, so a string that steers one
    # checkpoint can degenerate another. Qwen3.8 is the measured case: it pads a self-regenerating
    # list on the `- Step 1: ` stem that Gemma-4 fills once and moves on from. The defaults below are
    # therefore the shipped text, not a claim that any particular family was measured on it.
    #
    # `strip_injected_think_boilerplate` matches these by exact content, so it takes the family and
    # reads them from there; a family whose text diverges without that would keep its boilerplate in
    # `reasoning_content`.

    #: Prefilled after `thought_open` on the step-1 modes, handing the thought to the model.
    open_think_prefix: str = OPEN_THINK_PREFIX
    #: The entire closed thought on the pre-closed modes, where the model contributes no CoT tokens.
    fake_think_rubric: str = FAKE_THINK_RUBRIC
    #: Force-emitted as the first visible tokens once the thought closes. Suppressed on a
    #: tool-bearing turn, whose answer cannot be a tool call if it is already committed to prose.
    post_think_opener: str = POST_THINK_OPENER
    #: Injected mid-decode when the CoT drifts toward refusal. Ends in a dangling `- Next Step: `
    #: slot of the same shape Qwen3.8 degenerates on -- live on Gemma-4, inert on Qwen3.8.
    inline_cot_suppression: str = INLINE_COT_SUPPRESSION
    #: Closes a CL2 thought: declares the derivation done and demands the answer start immediately.
    inline_final_step_clearance: str = INLINE_FINAL_STEP_CLEARANCE
    #: Closes a step-2 thought with the final compliance assertion, just before the close tag.
    bottom_compliance_lock: str = BOTTOM_COMPLIANCE_LOCK
    #: Default CoT token budget before the controller force-closes `<think>`. `0` means uncapped
    #: (let the model finish its own natural `<think>` derivation and intercept `</think>`).
    default_thought_budget: int = 180

    # --- 10. middle/late category probe consensus -------------------------------------------------
    #: Candidate layers the read-only bundle telemetry hooks attach to. Empty means no bundle probes
    #: and therefore no middle/late consensus -- which is the safe default for a family that has not
    #: measured one.
    bundle_probe_layers: tuple[int, ...] = ()

    #: Candidate probe keys (`L<n>_<pool>`) the middle/late veto and rescue consider. Empty disables
    #: both, which is why `_apply_mid_late_category_rescue` needs no family arm.
    mid_late_probes: tuple[str, ...] = ()

    def operating_point_values(self, gate_layer: int) -> tuple[float, float, float, float]:
        """`(revoke_ceil, hysteresis_thresh, in_context_pivot_floor, s_cat_last_pivot_floor)`.

        A method rather than four attributes because at least one family's operating point depends on
        where it is gated -- see `Gemma4Family`. The default returns the four class attributes, so a
        family whose operating point is layer-independent overrides nothing.
        """
        return (
            self.revoke_ceil,
            self.hysteresis_thresh,
            self.in_context_pivot_floor,
            self.s_cat_last_pivot_floor,
        )

    def mid_late_veto(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> MidLateVeto | None:
        """Veto a *fired* decision when the middle/late probes disagree with it, or `None` to allow.

        Returning `None` by default is deliberate. The thresholds and probe counts these predicates
        compare against were fitted per model, so a family that has not fitted its own must not
        inherit another's -- it gets no veto instead, and the raw threshold decision stands. That is
        the opposite of what the previous `if gemma4 / elif qwen36 / else` structure did, where the
        trailing `else` silently handed any new family Qwen's tuning.
        """
        return None

    def mid_late_rescue(
        self,
        decision: dict[str, Any],
        s_z: float,
        bundle_scores: dict[str, float],
        hits: MidLateHits,
        thresh_val: float,
        raw_early: float,
    ) -> bool:
        """True to rescue a *non-fired* decision on middle/late probe agreement.

        May adjust `decision` first -- some families damp or lift intermediate scores before deciding.
        Stamping `score`/`fired`/`source` for an accepted rescue is the caller's job, so that part
        stays identical across families. Default `False`, for the same reason as `mid_late_veto`.
        """
        return False

    # =============================================================================================
    # Token-span location
    # =============================================================================================
    def locate_all_user_spans(
        self, seq_ids: Sequence[int]
    ) -> tuple[list[tuple[int, int]], str]:
        """Locate every user-turn token span `[(start, end), ...]` in `seq_ids`.

        Returns `(spans, span_method)`. `span_method` names which rule produced the spans, and the
        policy path records it, so the three fallbacks must stay distinguishable.
        """
        n = len(seq_ids)
        spans: list[tuple[int, int]] = []
        marker = self.user_turn_marker

        if marker:
            m_len = len(marker)
            i = 0
            while i <= n - (m_len + 1):
                if all(seq_ids[i + k] == marker[k] for k in range(m_len)):
                    u_start = i + m_len
                    u_end = -1
                    for j in range(u_start, n):
                        if seq_ids[j] == self.turn_end_id:
                            u_end = j
                            break
                    if u_end > u_start:
                        spans.append((u_start, u_end))
                        i = u_end + 1
                        continue
                i += 1
            if spans:
                return spans, self.span_label

            cached = self._prefix_cached_span(seq_ids)
            if cached is not None:
                return cached

        if n > (self.default_pre + self.default_suf + 1):
            return (
                [(self.default_pre, n - self.default_suf)],
                f"default_offsets({self.default_pre}:-{self.default_suf})",
            )
        return [(0, n)], "full_sequence_fallback"

    def _prefix_cached_span(
        self, seq_ids: Sequence[int]
    ) -> tuple[list[tuple[int, int]], str] | None:
        """Recover a user span when the opening marker fell inside a cached prefix block.

        Only Gemma-4 implements this in the canonical package; every other family returns `None` and
        falls through to the fixed-offset rule.
        """
        return None

    def looks_like_in_context_prefill(
        self,
        user_spans: Sequence[tuple[int, int]],
        seq_ids: Sequence[int] | None,
    ) -> bool:
        """True when this prefill carries conversation history, not just one standalone turn.

        It decides which cache key the decision is published under, so getting it wrong makes the
        gate inert: a single turn misfiled as in-context is written only to the `ctx_` key, which the
        single-turn lookup never reads.

        Three signals, any one sufficient:

        1. more than one user span -- decisive, and family-independent;
        2. a lone span starting later than `in_context_user_start_threshold` -- the prefix-cache case,
           where the history was cached away and only the final turn is visible, at a late offset;
        3. for a family that uses a BOS token, a sequence that does not begin with it and whose span
           does not start where a minimal single-turn prompt would put it.

        Signal 3 is the canonical package's third disjunct, and it is **guarded on the family
        actually having a BOS token**. Unguarded it compared against Gemma-4's `<bos>` (2) and
        Gemma-4's offset (4) for every architecture, so every non-Gemma prefill matched it and was
        misfiled -- confirmed live on Qwen-3.6, whose record store held only `ctx_` keys while the
        gate reported nothing. Defect D3.
        """
        if len(user_spans) > 1:
            return True
        u_start = user_spans[-1][0] if user_spans else 0
        if u_start > self.in_context_user_start_threshold:
            return True
        if self.bos_token_id is None or not seq_ids:
            return False
        return bool(
            seq_ids[0] != self.bos_token_id and u_start != self.single_turn_user_start
        )

    def locate_latest_user_span(self, seq_ids: Sequence[int]) -> tuple[int, int, str]:
        """Locate the latest user-turn span as `(start, end, span_method)`."""
        spans, method = self.locate_all_user_spans(seq_ids)
        u_start, u_end = spans[-1]
        return u_start, u_end, method

    def is_baseline_prefill(
        self,
        u1_start: int,
        suffix_len: int,
        span_method: str,
        max_suffix: int = 12,
    ) -> bool:
        """True when a prefill is a baseline (Pass-1) observation rather than a Pass-2 route prefill."""
        del u1_start
        return bool(suffix_len <= max_suffix or span_method == "full_sequence_fallback")

    def segment_by_bos(self, input_ids_1d: Sequence[int], n_tok: int) -> list[tuple[str, int, int]] | None:
        """Split a packed prefill batch on BOS boundaries, or return `None` if not applicable.

        Consecutive BOS tokens count as one boundary. Returns `None` unless at least two sequences
        were found, so the caller can fall back to treating the batch as one sequence.
        """
        if self.bos_token_id is None:
            return None
        if len(input_ids_1d) != n_tok:
            return None
        bos_positions = [
            i
            for i, tid in enumerate(input_ids_1d)
            if tid == self.bos_token_id and (i == 0 or input_ids_1d[i - 1] != self.bos_token_id)
        ]
        if len(bos_positions) < 2:
            return None
        bounds = bos_positions + [n_tok]
        return [(f"bos_seq_{b}", bounds[b], bounds[b + 1]) for b in range(len(bos_positions))]

    # =============================================================================================
    # Chat template
    # =============================================================================================
    def build_raw_prompt(
        self,
        messages: Sequence[dict[str, Any]],
        mode: str = "baseline",
        custom_think_text: str | None = None,
        post_think_opener: str | None = None,
        include_bos: bool = False,
    ) -> str:
        """Construct the exact raw prompt string for `/v1/completions`.

        Raises `ValueError` on an unknown `mode`.
        """
        sys_prompt, _, turns = extract_system_and_user(messages)
        opener = self.post_think_opener if post_think_opener is None else post_think_opener
        rubric = self.fake_think_rubric if custom_think_text is None else custom_think_text

        bos = self.bos_text if (include_bos and self.supports_bos) else ""
        think_prefix = self.think_system_prefix if mode in _OPEN_THINK_MODES else ""

        parts = [f"{bos}{self.turn_open}system\n{think_prefix}{sys_prompt}{self.turn_close}\n"]
        for t in turns:
            role = self.assistant_role if t["role"] == "assistant" else t["role"]
            parts.append(f"{self.turn_open}{role}\n{t['content'].strip()}{self.turn_close}\n")
        parts.append(f"{self.turn_open}{self.assistant_role}\n")
        parts.append(self._thought_block(mode, rubric, custom_think_text, opener))
        return "".join(parts)

    def _thought_block(
        self,
        mode: str,
        rubric: str,
        custom_think_text: str | None,
        opener: str,
    ) -> str:
        """Render the trailing thought block for `mode`."""
        t_open = self.thought_open
        t_close = self.thought_close
        cot_body = custom_think_text or ""
        prefix = self.open_think_prefix

        if mode == "baseline":
            return self.empty_thought_block
        if mode == "preclosed_no_opener":
            return f"{t_open}{rubric}{t_close}"
        if mode == "preclosed":
            return f"{t_open}{rubric}{t_close}{opener}"
        if mode in ("sandwich_step1", "closed_loop_step1"):
            return f"{t_open}{prefix}"
        if mode == "closed_loop_step1b":
            return f"{t_open}{prefix}{cot_body}{self.inline_cot_suppression}"
        if mode in ("sandwich_step2", "closed_loop_step2"):
            return f"{t_open}{prefix}{cot_body}{self.bottom_compliance_lock}{t_close}{opener}"
        if mode == "closed_loop_cl2_step2":
            return f"{t_open}{prefix}{cot_body}{self.inline_final_step_clearance}{t_close}"
        raise ValueError(f"Unknown mode: {mode!r}")

    # =============================================================================================
    # Matching
    # =============================================================================================
    @classmethod
    def matches_model_id(cls, model_id: str) -> bool:
        """True when `model_id` names this family on the serving path."""
        m = (model_id or "").lower()
        return any(s in m for s in cls.model_id_substrings)

    @classmethod
    def matches_model_key(cls, model_key: str) -> bool:
        """True when an operator's `meta.model_key` selects this family on the worker path."""
        k = (model_key or "").lower()
        return any(k.startswith(p) for p in cls.model_key_prefixes)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} verified_live={self.verified_live}>"
