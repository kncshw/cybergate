"""Evidence gathering: **all** the I/O that feeds a decision.

Every cache read, every HTTP sub-probe and every precomputed lookup in the decision path lives here,
so that `policy.decide` can be a pure function. That split is the single highest-value outcome of the
refactor (REFACTOR_PLAN.md §1): once the I/O is on this side of the boundary, every defect found
across four audit rounds becomes a table-driven unit test.

Port of `gate_probe.evaluate_gate_for_prompt` (the resolution ladder) and the first half of
`gate_probe.evaluate_gate_for_messages` (lines 1241-1335 and the prior-turn loop at 1379-1400).
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Sequence

from ..config import DEFAULT_REF_MAINT_Z_THRESH, env_first
from ..families import ModelFamily, family_for_model_id
from ..messages import extract_user_turns
from ..ops import CyberGateOperator, load_cybergate_op
from ..ops.precomputed import PRECOMPUTED_CACHE
from ..state import keys as key_builders
from ..state.backend import DecisionStore
from ..state.request_state import (
    LIVE_GPU_PROMPT_CACHE,
    REQUEST_GATE_STATES,
)
from .inputs import GateInputs, OperatingPoint, PriorTurnEvidence
from .records import (
    optional_float,
    apply_in_context_two_factor,
    finalize_live_record,
    in_context_evidence,
    standalone_evidence,
)

__all__ = ["EvidenceGatherer", "legacy_op_for_model_id"]


def legacy_op_for_model_id(model_id: str) -> tuple[CyberGateOperator, str]:
    """Load the operator for a decision, and return `(op, precomputed_tag)`.

    **Defect D1 is fixed here.** This was a hardcoded two-way branch:

        family = "gemma" if "gemma" in model_id.lower() else "qwen36"
        op = load_cybergate_op(model_family="gemma4" if family == "gemma" else "qwen36")

    which collapsed every non-Gemma model id into the Qwen-3.6 arm. While Phi-4 and Llama-3.1 had
    family classes that was a live inconsistency: those requests were scored against **Qwen's** tau
    while the worker-side hook loaded the model's own operator via `op.model_key`. The branch
    happened to be correct once only two families remained, but it would have broken again, silently,
    on the third -- so it is now a registry lookup and the family supplies its own tag.

    `precomputed_tag` is the key this family's rows sit under in a supplied
    `precomputed_l8_scores.json` (none ships; see `ops/schema.md`). A
    family with no rows there misses every lookup and the resolver falls through, which is the
    intended behaviour for a family whose offline scores were never computed.
    """
    family = family_for_model_id(model_id)
    op = load_cybergate_op(model_family=family.name)
    return op, family.precomputed_tag


class EvidenceGatherer:
    """Resolves observations for a request. Holds the store and the operating point; owns the I/O."""

    def __init__(
        self,
        store: DecisionStore,
        op: CyberGateOperator,
        family: ModelFamily,
        *,
        family_tag: str | None = None,
        gate_url: str | None = None,
        precomputed: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.op = op
        self.family = family
        # Defaults to the family's own tag rather than to a literal "gemma".
        self.family_tag = family.precomputed_tag if family_tag is None else family_tag
        self.gate_url = gate_url
        self.precomputed = PRECOMPUTED_CACHE if precomputed is None else precomputed

    @property
    def ref_maint_z(self) -> float:
        """The refusal maintenance floor. One value for every family in the canonical package."""
        return DEFAULT_REF_MAINT_Z_THRESH

    # ---------------------------------------------------------------------------------------------
    # Operating point
    # ---------------------------------------------------------------------------------------------
    def operating_point(self) -> OperatingPoint:
        """The thresholds this request is decided against.

        The four floors come from the family, which may vary them by gate layer -- Gemma-4 does, and
        the shipped Gemma kit gates where it varies them. This used to be an
        `if self.family.name == "gemma4" and self.op.gate_layer >= 15` arm with the four values
        inline; it now asks the family, so a new family expresses its own operating point in its own
        module.
        """
        revoke_ceil, hysteresis, ic_pivot, last_pivot = self.family.operating_point_values(
            self.op.gate_layer
        )
        return OperatingPoint(
            thresh=self.op.thresh,
            gate_layer=self.op.gate_layer,
            revoke_ceil=revoke_ceil,
            hysteresis_thresh=hysteresis,
            in_context_pivot_floor=ic_pivot,
            s_cat_last_pivot_floor=last_pivot,
        )

    # ---------------------------------------------------------------------------------------------
    # Single-prompt resolution ladder
    # ---------------------------------------------------------------------------------------------
    def resolve_prompt(
        self,
        user_text: str,
        *,
        prompt_key: str | None = None,
        live_record: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one prompt's decision record, trying each source in the canonical order.

        1. an explicit live observation passed in by the caller,
        2. the `/dev/shm` record under `st_<sha>` then the bare `<sha>`,
        3. this process's in-memory decision cache,
        4. this process's per-request hook state,
        5. an external `/gate` HTTP endpoint, when `CYBERGATE_GATE_URL` is configured,
        6. the offline precomputed score table, by key, then by sha, then by normalised text,
        7. an abstain record (`decided=False`).

        Sources 1-4 get the two-factor treatment; 5 and 6 carry their own `fired`.
        """
        text_clean = user_text.strip()
        sha = key_builders.text_sha256(text_clean)
        thresh = self.op.thresh
        ref_z = self.ref_maint_z

        # 1. Explicit live observation from the caller.
        if live_record and live_record.get("decided"):
            res = finalize_live_record(live_record, thresh, ref_z)
            LIVE_GPU_PROMPT_CACHE[sha] = res
            return res

        # 2. Cross-process record.
        ipc_record = self.store.get(key_builders.standalone_turn_key(text_clean)) or self.store.get(
            key_builders.bare_turn_key(text_clean)
        )
        if ipc_record and ipc_record.get("decided"):
            res = finalize_live_record(ipc_record, thresh, ref_z)
            LIVE_GPU_PROMPT_CACHE[sha] = res
            return res

        # 3. In-process cache. Already finalized when it was stored.
        if sha in LIVE_GPU_PROMPT_CACHE:
            return dict(LIVE_GPU_PROMPT_CACHE[sha])

        # 4. Per-request hook state, single-process mode. Consumed on read.
        if (
            request_id
            and request_id in REQUEST_GATE_STATES
            and REQUEST_GATE_STATES[request_id].get("decided")
        ):
            res = finalize_live_record(REQUEST_GATE_STATES[request_id], thresh, ref_z)
            REQUEST_GATE_STATES[request_id]["decided"] = False
            LIVE_GPU_PROMPT_CACHE[sha] = res
            return res

        # 5. External gate endpoint.
        gate_url = self.gate_url or env_first("CYBERGATE_GATE_URL")
        if gate_url:
            return self._resolve_over_http(gate_url, user_text)

        # 6. Offline precomputed scores.
        precomputed = self._resolve_precomputed(prompt_key, sha, text_clean)
        if precomputed is not None:
            return precomputed

        # 7. Abstain. R14: the canonical policy lets this fall through with `score = 0.0` rather than
        # treating "no observation" as its own state.
        return {
            "decided": False,
            "fired": False,
            "score": 0.0,
            "threshold": self.op.thresh,
            "layer": self.op.gate_layer,
            "source": "no_live_gpu_hook_attached",
        }

    def _resolve_over_http(self, gate_url: str, user_text: str) -> dict[str, Any]:
        """POST to an external `/gate` endpoint. No two-factor: the endpoint's `fired` is taken."""
        req = urllib.request.Request(
            gate_url.rstrip("/") + "/gate",
            data=json.dumps({"prompt": user_text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - operator-configured URL
            gdata = json.loads(resp.read().decode("utf-8"))
        score = float(gdata.get("score", 0.0))
        threshold = float(gdata.get("threshold", self.op.thresh))
        return {
            "decided": True,
            "fired": bool(gdata.get("fired", score >= threshold)),
            "score": score,
            "threshold": threshold,
            "layer": self.op.gate_layer,
            "source": "tempx_gate_http",
        }

    def _resolve_precomputed(
        self,
        prompt_key: str | None,
        sha: str,
        text_clean: str,
    ) -> dict[str, Any] | None:
        """Look the prompt up in the offline score table. No two-factor; `fired_M1` is taken."""
        layer = self.op.gate_layer
        tables: tuple[tuple[str, Any, str], ...] = (
            ("by_key", prompt_key, f"offline_L{layer}_M1_precomputed(key={prompt_key})"),
            ("by_sha256", sha, f"offline_L{layer}_M1_precomputed(sha256={sha[:10]})"),
            ("by_text_norm", text_clean.lower(), ""),
        )
        for table_name, lookup, source in tables:
            if not lookup:
                continue
            table = self.precomputed.get(table_name, {})
            if lookup not in table:
                continue
            entry = table[lookup].get(self.family_tag)
            if not entry:
                continue
            resolved_source = source or (
                f"offline_L{layer}_M1_precomputed(id={entry.get('id')})"
            )
            return {
                "decided": True,
                "fired": bool(entry["fired_M1"]),
                "score": float(entry["s_cat_M1"]),
                "threshold": self.op.thresh,
                "layer": layer,
                "source": resolved_source,
            }
        return None

    # ---------------------------------------------------------------------------------------------
    # Conversation-level gathering
    # ---------------------------------------------------------------------------------------------
    def gather(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        live_gpu_decision: dict[str, Any] | None = None,
        standalone_user_decision: dict[str, Any] | None = None,
        enable_multi_turn_stickiness: bool = True,
        prompt_key: str | None = None,
        request_id: str | None = None,
    ) -> GateInputs:
        """Resolve every observation a decision needs and return the policy's complete input.

        Assumes `messages` contains at least one non-empty user turn; the caller handles the empty
        case, as the canonical code does.
        """
        user_turns = extract_user_turns(messages)
        n_turns = len(user_turns)
        latest_user = user_turns[-1]
        op_thresh = self.op.thresh

        latest_record = self._resolve_in_context(
            user_turns,
            live_gpu_decision=live_gpu_decision,
            prompt_key=prompt_key,
            request_id=request_id,
        )
        standalone_record = self._resolve_standalone(
            latest_user,
            n_turns=n_turns,
            standalone_user_decision=standalone_user_decision,
        )

        latest = in_context_evidence(latest_record, op_thresh, self.op.gate_layer)
        standalone = (
            standalone_evidence(standalone_record, op_thresh)
            if standalone_record is not None
            else None
        )

        priors: tuple[PriorTurnEvidence, ...] = ()
        if n_turns > 1 and enable_multi_turn_stickiness:
            priors = self._resolve_priors(user_turns)

        return GateInputs(
            op=self.operating_point(),
            n_turns=n_turns,
            latest=latest,
            standalone=standalone,
            priors=priors,
            enable_multi_turn_stickiness=enable_multi_turn_stickiness,
            raw_record=latest_record,
        )

    def _resolve_in_context(
        self,
        user_turns: Sequence[str],
        *,
        live_gpu_decision: dict[str, Any] | None,
        prompt_key: str | None,
        request_id: str | None,
    ) -> dict[str, Any]:
        """Resolve the latest turn as observed inside its conversation.

        **R11 was here.** The canonical code looked the in-context record up under
        `ctx_<sha(final turn)>` *before* the conversation-keyed record. That key carries no
        conversation component, so two conversations ending in the same user text shared one
        decision -- and because it was read first, the collision took precedence over the correct
        lookup instead of acting as a fallback. The conversation key is now the only one consulted.
        See `state.keys.legacy_in_context_key_from_sha`.
        """
        latest_user = user_turns[-1]
        record = live_gpu_decision
        if record is None or not record.get("decided"):
            if len(user_turns) > 1:
                record = self.store.get(key_builders.conversation_key(user_turns))
            else:
                record = self.store.get(
                    key_builders.conversation_key(user_turns)
                ) or self.store.get(key_builders.standalone_turn_key(latest_user))

        if record is not None and record.get("decided"):
            return apply_in_context_two_factor(record, self.op.thresh, self.ref_maint_z)

        return dict(
            self.resolve_prompt(
                latest_user,
                prompt_key=prompt_key if len(user_turns) == 1 else None,
                live_record=live_gpu_decision,
                request_id=request_id,
            )
        )

    def _resolve_standalone(
        self,
        latest_user: str,
        *,
        n_turns: int,
        standalone_user_decision: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Resolve the latest turn scored alone. `None` for a single-turn request."""
        if standalone_user_decision is not None:
            return standalone_user_decision
        if n_turns <= 1:
            return None
        record = self.store.get(key_builders.standalone_turn_key(latest_user)) or self.store.get(
            key_builders.bare_turn_key(latest_user)
        )
        if record is not None:
            return record
        fallback = self.resolve_prompt(latest_user)
        return fallback if fallback.get("decided") else None

    def _resolve_priors(self, user_turns: Sequence[str]) -> tuple[PriorTurnEvidence, ...]:
        """Resolve every earlier user turn to a fired/not-fired decision, oldest first."""
        out: list[PriorTurnEvidence] = []
        for idx, prev_text in enumerate(user_turns[:-1], start=1):
            prev_ipc = self.store.get(
                key_builders.standalone_turn_key(prev_text)
            ) or self.store.get(key_builders.bare_turn_key(prev_text))
            record = self.resolve_prompt(prev_text, live_record=prev_ipc)
            out.append(
                PriorTurnEvidence(
                    index=idx,
                    decided=bool(record.get("decided", False)),
                    fired=bool(record.get("fired", False)),
                    score=float(record.get("score", 0.0)),
                    s_ref_z=optional_float(record.get("s_ref_z")),
                )
            )
        return tuple(out)
