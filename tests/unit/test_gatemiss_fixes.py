"""Unit tests for the four GUI gate-miss fixes:
1. Disjunct 3 non-target-harm pivot floor (`s_cat_standalone < in_context_pivot_floor`),
2. Middle/late layer category rescue (`_apply_mid_late_category_rescue`),
3. Chunked prefill reassembly (`update_chunk_state_on_embed`),
4. APC prefix-cached multi-turn decision publishing (`publish_decision` -> `pc_ctx_<user_sha>`).
"""
from __future__ import annotations

from typing import Any

from cybergate.decision.rules import is_nontarget_harm_pivot
from cybergate.families import family_by_name
from cybergate.hooks._common import publish_decision, update_chunk_state_on_embed
from cybergate.hooks.refusal_monitor import _apply_mid_late_category_rescue
from cybergate.state.memory import MemoryDecisionStore


def test_disjunct3_paraphrased_cyber_not_revoked_when_above_pivot_floor() -> None:
    """A paraphrased cyber turn in [in_context_pivot_floor, thresh) with healthy last-token score is not revoked."""
    # Exact gatemiss.png Turn 3/4 numbers on Gemma-4:
    # s_cat_standalone = 0.081968 (>= 0.0750 pivot floor, < 0.108077 tau), st_ref_z = 8.34
    assert not is_nontarget_harm_pivot(
        s_cat_standalone=0.081968,
        s_cat_in_context=0.165521,
        s_cat_last=0.055865,
        st_cat_last=0.055865,
        s_ref_z=8.34,
        st_ref_z=8.34,
        thresh=0.10807704,
        revoke_ceil=0.0520,
        in_context_pivot_floor=0.0750,
        s_cat_last_pivot_floor=0.0350,
        st_ref_z_pivot=7.0,
    )
    # Non-target harm pivot below in_context_pivot_floor still revokes via Disjunct 3:
    assert is_nontarget_harm_pivot(
        s_cat_standalone=0.0680,
        s_cat_in_context=0.1100,
        s_cat_last=0.0500,
        st_cat_last=0.0500,
        s_ref_z=8.50,
        st_ref_z=8.50,
        thresh=0.10807704,
        revoke_ceil=0.0520,
        in_context_pivot_floor=0.0750,
        s_cat_last_pivot_floor=0.0350,
        st_ref_z_pivot=7.0,
    )


def test_mid_late_category_rescue_gemma4_and_qwen36() -> None:
    """Middle/late layer consensus rescues paraphrased cyber turns while rejecting benign and non-target harm."""
    gemma = family_by_name("gemma4")
    qwen = family_by_name("qwen36")

    gemma_thresholds = {
        "L10_last": {"R3": 0.0444, "Lk16": 0.0467, "R2": 0.0558},
        "L11_mean_user": {"R3": 0.0857, "Lk16": 0.0824, "R2": 0.1036},
        "L11_last": {"R3": 0.0546, "Lk16": 0.0568, "R2": 0.0609},
        "L16_last": {"R3": 0.0830, "Lk16": 0.0796, "R2": 0.0947},
        "L21_last": {"R3": 0.0395, "Lk16": 0.0417, "R2": 0.0456},
    }
    gemma_scores = {
        "L10_last": 0.0559,
        "L11_mean_user": 0.0897,
        "L11_last": 0.0725,
        "L16_last": 0.0972,
        "L21_last": 0.0762,
    }

    dec_gemma: dict[str, Any] = {
        "fired": False,
        "score": 0.081968,
        "s_cat_mean_user": 0.081968,
        "s_cat_last": 0.055865,
        "threshold": 0.10807704,
        "source": "live_vllm_gpu_L10_mean_user_hook",
    }
    _apply_mid_late_category_rescue(dec_gemma, 8.34, gemma_scores, gemma_thresholds, gemma)
    assert dec_gemma["fired"] is True
    assert dec_gemma["rescued_by_mid_late"] is True
    assert dec_gemma["raw_early_score"] == 0.081968
    assert dec_gemma["score"] > dec_gemma["threshold"]

    # Low refusal pressure (benign prompt) must NOT be rescued:
    dec_benign: dict[str, Any] = {
        "fired": False,
        "score": 0.081968,
        "s_cat_mean_user": 0.081968,
        "s_cat_last": 0.055865,
        "threshold": 0.10807704,
        "source": "live_vllm_gpu_L10_mean_user_hook",
    }
    _apply_mid_late_category_rescue(dec_benign, 0.45, gemma_scores, gemma_thresholds, gemma)
    assert dec_benign["fired"] is False

    # Non-target harm (low s_cat_last) must NOT be rescued even if refusal is high:
    dec_nontarget: dict[str, Any] = {
        "fired": False,
        "score": 0.0680,
        "s_cat_mean_user": 0.0680,
        "s_cat_last": -0.0031,
        "threshold": 0.10807704,
        "source": "live_vllm_gpu_L10_mean_user_hook",
    }
    _apply_mid_late_category_rescue(dec_nontarget, 9.20, gemma_scores, gemma_thresholds, gemma)
    assert dec_nontarget["fired"] is False

    # Early-only hits (L10..L12 without L16..L21 consensus) must NOT rescue non-target harm:
    dec_early_only: dict[str, Any] = {
        "fired": False,
        "score": 0.072428,
        "s_cat_mean_user": 0.072428,
        "s_cat_last": 0.051209,
        "threshold": 0.10807704,
        "source": "live_vllm_gpu_L10_mean_user_hook",
    }
    gemma_early_only_scores = {
        "L10_last": 0.0559,
        "L11_mean_user": 0.0897,
        "L11_last": 0.0725,
        "L16_last": 0.0500,
        "L21_last": 0.0200,
    }
    _apply_mid_late_category_rescue(
        dec_early_only, 12.11, gemma_early_only_scores, gemma_thresholds, gemma
    )
    assert dec_early_only["fired"] is False

    # Follow-up fragment with modest refusal z (< 4.0 on Gemma-4) must NOT be rescued standalone:
    dec_fragment: dict[str, Any] = {
        "fired": False,
        "score": 0.096390,
        "s_cat_mean_user": 0.096390,
        "s_cat_last": 0.069896,
        "threshold": 0.10807704,
        "source": "live_vllm_gpu_L10_mean_user_hook",
    }
    _apply_mid_late_category_rescue(dec_fragment, 2.43, gemma_scores, gemma_thresholds, gemma)
    assert dec_fragment["fired"] is False

    # Qwen-3.6 paraphrased cyber rescue:
    qwen_thresholds = {
        "L10_last": {"R3": 0.0817, "Lk16": 0.0956, "R2": 0.0897},
        "L18_last": {"R3": 0.0631, "Lk16": 0.0655, "R2": 0.0690},
        "L19_last": {"R3": 0.0402, "Lk16": 0.0432, "R2": 0.0468},
    }
    qwen_scores = {
        "L10_last": 0.1020,
        "L18_last": 0.0715,
        "L19_last": 0.0520,
    }
    dec_qwen: dict[str, Any] = {
        "fired": False,
        "score": 0.058838,
        "s_cat_mean_user": 0.058838,
        "s_cat_last": 0.042420,
        "threshold": 0.07104209,
        "source": "live_vllm_gpu_L15_mean_user_hook",
    }
    _apply_mid_late_category_rescue(dec_qwen, 4.10, qwen_scores, qwen_thresholds, qwen)
    assert dec_qwen["fired"] is True
    assert dec_qwen["rescued_by_mid_late"] is True
    assert dec_qwen["score"] > dec_qwen["threshold"]

    # Middle/late veto on lexical mean_user false positive (e.g. "show me files in current directory")
    # where L16_mean_user >= R3 but all 3 middle/late last-token probes (L16_last, L20_last, L21_last) < R3:
    gemma_l16_thresholds = {
        "L16_mean_user": {"R3": 0.0828, "Lk16": 0.0783, "R2": 0.0942},
        "L16_last": {"R3": 0.0830, "Lk16": 0.0796, "R2": 0.0947},
        "L20_mean_user": {"R3": 0.0713, "Lk16": 0.0705, "R2": 0.0741},
        "L20_last": {"R3": 0.0334, "Lk16": 0.0343, "R2": 0.0391},
        "L21_mean_user": {"R3": 0.0769, "Lk16": 0.0803, "R2": 0.0810},
        "L21_last": {"R3": 0.0395, "Lk16": 0.0417, "R2": 0.0456},
    }
    show_files_scores = {
        "L16_mean_user": 0.1295,
        "L16_last": 0.0718,
        "L20_mean_user": 0.0928,
        "L20_last": 0.0321,
        "L21_mean_user": 0.0987,
        "L21_last": 0.0384,
    }
    dec_show_files: dict[str, Any] = {
        "fired": True,
        "layer": 16,
        "score": 0.1295,
        "s_cat_mean_user": 0.1295,
        "s_cat_last": 0.0718,
        "threshold": 0.08284618,
        "source": "live_vllm_gpu_L16_mean_user_hook",
    }
    _apply_mid_late_category_rescue(
        dec_show_files, 4.78, show_files_scores, gemma_l16_thresholds, gemma
    )
    assert dec_show_files["fired"] is False
    assert dec_show_files["suppressed_by_mid_late"] is True
    assert dec_show_files["score"] < dec_show_files["threshold"]

    # Also veto when L16_last is elevated by an OpenCode <tool_call> system prompt
    # while both authoritative late-last probes (L20_last, L21_last) are below R3:
    opencode_tool_scores = {
        "L16_mean_user": 0.12037,
        "L16_last": 0.096916,
        "L20_mean_user": 0.105398,
        "L20_last": -0.010313,
        "L21_mean_user": 0.120392,
        "L21_last": 0.00085,
    }
    dec_opencode_tool: dict[str, Any] = {
        "fired": True,
        "layer": 16,
        "score": 0.12037,
        "s_cat_mean_user": 0.12037,
        "s_cat_last": 0.096916,
        "threshold": 0.08284618,
        "source": "live_vllm_gpu_L16_mean_user_hook",
    }
    _apply_mid_late_category_rescue(
        dec_opencode_tool, 4.58, opencode_tool_scores, gemma_l16_thresholds, gemma
    )
    assert dec_opencode_tool["fired"] is False
    assert dec_opencode_tool["suppressed_by_mid_late"] is True
    assert dec_opencode_tool["score"] < dec_opencode_tool["threshold"]
    # Preserve raw L16_last so multi-turn follow-up code turns are not falsely flagged as Disjunct 4 pivots
    assert dec_opencode_tool["s_cat_last"] == 0.096916

    # Multi-turn in-context decision (num_user_turns > 1) must NEVER be vetoed:
    multiturn_scores = {
        "L16_mean_user": 0.10875,
        "L16_last": 0.0776,
        "L20_mean_user": 0.0696,
        "L20_last": -0.0235,
        "L21_mean_user": 0.0785,
        "L21_last": -0.0186,
    }
    dec_multiturn: dict[str, Any] = {
        "fired": True,
        "layer": 16,
        "num_user_turns": 3,
        "score": 0.10875,
        "s_cat_mean_user": 0.10875,
        "s_cat_last": 0.0776,
        "threshold": 0.08284618,
        "source": "live_vllm_gpu_L16_mean_user_hook",
    }
    _apply_mid_late_category_rescue(
        dec_multiturn, 3.635, multiturn_scores, gemma_l16_thresholds, gemma
    )
    assert dec_multiturn["fired"] is True
    assert dec_multiturn.get("suppressed_by_mid_late") is None


class _FakeIdsTensor:
    def __init__(self, ids: list[int]) -> None:
        self._ids = list(ids)

    def numel(self) -> int:
        return len(self._ids)

    def reshape(self, *_args: int) -> _FakeIdsTensor:
        return self

    def cpu(self) -> _FakeIdsTensor:
        return self

    def tolist(self) -> list[int]:
        return list(self._ids)


def test_chunked_prefill_reassembly_across_two_chunks() -> None:
    """A 2496-token Chunk 1 with unclosed user turn + 12-token Chunk 2 transitions start -> complete."""
    gemma = family_by_name("gemma4")
    # Build a 2508-token 3-turn Gemma-4 sequence split at 2496:
    user_open = [105, 2364, 107]
    model_open = [105, 4368, 107]
    turn_end = [106]
    # Turn 1 + model reply + Turn 2 + long model reply = 2480 tokens
    prefix = (
        [2]
        + user_open + [7001] * 16 + turn_end
        + model_open + [8001] * 485 + turn_end
        + user_open + [7002] * 7 + turn_end
        + model_open + [8002] * 1955 + turn_end
    )
    assert len(prefix) == 2480
    # Turn 3: user_open (3) + 20 payload tokens + turn_end (1) + suffix (4) = 28 tokens -> total 2508
    turn3 = user_open + [7003] * 20 + turn_end + model_open + [109]
    full_seq = prefix + turn3
    assert len(full_seq) == 2508

    chunk1 = full_seq[:2496]
    chunk2 = full_seq[2496:]
    assert len(chunk1) == 2496
    assert len(chunk2) == 12

    state: dict[str, Any] = {"last_input_ids": _FakeIdsTensor(chunk1)}
    update_chunk_state_on_embed(state, gemma)
    assert state["chunk_mode"] == "start"
    assert state["chunk_full_ids"] == chunk1

    # Simulate category_probe saving chunk_ids_buf after Chunk 1:
    state["chunk_ids_buf"] = state["chunk_full_ids"]

    # Chunk 2 arrives:
    state["last_input_ids"] = _FakeIdsTensor(chunk2)
    update_chunk_state_on_embed(state, gemma)
    assert state["chunk_mode"] == "complete"
    assert state["chunk_full_ids"] == full_seq


def test_apc_prefix_cached_multiturn_publish_decision() -> None:
    """When APC hides prior turns so conv_sha == user_sha on an in-context pass, pc_ctx_<user_sha> is written."""
    store = MemoryDecisionStore()
    cache: dict[str, dict[str, Any]] = {}
    decision = {
        "is_baseline_prefill": True,
        "is_multiturn_ctx": True,
        "conv_sha256": "a3075e892c",
        "user_sha256": "a3075e892c",
        "score": 0.165521,
        "fired": True,
        "s_ref_z": 8.34,
        "ts": 1000.0,
    }
    publish_decision(decision, store, cache)
    assert store.get("pc_ctx_a3075e892c") == decision
    assert store.get("last_multiturn_ctx") == decision
    # And R11 invariant: bare user_sha and st_<user_sha> are NOT overwritten by a multi-turn prefill
    assert store.get("a3075e892c") is None
    assert store.get("st_a3075e892c") is None
