"""QUARANTINE: packed-batch segmentation from vLLM's forward context.

Verbatim port of `gate_probe._extract_vllm_batch_slices`.

vLLM packs several sequences into one forward pass, so a hook sees a single 2D tensor covering all of
them. Scoring it without splitting would average activations across unrelated requests.
"""
from __future__ import annotations

from typing import Sequence

from ..families import ModelFamily

__all__ = ["extract_batch_slices"]


def _candidate_metadata(ctx: object) -> list[object]:
    """Every attention-metadata object this forward context might carry.

    **This is the fix for a measured defect.** `ForwardContext.attn_metadata` is annotated
    `dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]` -- keyed per KV-cache group,
    and wrapped in a list when micro-batching. The previous code read `query_start_loc` straight off
    it with `getattr`, which on a dict returns `None` **silently**, so segmentation never ran and
    every forward pass was reported as a single sequence.

    Measured live on 2026-10-28: 221 of 221 monitor lines reported one sequence, including during
    four-way concurrency, and `generation == published` across 5538 passes. The consequence is worse
    than a lost feature -- with the pack unsegmented, the refusal projection is taken from the last
    token of the *whole batch* and the category score is pooled across *several requests'* spans, so
    under batching the gate scores a mixture of requests. Results from this plugin are only valid
    where requests were strictly sequential.

    `query_start_loc` is a field on the metadata objects themselves (`CommonAttentionMetadata`,
    `FlashAttentionMetadata`), so unwrapping the container is all that is needed. Request *ids* are
    not reachable anywhere in this build's forward context, so slices carry positional
    `batch_seq_<n>` ids and the callers must not assume a stable request identity from them.
    """
    meta = getattr(ctx, "attn_metadata", None)
    if meta is None:
        return []
    if isinstance(meta, dict):
        return list(meta.values())
    if isinstance(meta, (list, tuple)):
        out: list[object] = []
        for item in meta:
            if isinstance(item, dict):
                out.extend(item.values())
            else:
                out.append(item)
        return out
    return [meta]


def extract_batch_slices(
    n_tok: int,
    input_ids_1d: Sequence[int] | None,
    family: ModelFamily,
) -> list[tuple[str | None, int, int]]:
    """Return per-sequence `(req_id, q_start, q_end)` ranges for a packed batch.

    Three strategies, in order:
      1. vLLM's own `attn_metadata.query_start_loc`, which is authoritative and carries request ids,
      2. the family's BOS-token boundaries, when `query_start_loc` is unavailable,
      3. treat the batch as one sequence.

    Strategy 1 is only accepted when it reports more than one sequence; with a single sequence it
    falls through, so a one-sequence batch always reports `req_id=None`. That is the canonical
    behaviour and it is why a single-request decision is keyed by text hash rather than request id.
    """
    #: The end of the only sequence, when `query_start_loc` authoritatively reported one. Used by
    #: the final fallback in place of `n_tok`, which is the *padded* width -- see below.
    sole_end: int | None = None

    try:
        from vllm.forward_context import get_forward_context  # type: ignore

        ctx = get_forward_context()
        for attn_meta in _candidate_metadata(ctx):
            qsl = getattr(attn_meta, "query_start_loc", None)
            if qsl is None or not hasattr(qsl, "tolist"):
                continue
            locs = [int(x) for x in qsl.detach().cpu().tolist()]
            if len(locs) < 2 or locs[-1] > n_tok:
                continue
            req_ids = (
                getattr(attn_meta, "req_ids", None)
                or getattr(ctx, "req_ids", None)
            )
            slices: list[tuple[str | None, int, int]] = []
            for b in range(len(locs) - 1):
                rid = str(req_ids[b]) if (req_ids and b < len(req_ids)) else f"batch_seq_{b}"
                slices.append((rid, locs[b], locs[b + 1]))
            if len(slices) > 1:
                return slices
            # One sequence. Stop here rather than reading the next KV-cache group's
            # `query_start_loc`: this loop runs inside the forward pass and each iteration costs a
            # device-to-host sync, so continuing paid for one sync per KV group on every
            # single-request pass and learned nothing new. The fallthrough below is deliberate and
            # canonical -- a one-sequence batch reports `req_id=None` -- but it can at least use the
            # true length.
            sole_end = locs[-1]
            break
    except Exception:
        pass

    if input_ids_1d is not None:
        bos_slices = family.segment_by_bos(input_ids_1d, n_tok)
        if bos_slices is not None:
            return list(bos_slices)

    # `n_tok` is the padded width of the hidden-state tensor. With `--enforce-eager` it equals the
    # real token count, which is why this never showed: the score is read at `h_2d[q_end - 1]`, so
    # under cudagraph padding the pre-fix code read a *padding* position instead of the last real
    # token. `query_start_loc`'s own answer is used when there is one.
    return [(None, 0, sole_end if sole_end is not None else n_tok)]

