# Attribution and licensing

This repository is **MIT** (`LICENSE`). Two things make that worth spelling out.

## It is not the sibling research package, which is AGPL-3.0-or-later

An adjacent, unpublished research package (`resgate`) is licensed AGPL-3.0-or-later, deliberately
matching [Heretic](https://github.com/p-e-w/heretic), because it reimplements Heretic components
directly: the rank-1 `RowNormalization.PRE` weight-update rule, and two of its three evaluation
metrics, one of which copies a marker list verbatim.

**None of that is here.** This repository ships no weight-editing code, no abliteration, no
`RowNormalization`, and none of those metrics — it edits no weights at all (`K = 0.0`). It shares no
source file with that package; the `ResGate*` aliases in `src/cybergate_plugin/` are this project's
own pre-rename API names, kept for backwards compatibility, and the pre-rename plugin was never
AGPL-licensed.

## The one component with upstream provenance

`scripts/fit/fit_refusal.py` projects the refusal direction onto the orthogonal complement of the
harmless centroid:

```python
r_unit = unit(raw - <raw, gd> gd)
```

* The **method** is published research: Arditi et al. 2024, *Refusal in Language Models Is Mediated
  by a Single Direction* (arXiv:2406.11717).
* Heretic implements it as `orthogonalize_direction`. That implementation is what this one was
  checked against, and it is credited in the fitting script and in `kits/gemma4/MANIFEST.json`.
* The code here is an independent NumPy implementation of that projection, written for this
  repository. No Heretic source was copied into it.

**The bound, stated plainly:** the sibling package's own attribution notes record that whether
reimplementing a Heretic *component* with credit "would have counted as sufficient separation under
a permissive licence was never settled" — it chose AGPL so the question would not arise. Choosing
MIT here reopens that question for this one projection. The judgement is that a three-line vector
projection from a published paper is not a derivative of any particular implementation of it, and
that the components which actually forced the AGPL choice are absent. That is a judgement, not a
legal opinion, and it is recorded here rather than left implicit so it can be revisited.

## Other provenance

* `tests/equivalence/baseline/` vendors this project's **own** prior implementation, pinned by
  sha256, so the refactor's behaviour-preservation claim can be tested rather than asserted.
* **No model weights are distributed here.** Every checkpoint named in this repository is fetched by
  you from its own publisher, under that publisher's licence.
* The fit and evaluation corpora are not in this repository (`docs/EVALUATOR_QUICKSTART.md` §4).
