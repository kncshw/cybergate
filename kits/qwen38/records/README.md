# Records

Evidence for the numbers in `../MANIFEST.json`. The manifest's `measured_live` block said
"TRANSCRIBED, not retained" because its figures had been read off a terminal and nothing here
backed them. These files are the part that can be retained.

## `../qwen38_L55_M1_vllm.json`

The candidate operator from the vLLM re-fit, and its own best evidence: `meta` carries
`holdout_recall`, `holdout_split`, `leak_splits`, `in_sample_auc`, `fit_n_pos`, `fit_n_neg`,
`fit_n_neg_cats`, `fit_source` and `serve_thresh_rule`. Those were written by the fit itself, so
they are not a transcription of anything. **It is not the kit's operator.** `MANIFEST.json` still
points at `qwen38_L39_M1_serving.json`, because this one has never been served and every number
attached to it is an offline score over harvested activations.

## `harvest_2026-11-02_corpus_digest.json`

Per-split row counts and a sha256 over each split's concatenated prompt text, in replay order.

The replay order file itself is **not** committed. It carries the full text of all 3,284 prompts,
including CySecBench, CatQA, WMDP and SOSBench rows, and this repository does not redistribute the
evaluation corpora -- see the README. The digest lets a later run prove it replayed the same
corpora in the same order without shipping them, which is the same approach `kits/gemma4` takes
with its fit prompts' md5s.

## What is still not retained

The 4.1 GB activation dump, the decision records, and the 64-layer sweep table. The dump and the
records are reproducible from `scripts/fit/harvest_vllm_acts.py`; the sweep table is reproducible
from `scripts/fit/sweep_layers_from_dump.py` over the same dump. Neither is in this repository.
