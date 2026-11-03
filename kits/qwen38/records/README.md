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

## `harvest_2026-11-02_layer_sweep.txt`

The full 64-layer sweep exactly as `sweep_layers_from_dump.py` printed it, including its own
positive control (the dumped rows reproduce the serving hook's recorded `X39_mean_user` to
1.16e-05 across all 3,284 rows). Every figure quoted from it in `../MANIFEST.json` was verified
against this file line by line, 0 mismatches.

It is the evidence for the finding that matters most here: **the high-recall layers are the leaky
ones.** L39 leaks 0.2104 into WMDP-bio and L25 leaks 0.2402, at recall 0.9526 and 0.9422, while
L55 reaches 0.9230 at 0.0047. The original fit chose L39 on an in-sample AUC of 0.9985, and the
in-sample AUC column in this table is 0.99+ at all 64 layers -- including layers whose held-out
recall is 0.21. It could not have distinguished any of them.

## What is still not retained

The 4.1 GB activation dump and the decision records. Both are reproducible from
`scripts/fit/harvest_vllm_acts.py`. Neither is in this repository.
