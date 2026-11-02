# Verified environment

Everything in this package reads private vLLM or HuggingFace structure. These are the versions it was
verified against, on gpubox, container `vllm_cybergate_gpubox`:

```
vLLM         0.19.1rc1.dev15+g50cd5674b
model class  Gemma4ForConditionalGeneration
total_layers 30
hidden dim   2816
flags        --enforce-eager --enable-prefix-caching --max-model-len 4096
             --gpu-memory-utilization 0.88
             --middleware cybergate_plugin.middleware.CyberGateThinkMiddleware
```

## What each module depends on

| Module | Private surface | Breaks if |
|---|---|---|
| `introspect.find_transformer_layers` | `model.language_model.model.layers` and three other attribute paths | the model wrapper is restructured, or a new architecture nests the decoder differently |
| `introspect.find_embed_tokens_module` | `...model.embed_tokens`, `register_forward_pre_hook` | embedding is fused into the first decoder layer, or moved out of the module tree |
| `introspect.load_worker_tokenizer` | HF snapshot layout under `/root/.cache/huggingface/hub/models--*/snapshots/*` | the cache layout changes, or the checkpoint is mounted as a flat directory instead (which is how the checkpoints on gpubox are actually laid out -- see `tokenizer_dirs` on `Gemma4Family` for the fallback that exists because of it) |
| `batch.extract_batch_slices` | `vllm.forward_context.get_forward_context()`, `ctx.attn_metadata.query_start_loc`, `.req_ids` | the forward-context API moves, or `query_start_loc` changes units or dtype |

## Failure mode to expect

Every one of these degrades quietly rather than raising:

* `find_transformer_layers` raises, which is loud, and is the only one that does.
* `find_embed_tokens_module` returns `None`, and the gate then has no `input_ids`, so turn markers are
  never found and span location falls back to fixed offsets.
* `load_worker_tokenizer` returns `None`, and the hook then scores but publishes no decision record,
  so the serving path sees `decided=False` and abstains.
* `extract_batch_slices` falls back to treating a packed batch as one sequence, which averages
  activations across unrelated requests.

After any vLLM upgrade, check the startup log for the three attach lines before trusting a decision:

```
[CYBERGATE-EMBED-HOOK]  Attached input_ids pre-hook on ...
[CYBERGATE-L10-HOOK]    Attached read-only Layer-10 (mean_user) ... total_layers=30
[CYBERGATE-L21-HOOK]    Attached read-only Layer-21 refusal cosine monitor hook ...
```
