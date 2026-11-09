# Process topology

**The process layout is the architecture.** Getting this wrong is the origin of the whole class of
state bugs in this plugin, so read this before changing where anything lives.

## Two processes, no shared memory

`vllm serve` runs the plugin's entry point in **both** of its processes:

```
MainProcess  (pid=1)   API server + ASGI middleware
EngineCore   (pid=73)  model worker + forward hooks
```

Confirmed from the live vLLM startup log:

```
[CYBERGATE-PLUGIN] register_cybergate_plugin() invoked in process=MainProcess
[CYBERGATE-PLUGIN] register_cybergate_plugin() invoked in process=EngineCore
[CYBERGATE-L10-HOOK] Attached read-only Layer-10 (mean_user) M1_ridge_lam16 hook
                     (dim=2816, thresh=0.10807704, total_layers=30, K=0.0)
```

They are separate OS processes. **Every module global is per-process.** A score the worker computes
is invisible to the API server until it is written to `/dev/shm`.

## Which module runs where

| Module | Process | Imports torch |
|---|---|---|
| `_entrypoint` | both | no (imports `hooks` lazily, inside the loader wrapper) |
| `config`, `boilerplate`, `messages` | both | no |
| `families/*` | both | no |
| `ops/*` | both | only on the `.pt` path, and falls back to a stdlib unpickler |
| `state/keys`, `state/backend`, `state/ipc_shm`, `state/memory` | both | no |
| `state/request_state` | both, **separately** | no |
| `decision/*` | API server | no |
| `telemetry/log` | both | no |
| `serving/*` | API server | no |
| `hooks/*` | **worker only** | **yes** |
| `vllm_adapter/*` | worker only | no (torch is passed in, vllm imported lazily) |

`tests/unit/test_purity.py::test_importing_the_package_does_not_import_torch` asserts the API-server
path stays torch-free, by importing it in a subprocess and checking `sys.modules`.

## What crosses the boundary

Only decision records, only as JSON files under `/dev/shm/cybergate_ipc/`:

```
worker                                         API server
------                                         ----------
category probe scores the user span
  -> builds a decision dict
  -> publish_decision()
       writes sha_<conv>.json
              sha_st_<turn>.json
              sha_<turn>.json
              sha_ctx_<turn>.json
refusal monitor attaches s_ref_z
  -> publish_decision() again (same keys)
       writes l21_monitor.json
                                               middleware probes to force a prefill
                                               evidence layer reads the records
                                               policy decides
                                               route injects <think> or passes through
```

`state/ipc_shm.ShmDecisionStore` is the only thing that touches those files, and `state/keys` is the
only thing that names them. Nothing else may construct a key or a path.

Writes go through a pid-and-thread-unique temp file plus `Path.replace`, so a reader sees either the
old record or the new one, never a partial write. That matters because the two processes are reading
and writing concurrently.

## The probe, and why it exists

The API-server process cannot score a prompt: the probe lives in the other process, attached to a
decoder layer. So the middleware sends the prompt back through its own app as a 1-token completion,
purely to make the worker prefill it and publish a record.

`--enable-prefix-caching` makes this subtler. **A repeated prefix produces no prefill, so the hook
does not run.** That is protocol, not a bug, but it means "no fresh record" is a normal state rather
than an error. The middleware's probe skips a turn only when a record already carries `s_ref_z`,
because a record without one came from a pass the refusal monitor never saw.

When no record can be obtained at all, the evidence layer returns
`decided=False, source=no_live_gpu_hook_attached`, and the policy currently proceeds with
`score = 0.0`. That is defect R14; `Branch.NO_OBSERVATION` exists for the fix.

## Per-process state, and what it is safe for

| Global | Lives in | Safe for |
|---|---|---|
| `LAST_GATE_DECISION` | both, separately | the latest decision *in this process*; a single slot, so under continuous batching it holds whichever sequence the hook touched last |
| `REQUEST_GATE_STATES` | both, separately | per-request isolation within one process, keyed by vLLM request id |
| `LIVE_GPU_PROMPT_CACHE` | both, separately | avoiding a re-read of a record this process already resolved |

None of these are safe for anything that has to be seen by the other process. If you find yourself
wanting that, the answer is a `DecisionStore` key.

## Latency budget on the forward path

Measured on the live server:

| Metric | Value |
|---|---|
| `gpu_dot_us` | 64-70 us |
| `hook_total_us` | 85-240 us |
| first-touch spike | ~1100 us, while the direction tensor moves to the device |

An earlier revision of this plugin had a 34 ms median `hook_total_us`, and that was treated as a
defect. The hook bodies are therefore closures built once at attach time. **Do not add per-forward
dispatch, per-forward object construction, abstract base classes or registry lookups to a hook body.**
Modularise around the hook, not inside it: that is why `hooks/_common.py`, `hooks/_scoring.py` and
`hooks/bundle_telemetry.py` contain only setup helpers and the one post-decision publish step.

## Operational rules for live GPU evaluation

* One model-loading job per GPU. Never run two 26B/27B servers on the same GPU; stop any existing
  container first.
* Pre-flight `cat /proc/loadavg` and confirm idle; post-flight confirm it returned. Reference values:
  idle 0.00-0.08, HTTP-only testing peaked ~0.80.
* Send test requests strictly sequentially when measuring latency or determinism. Self-inflicted
  concurrency produced a spurious `decided=False` during the audit and cost a result.
* Truncate fed-back assistant turns and cap `max_tokens` to stay within `max_model_len`.
