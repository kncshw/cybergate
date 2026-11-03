#!/usr/bin/env python3
"""Replay corpora at a live vLLM server and harvest the pooled activations it computes.

**Why.** `fit_category.py` captures through `transformers` and the gate then scores under vLLM.
On qwen38 that assumption failed: the shipped direction, measured live on 12 of its own 100 fit
positives against 14 of its own 140 fit references, scored AUC 0.5893 at its own gate layer --
chance -- having claimed 0.9974 in-sample. Fitting on vLLM's own vectors removes the assumption
instead of testing it.

**Row order is the whole contract.** The dump hook appends one fp16 row per request per layer, so
row `i` must be request `i`. One retried, concurrent or dropped request shifts every label after
it and produces a direction that looks fine and means nothing. This sends strictly one request at
a time and verifies the dump grew by exactly one row after each; on any anomaly it aborts rather
than continuing, because nothing downstream can detect a gap.

Serve with the dump armed and unrestricted, then:

    python3 scripts/fit/harvest_vllm_acts.py \\
      --corpus pos=/content/pos_train_cyber_100.json \\
      --corpus neg=/content/neg_ref_14cats_140.json \\
      --corpus cysec_val=/content/cysecbench_val_675_gemma_llama.json \\
      --dump-dir /content/acts_vllm_fit --ipc-dir /dev/shm/cybergate_ipc_colab \\
      --model Qwen/Qwen3.8-27B --hidden 5120 --out /content/fit_order.json

The `--out` file records the replay order, which is what makes the dump interpretable at all.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

TEXT_KEYS = ("text", "prompt", "instruction", "question")


def load(path: str, label: str) -> list[dict]:
    rows = json.loads(pathlib.Path(path).read_text())
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, str):
            out.append({"text": r, "key": f"{label}-{i}", "split": label})
            continue
        text = next((r[k] for k in TEXT_KEYS if r.get(k)), None)
        if text is None:
            raise SystemExit(f"{path} row {i} has none of {TEXT_KEYS}")
        # `cat` is carried because the fit centres the negatives WITHIN subcategory. Deriving it
        # from the key instead gives 17 groups for the 14-category reference set, which changes
        # the covariance and the degrees of freedom without changing anything visible.
        out.append({"text": text, "key": r.get("key", f"{label}-{i}"), "split": label,
                    "cat": r.get("cat")})
    return out


def ask(base_url: str, model: str, prompt: str, timeout: float) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "cybergate_mode": "baseline"}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=timeout).read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable; replayed in the order given")
    ap.add_argument("--dump-dir", required=True, help="the server's CYBERGATE_DUMP_ACTS_DIR")
    ap.add_argument("--ipc-dir", required=True, help="cleared first: a record carrying s_ref_z "
                                                     "makes the worker skip the probe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--out", required=True, help="where to write the replay order")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--probe-layer", type=int, default=0,
                    help="which L<n>.bin to watch for the per-request row check")
    ap.add_argument("--wipe-dump", action="store_true",
                    help="delete every .bin in --dump-dir first. vLLM's startup profiling runs "
                         "dummy forward passes that fire the dump hook, so a freshly started "
                         "server leaves rows behind before any request is sent.")
    args = ap.parse_args()

    order, splits = [], []
    for item in args.corpus:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--corpus {item!r} must be LABEL=PATH")
        rows = load(path, label)
        order += rows
        splits.append((label, len(rows)))
    print(f"{len(order)} prompts: " + "  ".join(f"{a}={b}" for a, b in splits))

    texts = [r["text"] for r in order]
    if len(set(texts)) != len(texts):
        raise SystemExit("duplicate prompt text across corpora: the IPC records would collide "
                         "and the positive control could not be matched to a row")

    ipc = pathlib.Path(args.ipc_dir)
    ipc.mkdir(parents=True, exist_ok=True)
    gone = [p for p in ipc.glob("*.json")]
    for p in gone:
        p.unlink()
    print(f"cleared {len(gone)} IPC record(s) from {ipc}")

    dump = pathlib.Path(args.dump_dir)
    probe = dump / f"L{args.probe_layer}.bin"
    row_bytes = args.hidden * 2

    def rows() -> int:
        return probe.stat().st_size // row_bytes if probe.is_file() else 0

    def wait(target: int, timeout: float = 5.0) -> int:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if rows() >= target:
                break
            time.sleep(0.05)
        return rows()

    if args.wipe_dump:
        wiped = [p for p in dump.glob("*.bin")]
        for p in wiped:
            p.unlink()
        print(f"wiped {len(wiped)} .bin file(s) from {dump}")
    # The row check watches ONE file. If the server narrowed the dump with
    # CYBERGATE_DUMP_ACTS_LAYERS and that file is not among them, the check can never pass, so
    # say so in a second instead of after a request.
    present = sorted(q.name for q in dump.glob("*.bin"))
    if present and probe.name not in present:
        raise SystemExit(
            f"--probe-layer {args.probe_layer} watches {probe.name}, which this dump does not "
            f"contain.\nPresent: {present}\nThe watched layer must be one the server actually "
            f"dumps -- the per-request row check is what guarantees row i is request i. Pass "
            f"--probe-layer with one of the layers above.")
    if rows():
        raise SystemExit(
            f"{probe} already holds {rows()} rows. Row i must be request i, so this dump cannot "
            f"be appended to.\nUsually these are vLLM's startup profiling passes -- it runs dummy "
            f"forwards that fire the dump hook before any request arrives. Re-run with "
            f"--wipe-dump, which clears every layer together; deleting some files by hand leaves "
            f"the layers at different lengths.")

    t0 = time.time()
    for i, r in enumerate(order):
        before = rows()
        try:
            ask(args.base_url, args.model, r["text"], args.timeout)
        except Exception as exc:                                        # noqa: BLE001
            raise SystemExit(f"ABORT at request {i} ({r['key']}): {exc!r}\nRow alignment is "
                             f"unknown from here. Empty {dump} and start over; do not resume.")
        after = wait(before + 1)
        if after == before == 0:
            others = sorted(q.name for q in dump.glob("*.bin"))
            if others:
                raise SystemExit(
                    f"ABORT at request {i} ({r['key']}): the request succeeded and the dump grew, "
                    f"but not {probe.name}.\nPresent: {others}\nThe server IS dumping to this "
                    f"directory -- it is not dumping layer {args.probe_layer}. Pass --probe-layer "
                    f"with one of the layers above.")
            raise SystemExit(
                f"ABORT at request {i} ({r['key']}): the request succeeded but nothing was "
                f"written to {dump}.\n--dump-dir says where to WATCH; the server writes to "
                f"whatever "
                f"CYBERGATE_DUMP_ACTS_DIR was set to when it STARTED, and changing --dump-dir does "
                f"not move it. Check the serve log:\n    grep ACTDUMP <serve log>\nand pass that "
                f"path, or restart the server with the path you want.")
        if after != before + 1:
            raise SystemExit(f"ABORT at request {i} ({r['key']}): {probe.name} went {before} -> "
                             f"{after}, expected +1. Row i is no longer request i.")
        if (i + 1) % 25 == 0 or i + 1 == len(order):
            el = time.time() - t0
            print(f"  {i + 1:>4}/{len(order)}  {el:6.0f}s  "
                  f"eta {el / (i + 1) * (len(order) - i - 1):5.0f}s", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(
        {"hidden": args.hidden, "model": args.model, "dump_dir": str(dump),
         "splits": splits, "order": order}, indent=1))

    bad = []
    for path in sorted(dump.glob("*.bin")):
        n = path.stat().st_size / row_bytes
        if n != len(order):
            bad.append((path.name, n))
    print(f"\n{len(order)} requests in {time.time() - t0:.0f}s")
    print(f"dump files not holding exactly {len(order)} rows: {bad or 'NONE'}")
    print(f"replay order -> {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
