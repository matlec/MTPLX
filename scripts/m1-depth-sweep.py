#!/usr/bin/env python3
"""Measure MTP depth 1..3 for one model on this checkout and pick the fastest (M1 fork).

    scripts/m1-depth-sweep.py --model ~/.mtplx/models/<pack> [--depths 1,2,3] [--apply-app]

Each depth starts ``bin/mtplx serve`` on its own port with ``--profile turbo`` and the app's and
omp's serving shape (256K context, q8 KV; ``--context-window`` / ``--kv`` change it) (turbo is the profile the
fork's M1 lanes are measured on; the app's own tune ran under sustained), sends three short
prompts with thinking on (temperature 0.6, fixed seeds, 1,500 tokens), reads the decode speed
from the server's request log and stops the server. ``--apply-app`` writes the winner into the
MTPLX app's per-model depth record (only while the app is closed: it rewrites its settings on
exit). Standard library only; run it with any python3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Explain how a B-tree insertion works, step by step, with an example.",
    "Write a short essay on the history of the printing press and its effects on science.",
    "Design a rate limiter for a public API. Discuss trade-offs between algorithms.",
)
APP_SETTINGS = Path.home() / "Library" / "Application Support" / "MTPLX" / "settings.json"


def _wait_health(port: int, proc: subprocess.Popen, timeout_s: float = 900.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError("server did not become healthy")


def _chat(port: int, prompt: str, seed: int, max_tokens: int) -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=3600).read()


def _decode_rows(port: int, since: float) -> list[dict]:
    path = Path.home() / ".mtplx" / "logs" / f"request-log-{port}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if float(row.get("logged_at_s") or 0) >= since and (row.get("completion_tokens") or 0) > 64:
            rows.append(row)
    return rows


def measure(model: str, depth: int, port: int, max_tokens: int, context: int = 262144, kv: str = "q8") -> dict:
    cmd = [
        str(ROOT / "bin" / "mtplx"), "serve", "--model", model, "--depth", str(depth),
        "--profile", "turbo", "--host", "127.0.0.1", "--port", str(port), "--no-auth", "--yes",
        "--warmup-tokens", "0", "--ssd-session-cache", "off",
        "--context-window", str(context), "--paged-kv-quantization", kv,
    ]
    log = open(Path(os.environ.get("TMPDIR", "/tmp")) / f"m1-depth-sweep-{port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_health(port, proc)
        _chat(port, "Say OK.", 1, 8)  # first-request warmup, not counted (max 8 tokens)
        since = time.time()
        for i, prompt in enumerate(PROMPTS):
            _chat(port, prompt, 11 + i, max_tokens)
        rows = _decode_rows(port, since)
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    tokens = sum(int(r["completion_tokens"]) for r in rows)
    seconds = sum(float(r.get("decode_elapsed_s") or 0) for r in rows)
    rounds = sum(int(r.get("verify_calls") or 0) for r in rows)
    return {
        "depth": depth,
        "decode_tok_s": round(tokens / seconds, 1) if seconds else None,
        "tokens_per_round": round(tokens / rounds, 2) if rounds else None,
        "requests": len(rows),
    }


def apply_app(model: str, depth: int) -> str:
    if subprocess.run(["pgrep", "-xq", "MTPLXApp"]).returncode == 0:
        return "the MTPLX app is running: quit it and rerun with --apply-app"
    if not APP_SETTINGS.exists():
        return f"no app settings at {APP_SETTINGS}"
    settings = json.loads(APP_SETTINGS.read_text())
    records = settings.get("tuned_control_records_by_model") or {}
    target = str(Path(model).expanduser().resolve()).lower()
    changed = 0
    for record in records.values():
        if str(record.get("model_id") or "").lower() == target and record.get("control_field") == "depth":
            record["control_value"] = int(depth)
            changed += 1
    if not changed:
        return "the app has no depth record for this model yet (run its tune once, or pick the depth in the app)"
    backup = APP_SETTINGS.with_name(f"settings.json.bak-depth-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(APP_SETTINGS, backup)
    APP_SETTINGS.write_text(json.dumps(settings, indent=2, ensure_ascii=False))
    return f"app depth set to {depth} in {changed} record(s); backup {backup.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="local pack directory")
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--context-window", type=int, default=262144, help="as the app and omp serve (256K)")
    ap.add_argument("--kv", default="q8", help="--paged-kv-quantization, as the app and omp serve (q8)")
    ap.add_argument("--apply-app", action="store_true", help="write the fastest depth into the MTPLX app's settings")
    a = ap.parse_args()
    model = str(Path(a.model).expanduser())
    results = []
    for depth in [int(d) for d in a.depths.split(",")]:
        print(f"[depth-sweep] depth {depth} ...", file=sys.stderr, flush=True)
        try:
            results.append(measure(model, depth, a.port, a.max_tokens, a.context_window, a.kv))
        except Exception as exc:  # a depth the runtime refuses must not stop the sweep
            results.append({"depth": depth, "error": str(exc)[:200]})
        print(json.dumps(results[-1]), flush=True)
    ok = [r for r in results if r.get("decode_tok_s")]
    if not ok:
        sys.exit("no depth produced a measurement")
    best = max(ok, key=lambda r: r["decode_tok_s"])
    print(f"fastest: depth {best['depth']} ({best['decode_tok_s']} tok/s)")
    if a.apply_app:
        print(apply_app(model, int(best["depth"])))


if __name__ == "__main__":
    main()
