"""Inkling 2 bring-up bench: decode/prefill speed and logit quality.

  python scripts/inkling2_bench.py MODEL [--decode 128] [--prefill 4096]
      [--save-ref ref.npz | --ref ref.npz]

``--save-ref`` stores the model's log-probabilities on a fixed evaluation
text; ``--ref`` compares against such a file (KL, top-1 agreement, NLL), so a
converted pack can be judged against the source checkpoint without holding
both in memory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtplx.models.inkling2.load import load_model  # noqa: E402

EVAL_FILES = ("docs/architecture.md", "mtplx/sampling.py")
EVAL_TOKENS = 1024


def _tokenizer(path: Path):
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"))
    template = path / "chat_template.jinja"
    if template.is_file():
        tok.chat_template = template.read_text()
    return tok


def _eval_ids(tok) -> list[int]:
    ids: list[int] = []
    for name in EVAL_FILES:
        ids += tok.encode((ROOT / name).read_text())[: EVAL_TOKENS // len(EVAL_FILES)]
    return ids


def _logprobs(model, ids: list[int], unpadded: int) -> mx.array:
    cache = model.make_cache()
    out = []
    for start in range(0, len(ids), 256):
        logits = model(mx.array([ids[start : start + 256]]), cache=cache)
        lp = logits[0, :, :unpadded].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        mx.eval(lp)
        out.append(lp)
    return mx.concatenate(out, axis=0)


def quality(model, tok, config, args) -> None:
    unpadded = int(config["text_config"].get("unpadded_vocab_size", 0)) or None
    ids = _eval_ids(tok)
    lp = _logprobs(model, ids, unpadded)
    targets = mx.array(ids[1:])
    nll = -mx.take_along_axis(lp[:-1], targets[:, None], axis=-1).mean()
    print(f"eval tokens={len(ids)} nll={nll.item():.4f} ppl={np.exp(nll.item()):.3f}")
    if args.save_ref:
        np.savez_compressed(args.save_ref, ids=np.array(ids), lp=np.array(lp.astype(mx.float16)))
        print(f"saved reference -> {args.save_ref}")
    if args.ref:
        ref = np.load(args.ref)
        if list(ref["ids"]) != ids:
            raise SystemExit("reference was recorded on different eval ids")
        rlp = mx.array(ref["lp"]).astype(mx.float32)
        kl = (mx.exp(rlp) * (rlp - lp)).sum(axis=-1)
        top1 = (mx.argmax(rlp, -1) == mx.argmax(lp, -1)).astype(mx.float32).mean()
        print(
            f"vs ref: KL mean={kl.mean().item():.5f} p99={np.percentile(np.array(kl), 99):.4f} "
            f"max={kl.max().item():.4f} top1={top1.item():.4f}"
        )


def speed(model, tok, config, args) -> None:
    prompt = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) if tok.chat_template else args.prompt
    ids = tok.encode(text, add_special_tokens=False)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    out = []
    # Warm the decode graph before timing.
    for _ in range(4):
        logits = model(token[:, None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        out.append(token.item())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.decode):
        logits = model(token[:, None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(token)
        out.append(token)
    out = [t if isinstance(t, int) else t.item() for t in out]
    dt = time.perf_counter() - t0
    print(f"decode {args.decode} tok in {dt:.2f}s = {args.decode / dt:.2f} tok/s")
    print("sample:", repr(tok.decode(out)[:600]))
    if args.prefill:
        filler = tok.encode((ROOT / "README.upstream.md").read_text())
        pids = (filler * (args.prefill // len(filler) + 1))[: args.prefill]
        cache = model.make_cache()
        mx.synchronize()
        t0 = time.perf_counter()
        for start in range(0, len(pids), args.chunk):
            logits = model(mx.array([pids[start : start + args.chunk]]), cache=cache)
            mx.eval(logits[:, -1])
        dt = time.perf_counter() - t0
        print(f"prefill {len(pids)} tok in {dt:.2f}s = {len(pids) / dt:.1f} tok/s")
        mx.synchronize()
        t0 = time.perf_counter()
        token = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(32):
            logits = model(token[:, None], cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.async_eval(token)
        token.item()
        dt = time.perf_counter() - t0
        print(f"decode@{len(pids)} = {32 / dt:.2f} tok/s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--prefill", type=int, default=0)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--prompt", default="Write a Python function that merges two sorted lists, then explain it briefly.")
    p.add_argument("--save-ref")
    p.add_argument("--ref")
    p.add_argument("--no-speed", action="store_true")
    p.add_argument("--quality", action="store_true")
    args = p.parse_args()
    path = Path(args.model)
    t0 = time.perf_counter()
    model, config = load_model(path)
    print(f"loaded in {time.perf_counter() - t0:.1f}s, active mem {mx.get_active_memory() / 2**30:.2f} GiB")
    tok = _tokenizer(path)
    if args.quality or args.save_ref or args.ref:
        quality(model, tok, config, args)
    if not args.no_speed:
        speed(model, tok, config, args)
    print(f"peak mem {mx.get_peak_memory() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
