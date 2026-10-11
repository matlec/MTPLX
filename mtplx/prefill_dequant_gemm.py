"""Prefill linear layers on an fp16 GEMM over weights dequantized per call (M1).

On M1 Max, MLX's 4-bit ``quantized_matmul`` runs a 2,048-row prefill chunk at
7.8-7.9 TFLOPS for every Qwen3.8 projection shape; ``dequantize`` + the fp16
GEMM runs the same shapes at 8.6-8.8 (2026-10-12, attn-s2/gemm_probe*.py):
1.08-1.12x, the dequantize included (it touches each weight once per call,
~1% of a 2,048-row GEMM). MLX's fp16 GEMM is slow on a transposed view when K
is much longer than N (the MLP down projection, 5.0 TFLOPS), so those take a
contiguous K x N copy instead (8.55).

Only prefill-sized calls (at least ``MIN_ROWS`` rows) of fp16 activations
against 4-bit affine weights take this path; everything else, and every call
when ``MTPLX_PREFILL_DEQUANT_GEMM=0``, is the stock kernel. The dequantized
weight is a transient of one call (178 MB for a 17,408 x 5,120 projection).
"""

from __future__ import annotations

import os

import mlx.core as mx

MIN_ROWS = 1024
_STATE: dict[str, object] = {"installed": False, "original": None, "calls": 0}


def enabled() -> bool:
    raw = (os.environ.get("MTPLX_PREFILL_DEQUANT_GEMM") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    from .cache_state import m1_long_context_defaults

    return m1_long_context_defaults()


def install() -> dict[str, object]:
    """Wrap ``mx.quantized_matmul`` (idempotent)."""
    if _STATE["installed"]:
        return {"installed": True, "already": True}
    original = mx.quantized_matmul

    def quantized_matmul(x, w, scales=None, biases=None, transpose=True, group_size=None,
                         bits=None, mode="affine", **kwargs):
        if (
            transpose
            and biases is not None
            and (bits in (4, None))
            and mode == "affine"
            and getattr(x, "dtype", None) == mx.float16
            and x.ndim >= 2
            and int(w.shape[-1]) * 8 == int(x.shape[-1])
            and not kwargs.get("stream")
        ):
            rows = 1
            for d in x.shape[:-1]:
                rows *= int(d)
            if rows >= MIN_ROWS:
                gs = 64 if group_size is None else int(group_size)
                wd = mx.dequantize(w, scales, biases, group_size=gs, bits=4)
                n, k = int(wd.shape[0]), int(wd.shape[1])
                _STATE["calls"] = int(_STATE["calls"]) + 1
                if k > 3 * n // 2:
                    return x @ mx.contiguous(wd.T)
                return x @ wd.T
        return original(x, w, scales, biases, transpose=transpose, group_size=group_size,
                        bits=bits, mode=mode, **kwargs)

    mx.quantized_matmul = quantized_matmul
    _STATE.update(installed=True, original=original)
    return {"installed": True, "min_rows": MIN_ROWS}
