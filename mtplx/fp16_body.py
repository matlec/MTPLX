"""Serve a bf16 checkpoint's floating parameters as fp16 on the M1 GPU family.

M1/M2 GPUs have no native bf16 arithmetic, and the fork's prefill lanes (the
MFA attention for cached prefixes, the mma8 verify lane, the packed decode
lanes) are fp16-only. A bf16 MLX pack therefore prefilled at 60-75% of its
fp16 twin's speed (orcarouter/Qwen3.8-27B-Uncensored-MLX, M1 Max: 153 tok/s
at 21K and 107 at 85K, against the README's fp16 numbers).

The fp16 packs the fork was measured on are exactly this conversion done
offline ("convert bf16 floating tensors to fp16; preserve packed
integer/quantized tensors", Youssofal's MTPLX_FP16_CONVERSION_MANIFEST.json;
``mtplx forge --dtype fp16``). Here it runs once at load, after MTP injection
and before the kernel lanes install: every bf16 parameter becomes fp16, packed
weights and fp32 parameters stay as they are, and activations follow the
embedding's dtype. bf16 -> fp16 is exact for every value in fp16's normal
range (fp16 has the wider mantissa); a value past 65504 would overflow, and
the cast refuses a pack that has one.

On by default on the M1 family for the Qwen3.5 / 3.6 / 3.8 / Qwen3-Next
families (the fork's fp16 packs; this checkpoint's residual stream peaked at
~950 over a 20K prompt, same top-5 as bf16, KL 4e-4). ``MTPLX_M1_FP16_BODY``:
"1" forces it for any model on any GPU, "0" keeps the checkpoint's dtype.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

FP16_MAX = 65504.0
_STATE: dict[str, Any] = {"applied": False}


def _qwen_family(config: dict[str, Any] | None) -> bool:
    config = config or {}
    text = config.get("text_config") or {}
    types = {str(text.get("model_type") or "").lower(), str(config.get("model_type") or "").lower()}
    return any(t.startswith(("qwen3_5", "qwen3_6", "qwen3_8", "qwen3_next")) for t in types if t)


def enabled(config: dict[str, Any] | None) -> bool:
    raw = (os.environ.get("MTPLX_M1_FP16_BODY") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    from .cache_state import m1_long_context_defaults

    return m1_long_context_defaults() and _qwen_family(config)


def cast_bf16_body(model: Any) -> dict[str, Any]:
    """Cast every bf16 parameter of ``model`` to fp16 in place; a no-op when it holds none."""

    bf16 = [(name, a) for name, a in tree_flatten(model.parameters()) if a.dtype == mx.bfloat16]
    if not bf16:
        return {"applied": False, "reason": "no bf16 parameters"}
    peak = max(float(mx.abs(a).max().item()) for _, a in bf16)
    if peak > FP16_MAX:
        return {"applied": False, "reason": f"a bf16 parameter reaches {peak:.3g}, past fp16's range"}
    model.set_dtype(mx.float16, lambda dtype: dtype == mx.bfloat16)
    mx.eval(model.parameters())
    report = {
        "applied": True,
        "tensors": len(bf16),
        "bytes": int(sum(a.nbytes for _, a in bf16)),
        "peak_abs": round(peak, 3),
    }
    _STATE.update(report)
    return report


def fingerprint() -> str:
    """Cache-identity component: KV and recurrent state banked under one body dtype must not
    restore into the other."""

    return "fp16_body=1" if _STATE.get("applied") else "fp16_body=0"
