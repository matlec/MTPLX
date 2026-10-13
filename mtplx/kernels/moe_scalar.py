"""Scalar type of the A3B row-owned router and whole-MoE kernels.

Upstream wrote these kernels for the BF16 checkpoints. The M1 build serves the
FP16 sibling (the M1 GPU has no native bfloat), whose quantized layout is the
same; the kernels are emitted with ``half`` in place of ``bfloat`` and the
launches produce float16. BF16 stays the default, so the upstream route is
unchanged; the installer sets the type from the model's norm weight.
"""

from __future__ import annotations

import re

import mlx.core as mx

_STATE = {"dtype": mx.bfloat16}


def set_scalar(dtype) -> None:
    _STATE["dtype"] = mx.float16 if dtype == mx.float16 else mx.bfloat16


def scalar():
    return _STATE["dtype"]


def supported(dtype) -> bool:
    return dtype in (mx.bfloat16, mx.float16)


def suffix() -> str:
    return "" if _STATE["dtype"] == mx.bfloat16 else "_f16"


def metal_source(source: str) -> str:
    if _STATE["dtype"] == mx.bfloat16:
        return source
    return re.sub(r"\bbfloat\b", "half", source)
