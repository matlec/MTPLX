"""Prefill attention on metal-flash-attention for the cached prefix (M1, D=256).

The fork's MMA prefill kernel (``sdpa_gqa_mma_prefill``) runs at 5.6-6.0
TFLOPS on M1 Max for Qwen3.8's D=256 GQA shape. metal-flash-attention's
forward kernel reaches 6.6-7.0 for the same rows, because it stages K/V
blocks with the GPU's async copy into threadgroup memory; the Metal front end
no longer accepts that intrinsic in source, so it lives in a precompiled
metallib behind a small native extension (``native_extensions/mfa_attn``).

A chunk of L queries after P cached keys splits in two:

* keys [0, P): no mask, MFA over every query row of a KV head's group at
  once (O normalized, L the log2 log-sum-exp);
* keys [P, P + L): causal, the fork's prefill kernel restricted to them,
  also writing its log2 log-sum-exp;

and the two are merged exactly by their log-sum-exps. M1 Max, 2,048-row
chunk (2026-10-12): 1.17x at P = 32K, 1.18x at 128K and 196K; outputs within
fp16 rounding of the fork kernel's (2-3e-4). Below P = 8K the fork kernel is
as fast and runs alone. ``MTPLX_MFA_PREFILL=0`` turns it off.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from . import sdpa_gqa_mma as _mma

MIN_PREFIX = 8192
_LN2 = math.log(2.0)
_failed: set[str] = set()


def _enabled() -> bool:
    raw = (os.environ.get("MTPLX_MFA_PREFILL") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    from ..cache_state import m1_long_context_defaults

    return m1_long_context_defaults()


@lru_cache(maxsize=1)
def _ext():
    try:
        import mtplx_mfa_attn  # noqa: F401
    except ImportError:
        here = Path(__file__).resolve().parents[2] / "native_extensions" / "mfa_attn"
        if not (here / "mtplx_mfa_attn").is_dir():
            return None
        sys.path.insert(0, str(here))
        try:
            import mtplx_mfa_attn  # noqa: F401
        except ImportError:
            return None
    import mtplx_mfa_attn

    return mtplx_mfa_attn


@lru_cache(maxsize=1)
def _diag_kernel():
    s = _mma._prefill_source()
    s = _mma._rep(
        s,
        """    const int k_begin = 0;
    const int k_end = min(n_kv, P_ + L_);""",
        """    const int k_begin = P_;
    const int k_end = min(n_kv, P_ + L_);""",
    )
    s = _mma._rep(
        s,
        """            const float inv = 1.0f / Ss[r * LDS + 1];""",
        """            const float inv = 1.0f / Ss[r * LDS + 1];
            if (sg == 0 && fn == 0) {
                lse[(size_t)(h * GQA_F + r / QL) * L_ + qb * QL + (r % QL)] =
                    Ss[r * LDS + 0] + log2(Ss[r * LDS + 1]);
            }""",
    )
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_gqa_mma_prefill_diag_lse",
        input_names=[
            "queries", "keys", "values", "k_scales", "v_scales", "offset",
            "k_hs", "k_ts", "v_hs", "v_ts", "s_hs", "s_ts", "scale", "q_total",
        ],
        output_names=["out", "lse"],
        source=s,
    )


@mx.compile
def _merge(o1, lse1, o2, lse2):
    m = mx.maximum(lse1, lse2)
    w1 = mx.exp((lse1 - m) * _LN2)[..., None]
    w2 = mx.exp((lse2 - m) * _LN2)[..., None]
    return (o1.astype(mx.float32) * w1 + o2 * w2) / (w1 + w2)


def mfa_prefill_attention(
    *, queries: mx.array, keys: mx.array, values: mx.array, prefix: int, scale: float
) -> mx.array | None:
    """``sdpa_gqa_mma_prefill``'s contract on MFA for the prefix, or None."""
    if not _enabled():
        return None
    _, hq, q_len, d = (int(x) for x in queries.shape)
    hk, cap = int(keys.shape[1]), int(keys.shape[2])
    prefix = int(prefix)
    if (
        prefix < MIN_PREFIX
        or d != 256
        or queries.dtype != mx.float16
        or keys.dtype != mx.float16
        or values.dtype != mx.float16
        or abs(float(scale) - 1.0 / 16.0) > 1e-9
        or hk <= 0
        or hq % hk
        or prefix + q_len > cap
    ):
        return None
    ext = _ext()
    if ext is None:
        return None
    g = hq // hk
    blocks = (q_len + 3) // 4
    dummy = _mma._dummy_scales()
    o2, lse2 = _diag_kernel()(
        inputs=[queries, keys, values, dummy, dummy, mx.array([prefix], dtype=mx.int32),
                cap * d, d, cap * d, d, 0, 0, float(scale), q_len],
        template=[("InT", queries.dtype), ("OutT", mx.float32), ("D", d), ("GQA_F", g),
                  ("QL", 4), ("KQ", 0), ("KT", 1), ("QKS", 1), ("NSG", 4)],
        grid=(hk * 32, 4, blocks),
        threadgroup=(32, 4, 1),
        output_shapes=[queries.shape, (hq, q_len)],
        output_dtypes=[mx.float32, mx.float32],
    )
    o1, lse1 = ext.mfa_attention_fwd(queries[0], keys[0], values[0], prefix)
    out = _merge(
        o1.reshape(hk, g * q_len, d),
        lse1.reshape(hk, g * q_len),
        o2[0].reshape(hk, g * q_len, d),
        lse2.reshape(hk, g * q_len),
    )
    return out.astype(queries.dtype).reshape(1, hq, q_len, d)
