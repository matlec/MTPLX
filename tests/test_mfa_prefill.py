"""Prefill attention on metal-flash-attention for the cached prefix (mtplx.kernels.mfa_prefill)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.kernels import mfa_prefill
from mtplx.kernels.sdpa_gqa_mma import sdpa_gqa_mma_prefill

pytestmark = [
    pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal"),
    pytest.mark.skipif(mfa_prefill._ext() is None, reason="mfa_attn extension not built"),
]


def _case(prefix: int, q_len: int = 256, spare: int = 64):
    mx.random.seed(prefix)
    cap = prefix + q_len + spare
    q = (mx.random.normal((1, 24, q_len, 256)) * 0.7).astype(mx.float16)
    k = (mx.random.normal((1, 4, cap, 256)) * 0.7).astype(mx.float16)
    v = mx.random.normal((1, 4, cap, 256)).astype(mx.float16)
    return q, k, v


@pytest.mark.parametrize("prefix", [8192, 12345])
def test_matches_the_fork_kernel(monkeypatch, prefix):
    q, k, v = _case(prefix)
    monkeypatch.setenv("MTPLX_MFA_PREFILL", "0")
    want = sdpa_gqa_mma_prefill(queries=q, keys=k, values=v, prefix=prefix, scale=1 / 16, num_kv_heads=4)
    monkeypatch.setenv("MTPLX_MFA_PREFILL", "1")
    got = mfa_prefill.mfa_prefill_attention(queries=q, keys=k, values=v, prefix=prefix, scale=1 / 16)
    assert got is not None
    diff = mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))).item()
    assert diff < 2e-3


def test_declines_what_it_does_not_serve(monkeypatch):
    monkeypatch.setenv("MTPLX_MFA_PREFILL", "1")
    q, k, v = _case(8192)
    assert mfa_prefill.mfa_prefill_attention(queries=q, keys=k, values=v, prefix=4096, scale=1 / 16) is None
    assert mfa_prefill.mfa_prefill_attention(queries=q, keys=k, values=v, prefix=8192, scale=0.1) is None
    assert mfa_prefill.mfa_prefill_attention(
        queries=q[:, :, :64], keys=k, values=v, prefix=8192, scale=1 / 16) is None
    assert mfa_prefill.mfa_prefill_attention(
        queries=q.astype(mx.bfloat16), keys=k.astype(mx.bfloat16), values=v.astype(mx.bfloat16),
        prefix=8192, scale=1 / 16) is None
    monkeypatch.setenv("MTPLX_MFA_PREFILL", "0")
    assert mfa_prefill.mfa_prefill_attention(queries=q, keys=k, values=v, prefix=8192, scale=1 / 16) is None
