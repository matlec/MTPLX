"""q8 prefill attention on fp16 rows dequantized once per layer (M1 default).

The q8-pages MMA prefill kernel dequantizes every key and value inside its
inner loop for 24 query rows, so it ran at 4.4 TFLOPS against the fp16
kernel's 5.9 on M1 Max (2026-10-12). One dequant pass to fp16 rows and the fp16
kernel is 1.32-1.36x per layer at 8K-196K, and warm suffix prefill went from
85-89 to 96-100 tok/s on a 56-64K conversation with the same greedy text.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.kernels.sdpa_gqa_mma import (
    dequant_q8_pages_dense,
    sdpa_gqa_mma_prefill,
    sdpa_gqa_mma_prefill_q8_pages,
)

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

HQ, HK, D, BS = 24, 4, 256, 256


def _pages(blocks: int, seed: int):
    mx.random.seed(seed)
    pages = mx.random.randint(-127, 128, (blocks, BS, HK, D)).astype(mx.int8)
    scales = mx.random.uniform(0.002, 0.008, (blocks, BS, HK, 1)).astype(mx.float32)
    return pages, scales


def test_dequant_matches_int8_times_scale_rounded_once():
    pages, scales = _pages(3, 1)
    rows = 700  # not a block multiple, and short of the capacity
    got = dequant_q8_pages_dense(pages, scales, rows, mx.float16)
    want = (
        (pages.astype(mx.float32) * scales)
        .astype(mx.float16)
        .reshape(3 * BS, HK, D)[:rows]
        .transpose(1, 0, 2)[None]
    )
    assert got.shape == (1, HK, rows, D)
    assert bool(mx.array_equal(got, want).item())


def test_dequant_refuses_what_it_cannot_serve():
    pages, scales = _pages(1, 2)
    assert dequant_q8_pages_dense(pages, scales, BS + 1, mx.float16) is None
    assert dequant_q8_pages_dense(pages, scales, 0, mx.float16) is None
    assert dequant_q8_pages_dense(pages.astype(mx.float16), scales, 8, mx.float16) is None


@pytest.mark.parametrize("prefix", [0, 300, 1500])
def test_fp16_rows_match_the_q8_kernel(prefix):
    q_len = 200
    blocks = (prefix + q_len + BS - 1) // BS
    kp, ks = _pages(blocks, 3)
    vp, vs = _pages(blocks, 4)
    mx.random.seed(5)
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.float16)
    scale = 1 / math.sqrt(D)
    rows = prefix + q_len
    want = sdpa_gqa_mma_prefill_q8_pages(
        queries=q, key_pages=kp, value_pages=vp, key_scales=ks, value_scales=vs,
        prefix=prefix, scale=scale,
    )
    got = sdpa_gqa_mma_prefill(
        queries=q,
        keys=dequant_q8_pages_dense(kp, ks, rows, mx.float16),
        values=dequant_q8_pages_dense(vp, vs, rows, mx.float16),
        prefix=prefix,
        scale=scale,
        num_kv_heads=HK,
    )
    diff = mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))).item()
    # fp16 rows round each dequantized value once; the outputs are fp16.
    assert diff < 5e-4


def test_the_bill_charges_one_layers_rows(monkeypatch):
    from mtplx.server import openai

    args = SimpleNamespace(num_key_value_heads=4, head_dim=256)
    monkeypatch.setenv("MTPLX_KV_QUANT_PREFILL_DEQUANT", "1")
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    assert openai._m1_q8_prefill_dequant_bytes(args, 196_608) == 2 * 196_608 * 4 * 256 * 2
    monkeypatch.setenv("MTPLX_KV_QUANT_PREFILL_DEQUANT", "0")
    assert openai._m1_q8_prefill_dequant_bytes(args, 196_608) == 0
    monkeypatch.setenv("MTPLX_KV_QUANT_PREFILL_DEQUANT", "1")
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "off")
    assert openai._m1_q8_prefill_dequant_bytes(args, 196_608) == 0
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    assert openai._m1_q8_prefill_dequant_bytes(SimpleNamespace(), 196_608) == 0
