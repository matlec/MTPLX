"""Prefill projections on an fp16 GEMM over dequantized weights (mtplx.prefill_dequant_gemm)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import prefill_dequant_gemm as pdg

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture
def installed():
    original = mx.quantized_matmul
    pdg._STATE.update(installed=False, original=None, calls=0)
    pdg.install()
    yield original
    mx.quantized_matmul = original
    pdg._STATE.update(installed=False, original=None, calls=0)


@pytest.mark.parametrize("n,k,m", [(1536, 512, 1024), (512, 1536, 1100), (512, 512, 2048)])
def test_bit_identical_to_the_stock_kernel(installed, n, k, m):
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((m, k)).astype(mx.float16)
    want = installed(x, wq, s, b, transpose=True, group_size=64, bits=4)
    got = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    layer.weight, layer.scales, layer.biases = wq, s, b
    assert bool(mx.array_equal(got, want).item())
    assert bool(mx.array_equal(layer(x), want).item())
    assert pdg._STATE["calls"] == 2


def test_small_and_foreign_calls_stay_stock(installed):
    w = (mx.random.normal((512, 512)) * 0.02).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    mx.quantized_matmul(mx.random.normal((4, 512)).astype(mx.float16), wq, s, b, group_size=64, bits=4)
    mx.quantized_matmul(mx.random.normal((2048, 512)).astype(mx.bfloat16), wq, s, b, group_size=64, bits=4)
    w8, s8, b8 = mx.quantize(w, group_size=64, bits=8)
    mx.quantized_matmul(mx.random.normal((2048, 512)).astype(mx.float16), w8, s8, b8, group_size=64, bits=8)
    assert pdg._STATE["calls"] == 0
