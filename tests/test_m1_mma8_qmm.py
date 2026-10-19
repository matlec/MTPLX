"""M1 verify qmm for 7-8 rows on simdgroup MMA (mtplx/kernels/mma8_qmm.py)."""

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.mark.parametrize("bits,group_size", [(4, 32), (4, 64), (4, 128), (8, 32), (8, 64)])
@pytest.mark.parametrize("rows", [7, 8])
def test_matches_the_stock_quantized_matmul(bits, group_size, rows):
    from mtplx.kernels.mma8_qmm import mma8v2_qmm

    mx.random.seed(0)
    K, N = 1024, 256
    w = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    x = mx.random.normal((rows, K)).astype(mx.float16)
    ref = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=group_size, bits=bits)
    out = mma8v2_qmm(x, wq, scales, biases, group_size=group_size, bits=bits)
    assert out.shape == ref.shape
    err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
    assert err / float(mx.max(mx.abs(ref.astype(mx.float32)))) < 3e-3


def test_each_column_ignores_the_others():
    """N-invariant: a column's value does not depend on how many columns the
    launch computes (so fused projections stay element-identical)."""

    from mtplx.kernels.mma8_qmm import mma8v2_qmm

    mx.random.seed(1)
    K = 1024
    w = (mx.random.normal((512, K)) * 0.02).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((8, K)).astype(mx.float16)
    whole = mma8v2_qmm(x, wq, scales, biases, group_size=64)
    part = mma8v2_qmm(x, wq[:256], scales[:256], biases[:256], group_size=64)
    assert mx.array_equal(whole[:, :256], part).item()


def test_the_lane_is_off_by_default(monkeypatch):
    from mtplx import nax_verify

    monkeypatch.delenv("MTPLX_M1_MMA8", raising=False)
    assert nax_verify._m1_mma8_enabled() is False
    monkeypatch.setenv("MTPLX_M1_MMA8", "1")
    assert nax_verify._m1_mma8_enabled() is True


def test_a_deep_draft_widens_the_compiled_verify_window(monkeypatch):
    """Depth 7 verifies 8 tokens; past the default window of 6 every verify
    ran eager on the generic paged cache (27B 32K: 221 ms against 120)."""

    from mtplx import generation as g

    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_MAX_LEN", raising=False)
    assert g._compiled_verify_len_for_depth(3) is None
    assert g._compiled_verify_len_for_depth(5) is None
    assert g._compiled_verify_len_for_depth(7) == 8
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_LEN", "6")
    assert g._compiled_verify_len_for_depth(7) is None
