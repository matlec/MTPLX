"""q8/q4 packed GQA verify attention: 5-8 query rows run as two launches of at most 4."""

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.mark.parametrize("q_len", [5, 6, 7, 8])
@pytest.mark.parametrize("n_kv", [64, 1000])
def test_split_rows_are_identical_to_one_launch(monkeypatch, q_len, n_kv):
    from mtplx.kernels import sdpa_gqa_packed_quant as m

    mx.random.seed(q_len * 7 + n_kv)
    hq, hk, d, cap = 24, 4, 256, 1088
    k_q = mx.random.randint(-127, 127, (1, hk, cap, d)).astype(mx.int8)
    v_q = mx.random.randint(-127, 127, (1, hk, cap, d)).astype(mx.int8)
    k_s = mx.random.uniform(shape=(1, hk, cap, 1)).astype(mx.float32) * 0.01
    v_s = mx.random.uniform(shape=(1, hk, cap, 1)).astype(mx.float32) * 0.01
    q = mx.random.normal((1, hq, q_len, d)).astype(mx.float16)

    def run():
        return m.sdpa_gqa_packed_tail_quant(
            queries=q, k_q=k_q, k_scale=k_s, v_q=v_q, v_scale=v_s,
            offset=n_kv, scale=d ** -0.5, bits=8,
        )

    monkeypatch.setenv("MTPLX_PACKED_QUANT_SPLIT_ROWS", "0")
    one = run()
    monkeypatch.setenv("MTPLX_PACKED_QUANT_SPLIT_ROWS", "1")
    split = run()
    assert one is not None and split is not None
    assert mx.array_equal(one, split).item()


def test_an_array_offset_splits_too(monkeypatch):
    from mtplx.kernels import sdpa_gqa_packed_quant as m

    hq, hk, d, cap = 24, 4, 256, 576
    k_q = mx.zeros((1, hk, cap, d), mx.int8) + 3
    v_q = mx.zeros((1, hk, cap, d), mx.int8) + 5
    s = mx.ones((1, hk, cap, 1), mx.float32) * 0.01
    q = mx.random.normal((1, hq, 8, d)).astype(mx.float16)
    out = m.sdpa_gqa_packed_tail_quant(
        queries=q, k_q=k_q, k_scale=s, v_q=v_q, v_scale=s,
        offset=mx.array(500), scale=d ** -0.5, bits=8, max_offset=512,
    )
    assert out is not None and out.shape == q.shape
