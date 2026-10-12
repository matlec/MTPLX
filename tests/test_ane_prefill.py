"""The prefill FFN split onto the Neural Engine (mtplx.ane_prefill): numerics and the GPU fallback."""

import math

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mtplx import ane_prefill as ap


def _available() -> bool:
    try:
        return ap._ext().unavailable() == ""
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="no ANE / extension not built")


def _mlp(hidden=5120, inter=17408, bits=4, gs=32):
    from mlx_lm.models.qwen3_next import Qwen3NextMLP

    mlp = Qwen3NextMLP(hidden, inter)
    mx.random.seed(3)
    for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.02).astype(mx.float16)
    nn.quantize(mlp, group_size=gs, bits=bits)
    return mlp


def _exact(mlp, x):
    return mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))


@pytest.fixture(scope="module")
def installed():
    import os

    os.environ["MTPLX_ANE_PREFILL"] = "1"
    ap.install(None)
    yield
    ap._STATE["disabled"] = False


def test_split_is_close_to_the_gpu(installed):
    mlp = _mlp()
    x = mx.random.normal((1, 1024, 5120)).astype(mx.float16)
    ref = _exact(mlp, x).astype(mx.float32)
    out = mlp(x).astype(mx.float32)
    err = float(mx.sqrt(mx.mean((out - ref) ** 2)) / mx.sqrt(mx.mean(ref ** 2)))
    assert err < 0.03, err
    assert ap.stats()["calls"] >= 1


def test_short_chunks_stay_exact(installed):
    mlp = _mlp()
    x = mx.random.normal((1, 64, 5120)).astype(mx.float16)
    assert mx.array_equal(mlp(x), _exact(mlp, x))


def test_failure_falls_back_to_the_gpu(installed, monkeypatch):
    mlp = _mlp()
    x = mx.random.normal((1, 1024, 5120)).astype(mx.float16)
    ref = _exact(mlp, x).astype(mx.float32)
    ffn = ap._STATE["ffn"]
    assert ffn is not None

    class Broken:
        def __getattr__(self, name):
            return getattr(ffn, name)

        def run(self, rows):
            raise RuntimeError("injected ANE failure")

    monkeypatch.setitem(ap._STATE, "ffn", Broken())
    out = mlp(x).astype(mx.float32)
    err = float(mx.sqrt(mx.mean((out - ref) ** 2)) / mx.sqrt(mx.mean(ref ** 2)))
    assert ap._STATE["disabled"] and "injected" in ap._STATE["error"]
    assert err < 1e-2, err
    # later calls run the stock FFN
    assert mx.array_equal(mlp(x), _exact(mlp, x))
