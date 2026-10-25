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


def test_wide_chunks_run_in_blocks(installed):
    mlp = _mlp()
    x = mx.random.normal((1, 3000, 5120)).astype(mx.float16)
    ref = _exact(mlp, x).astype(mx.float32)
    out = mlp(x).astype(mx.float32)
    assert out.shape == ref.shape
    err = float(mx.sqrt(mx.mean((out - ref) ** 2)) / mx.sqrt(mx.mean(ref ** 2)))
    assert err < 0.03, err
    # Every block, not only the whole: the blocks share the program's output
    # buffer, and the first block read the last block's output (2026-10-20:
    # past 32K the 27B prefilled 4,096-row chunks this way, and the first
    # 2,048 rows of every chunk got the wrong FFN share).
    step = ap._STATE["ffn"].max_rows
    for a in range(0, 3000, step):
        o, r = out[:, a : a + step], ref[:, a : a + step]
        block = float(mx.sqrt(mx.mean((o - r) ** 2)) / mx.sqrt(mx.mean(r ** 2)))
        assert block < 0.03, (a, block)


# Every format the native staging decodes. The scales take the weights' dtype: bf16 packs carry bf16
# scales, which the staging kernels once read as fp16 bits (NaN prefill on bf16 MLX conversions).
_FORMATS = [
    ("affine", 4, 64, mx.float16),
    ("affine", 4, 64, mx.bfloat16),
    ("affine", 8, 32, mx.bfloat16),
    ("mxfp4", 4, 32, mx.bfloat16),
    ("nvfp4", 4, 16, mx.bfloat16),
    ("mxfp8", 8, 32, mx.bfloat16),
]


def _mlp_in(mode, bits, gs, dtype, hidden=5120, inter=17408):
    from mlx_lm.models.qwen3_next import Qwen3NextMLP

    mlp = Qwen3NextMLP(hidden, inter)
    mx.random.seed(5)
    for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) / math.sqrt(lin.weight.shape[1])).astype(dtype)
    nn.quantize(mlp, group_size=gs, bits=bits, mode=mode)
    return mlp


def _rel(out, ref):
    out, ref = out.astype(mx.float32), ref.astype(mx.float32)
    return float(mx.sqrt(mx.mean((out - ref) ** 2)) / mx.sqrt(mx.mean(ref ** 2)))


@pytest.mark.parametrize("mode,bits,gs,dtype", _FORMATS, ids=lambda v: str(v).replace("mlx.core.", ""))
def test_every_format_stages_natively(installed, monkeypatch, mode, bits, gs, dtype):
    mlp = _mlp_in(mode, bits, gs, dtype)
    assert ap._native_layer(mlp)
    x = (mx.random.normal((1, 1024, 5120)) * 0.5).astype(dtype)
    # the program is built on the first long chunk; an earlier test may have left the split disabled
    ffn = ap._STATE["ffn"] or ap._ext().AneFfn(5120, ap._STATE["channels"], ap._ROWS, w8=ap._w8())
    monkeypatch.setitem(ap._STATE, "disabled", False)
    paths = []

    class Counting:
        def __getattr__(self, name):
            return getattr(ffn, name)

        def stage_weights_q(self, *a, **k):
            paths.append("native")
            return ffn.stage_weights_q(*a, **k)

        def stage(self, *a, **k):
            paths.append("dequant")
            return ffn.stage(*a, **k)

    monkeypatch.setitem(ap._STATE, "ffn", Counting())
    out = mlp(x)
    err = _rel(out, _exact(mlp, x))
    assert bool(mx.all(mx.isfinite(out)))
    assert paths and set(paths) == {"native"}, paths
    assert err < 0.03, err


@pytest.mark.parametrize("mode,bits,gs,dtype", _FORMATS, ids=lambda v: str(v).replace("mlx.core.", ""))
def test_every_format_on_the_ane_linear(mode, bits, gs, dtype):
    lin = ap._ext().AneLinear(5120, 2048, ap._ROWS)
    dense = nn.Linear(5120, 2048, bias=False)
    dense.set_dtype(dtype)
    q = nn.QuantizedLinear.from_linear(dense, group_size=gs, bits=bits, mode=mode)
    p = ap._qparams(q)
    x2 = (mx.random.normal((512, 5120)) * 0.5).astype(dtype)
    xq, tx = ap._prep_x(x2)
    mx.eval(lin.stage(xq, tx, ap._staged(p), [0], p[4], p[3], mode=p[5]))
    lin.run(512)
    out = lin.output()[:, :512].T
    assert _rel(out, ap._qmm(x2, p)) < 0.03


def test_unsupported_format_is_refused():
    lin = ap._ext().AneLinear(5120, 2048, ap._ROWS)
    q = nn.QuantizedLinear.from_linear(nn.Linear(5120, 2048, bias=False), group_size=32, bits=4, mode="mxfp4")
    p = ap._qparams(q)
    xq, tx = ap._prep_x(mx.zeros((512, 5120)))
    with pytest.raises(ValueError, match="unsupported quantization"):
        lin.stage(xq, tx, ap._staged(p), [0], 4, 32, mode="nvfp4")
