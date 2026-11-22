"""bf16 packs served as fp16 on M1 (mtplx.fp16_body): what is cast, what is not, and when."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mtplx import fp16_body


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(fp16_body, "_STATE", {"applied": False})


def _pack(dtype=mx.bfloat16):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            dense = nn.Linear(128, 64, bias=False)
            dense.set_dtype(dtype)
            self.proj = nn.QuantizedLinear.from_linear(dense, group_size=64, bits=4)
            self.norm = nn.RMSNorm(128)
            self.norm.set_dtype(dtype)
            self.a_log = mx.zeros((8,), dtype=mx.float32)

    return Tiny()


def test_casts_bf16_only():
    m = _pack()
    report = fp16_body.cast_bf16_body(m)
    assert report["applied"] and report["tensors"] == 3  # scales, biases, norm weight
    assert m.proj.scales.dtype == m.proj.biases.dtype == m.norm.weight.dtype == mx.float16
    assert m.proj.weight.dtype == mx.uint32
    assert m.a_log.dtype == mx.float32
    assert fp16_body.fingerprint() == "fp16_body=1"


def test_cast_is_exact_in_range():
    m = _pack()
    before = m.proj.scales.astype(mx.float32)
    fp16_body.cast_bf16_body(m)
    assert mx.array_equal(m.proj.scales.astype(mx.float32), before)


def test_fp16_pack_is_left_alone():
    m = _pack(mx.float16)
    assert not fp16_body.cast_bf16_body(m)["applied"]
    assert fp16_body.fingerprint() == "fp16_body=0"


def test_out_of_range_pack_is_refused():
    m = _pack()
    m.norm.weight = mx.full((128,), 1e5, dtype=mx.bfloat16)
    report = fp16_body.cast_bf16_body(m)
    assert not report["applied"] and "range" in report["reason"]
    assert m.norm.weight.dtype == mx.bfloat16


@pytest.mark.parametrize(
    "env,m1,model_type,expected",
    [
        ("", True, "qwen3_5", True),
        ("", True, "qwen3_8_text", True),
        ("", True, "llama", False),
        ("", False, "qwen3_5", False),
        ("1", False, "llama", True),
        ("0", True, "qwen3_5", False),
    ],
)
def test_enabled(monkeypatch, env, m1, model_type, expected):
    from mtplx import cache_state

    monkeypatch.setenv("MTPLX_M1_FP16_BODY", env)
    monkeypatch.setattr(cache_state, "m1_long_context_defaults", lambda: m1)
    assert fp16_body.enabled({"model_type": "x", "text_config": {"model_type": model_type}}) is expected
