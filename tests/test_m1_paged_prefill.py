"""M1 family, quantized KV: prompts of 32K+ prefill straight into the pages."""

from mtplx import generation as g


def _layout(monkeypatch, tokens, **env):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return g._sustained_prefill_layout(tokens)


def test_long_prompts_prefill_into_pages_on_m1(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_PAGED_PREFILL", raising=False)
    assert _layout(monkeypatch, 40_000, MTPLX_M1_LONG_CONTEXT="1") == "paged"
    assert _layout(monkeypatch, 8_000, MTPLX_M1_LONG_CONTEXT="1") == "contiguous_then_repage"


def test_switch_and_other_macs_keep_the_contiguous_layout(monkeypatch):
    assert _layout(monkeypatch, 40_000, MTPLX_M1_LONG_CONTEXT="1", MTPLX_M1_PAGED_PREFILL="0") == "contiguous_then_repage"
    monkeypatch.delenv("MTPLX_M1_PAGED_PREFILL", raising=False)
    assert _layout(monkeypatch, 40_000, MTPLX_M1_LONG_CONTEXT="0") == "contiguous_then_repage"
