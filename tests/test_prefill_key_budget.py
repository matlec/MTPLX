"""Prefill chunk narrowed by rows x context (macOS display-abort budget)."""

import pytest

from mtplx import generation as g


@pytest.fixture(autouse=True)
def _m1_moe_lane(monkeypatch):
    import mtplx.cache_state as cs

    monkeypatch.setattr(cs, "m1_long_context_defaults", lambda: True)
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_KEY_BUDGET", str(4096 * 49152))


def test_off_without_the_lane_env(monkeypatch):
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_KEY_BUDGET")
    assert g._key_budget_prefill_chunk(4096, 131072) == 4096


def test_the_moe_lane_sets_the_budget():
    from mtplx.commands.public import _M1_MOE_ENV

    assert int(_M1_MOE_ENV["MTPLX_PREFILL_CHUNK_KEY_BUDGET"]) == 4096 * 49152


@pytest.mark.parametrize(
    "context,expected",
    [(8192, 4096), (32768, 4096), (49152, 4096), (77446, 2048), (97307, 2048), (131072, 1024), (250000, 512)],
)
def test_default_budget_halves_until_it_fits(context, expected):
    assert g._key_budget_prefill_chunk(4096, context) == expected


def test_budget_off_and_override(monkeypatch):
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_KEY_BUDGET", "0")
    assert g._key_budget_prefill_chunk(4096, 200000) == 4096
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_KEY_BUDGET", str(4096 * 100000))
    assert g._key_budget_prefill_chunk(4096, 90000) == 4096
    assert g._key_budget_prefill_chunk(4096, 120000) == 2048


def test_long_context_cap_still_applies_past_its_threshold(monkeypatch):
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE_LONG_CONTEXT", raising=False)
    assert g._long_context_prefill_chunk(4096, 200000) == 512
    assert g._long_context_prefill_chunk(4096, 90000) == 2048


def test_only_the_a3b_lane_turns_on_context_copy_with_ramp():
    from mtplx.commands.public import _M1_A3B_ENV, _M1_MOE_ENV, _M1_MOE_GDN_ENV

    assert _M1_A3B_ENV["MTPLX_CONTEXT_COPY"] == "1"
    assert _M1_A3B_ENV["MTPLX_RAMP_ENABLED"] == "1"
    assert _M1_A3B_ENV["MTPLX_RAMP_BLOCK"] == "96"
    assert _M1_A3B_ENV["MTPLX_CCOPY_ADAPTIVE_BLOCK"] == "16"
    assert _M1_A3B_ENV["MTPLX_CCOPY_BANK_ROUTE"] == "1"
    assert int(_M1_A3B_ENV["MTPLX_CCOPY_BANK_MAX_LEN"]) >= 97
    assert "MTPLX_CONTEXT_COPY" not in _M1_MOE_ENV
    assert "MTPLX_CONTEXT_COPY" not in _M1_MOE_GDN_ENV
