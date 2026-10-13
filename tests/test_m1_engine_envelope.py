"""M1 family under 128 GB: the engine envelope is 13/16 of RAM, never past macOS's reserve."""

from mtplx import memory_plan as mp

GIB = mp.GIB


def test_m1_envelope_on_64_gb(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_ENGINE_RAM_FRACTION", raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    assert mp.usable_engine_bytes(64 * GIB) == 52 * GIB
    assert mp.usable_engine_bytes(32 * GIB) == 24 * GIB  # capped at RAM - reserve
    assert mp.usable_engine_bytes(128 * GIB) == 90 * GIB  # desktop headroom rule unchanged


def test_off_the_m1_family_the_75_percent_rule_stands(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_ENGINE_RAM_FRACTION", raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    assert mp.usable_engine_bytes(64 * GIB) == 48 * GIB
