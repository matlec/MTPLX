"""The per-chunk check narrows the rest of a prefill instead of stopping it.

2026-10-28, M1 Max 64 GB, Qwen3.8-27B: a 58K pi turn was stopped after the
shed with 3.5 GiB free and reclaimable, because the next 2,048-row chunk was
priced at 2.4 GiB against a 2.3 GiB floor. A 512-row chunk fit.
"""

from __future__ import annotations

from types import SimpleNamespace

import mtplx.server.openai as srv
import tests.test_memguard_admission as base
from mtplx import system_memory as sm
from mtplx.generation import _capped_prefill_spans
from tests.test_memguard_admission import Q27_KV, _q27_runtime

GB = 1_000_000_000
GIB = 1024**3
RAM = 68_719_476_736  # 64 GB


def _incident_reading(available: int) -> sm.SystemMemory:
    # The 2026-10-28 abort's reading, with ``available`` free and reclaimable.
    free = 195_641_344
    return sm.SystemMemory(
        available_bytes=available,
        total_bytes=RAM,
        level_percent=33,
        free_bytes=free,
        file_backed_bytes=max(0, available - free),
        wired_bytes=38_887_456_768,
        compressor_bytes=11_913_953_280,
        swap_used_bytes=3_049_586_688,
        monotonic_s=0.0,
    )


def _guard(monkeypatch, available, narrower):
    monkeypatch.setattr(sm, "_reader", lambda: _incident_reading(available))
    monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {"ok": True, "active_memory_bytes": 26 * GIB, "cache_memory_bytes": 0},
    )
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
    state = SimpleNamespace(dashboard=SimpleNamespace(), allow_swap=False)
    return srv._PrefillSystemGuard(
        state,
        chunk_reserve_bytes=2_587_953_561,
        narrower_chunk_reserves=narrower,
    )


def test_the_incident_runs_on_at_the_narrower_chunk(monkeypatch):
    guard = _guard(
        monkeypatch, 3_722_723_328, [(1024, int(1.6 * GIB)), (512, int(1.0 * GIB))]
    )
    assert guard() is False
    assert guard.tripped is None
    # 1,024 rows leave 3.47 - 1.6 under the 2.26 GiB floor; 512 rows fit.
    assert guard.chunk_cap == 512
    assert guard.chunk_reserve_bytes == int(1.0 * GIB)
    assert guard.narrowed["reason_before"] == "under_abort_floor"
    assert guard.narrowed["chunk_tokens"] == 512
    # Later checks hold the narrower reserve.
    assert guard() is False


def test_without_a_narrower_chunk_the_same_reading_stops_it(monkeypatch):
    guard = _guard(monkeypatch, 3_722_723_328, [])
    assert guard() is True
    assert guard.tripped["reason"] == "under_abort_floor"
    assert guard.tripped["narrowed"] is None
    assert guard.chunk_cap is None


def test_a_mac_too_short_even_for_the_narrowest_chunk_is_stopped(monkeypatch):
    guard = _guard(monkeypatch, int(2.5 * GIB), [(1024, int(1.6 * GIB)), (512, GIB)])
    assert guard() is True
    assert guard.tripped["reason"] == "under_abort_floor"
    assert guard.chunk_cap is None


def test_the_27b_prices_its_narrower_chunks_below_the_full_one(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    for name in (
        "MTPLX_GQA_MMA_PREFILL",
        "MTPLX_GQA_MMA_PREFILL_MIN_PREFIX",
        "MTPLX_M1_MMA_PREFILL_SCRATCH_FRACTION",
    ):
        monkeypatch.delenv(name, raising=False)
    geometry = base.TestGrowthModel._geometry(
        base.TestGrowthModel(), live_bytes_per_token=Q27_KV
    )
    monkeypatch.setattr(srv, "_admission_geometry", lambda state, prefill_width=None: geometry)
    state = SimpleNamespace(runtime=_q27_runtime())
    full = srv._prefill_forward_bill(
        state, new_tokens=55_204, width=2048, prompt_tokens=57_975, geometry=geometry
    )["chunk_bytes"]
    narrower = srv._prefill_narrower_chunk_reserves(
        state,
        prompt_tokens=57_975,
        chunk_tokens=2048,
        priced={"prefill_chunk_tokens": 2048, "miss_tokens": 55_204},
    )
    assert [w for w, _ in narrower] == [1024, 512]
    assert full > narrower[0][1] > narrower[1][1]


class _Check:
    def __init__(self, caps):
        self.caps = list(caps)
        self.cap = None
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.caps:
            self.cap = self.caps.pop(0)
        return False

    def prefill_chunk_cap(self):
        return self.cap


def test_the_spans_after_a_narrowing_are_cut_to_it():
    check = _Check([None, 512, 512, 512, 512, 512])
    spans = list(_capped_prefill_spans([(0, 2048), (2048, 4096), (4096, 5000)], check))
    assert spans == [
        (0, 2048),
        (2048, 2560),
        (2560, 3072),
        (3072, 3584),
        (3584, 4096),
        (4096, 4608),
        (4608, 5000),
    ]


def test_a_check_without_a_cap_leaves_the_spans_alone():
    spans = [(0, 2048), (2048, 3000)]
    assert list(_capped_prefill_spans(spans, lambda: False)) == spans
    assert list(_capped_prefill_spans(spans, None)) == spans
