"""Memory the engine gives back is waited out of the footprint before the
admission measures again.

2026-10-27, M1 Max 64 GB, the 27B: an 81,862-token Pi prompt (469 not
cached) was refused 0.3 GiB short of the Mac's abort floor. Reclamation had
given back 4.9 GiB (the allocator pool and a cached conversation), but the
guard measured straight after: the footprint had dropped 0.5 GiB, the Mac's
free pages not at all, and the receipt showed the freed bytes as 7.1 GiB
"outside MLX" (2.8 GiB before reclamation). macOS takes memory MLX frees
back asynchronously: ~0.2 s for 2 GiB on that Mac, mx.synchronize() or not.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.engine_session import EngineSessionManager
from mtplx.session_bank import SessionBank

GIB = 1024**3


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv("MTPLX_FOOTPRINT_RELEASE_WAIT_S", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setattr(srv, "_FOOTPRINT_RELEASE_POLL_S", 0.0)


class _LaggingMachine:
    """MLX's account drops at once when the pool is cleared; the kernel
    takes the bytes back ``lag`` footprint readings later, and only then do
    they show in the Mac's free pages."""

    def __init__(self, *, active_gib: float, cache_gib: float, host_gib: float, lag: int):
        self.active = int(active_gib * GIB)
        self.cache = int(cache_gib * GIB)
        self.host = int(host_gib * GIB)
        self.lag = lag
        self.pending = 0  # freed by MLX, still in the footprint
        self.released = 0  # back in the Mac's free pages
        self.reads_left = 0

    def stats(self) -> dict:
        return {"ok": True, "active_memory_bytes": self.active, "cache_memory_bytes": self.cache}

    def clear_cache(self) -> None:
        self.pending += self.cache
        self.cache = 0
        self.reads_left = self.lag

    def footprint(self, *args, **kwargs) -> int:
        if self.pending:
            if self.reads_left <= 0:
                self.released += self.pending
                self.pending = 0
            self.reads_left -= 1
        return self.active + self.cache + self.host + self.pending


def _install(monkeypatch, machine: _LaggingMachine, *, free: int) -> None:
    import mlx.core as mx

    monkeypatch.setattr(srv, "_mlx_memory_stats_live", machine.stats)
    monkeypatch.setattr(mx, "clear_cache", machine.clear_cache)
    monkeypatch.setattr(mx, "get_active_memory", lambda: machine.active)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: machine.cache)
    monkeypatch.setattr(srv, "phys_footprint_bytes", machine.footprint)
    wired = 30 * GIB
    compressor = 14 * GIB
    total = 64 * GIB

    def reading() -> sm.SystemMemory:
        free_now = free + machine.released
        return sm.SystemMemory(
            available_bytes=free_now + 2 * GIB,
            total_bytes=total,
            level_percent=int((total - wired - compressor) * 100 // total),
            free_bytes=free_now,
            file_backed_bytes=2 * GIB,
            wired_bytes=wired,
            compressor_bytes=compressor,
            swap_used_bytes=0,
            monotonic_s=0.0,
        )

    monkeypatch.setattr(sm, "_reader", reading)


def _state(manager):
    args = SimpleNamespace(
        layer_types=["linear_attention"] * 48 + ["full_attention"] * 16,
        num_attention_heads=24,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        hidden_size=5120,
    )
    plan = SimpleNamespace(
        available=True,
        kv_bytes_per_token=65_536,
        kv_bytes_per_token_effective=36_044,
        aux_bytes_per_token=0,
        prefill_transient_bytes_per_token=0,
        runtime_transients_bytes=3 * GIB,
        model_weights_bytes=int(15.7 * GIB),
    )
    return SimpleNamespace(
        metal_memory_caps={"memory_limit_bytes": 52 * GIB, "total_ram_bytes": 64 * GIB},
        memory_plan=plan,
        runtime=SimpleNamespace(model=SimpleNamespace(args=args), mtp_enabled=True),
        sessions=manager,
        dashboard=SimpleNamespace(),
        allow_swap=False,
        memory_budget_bytes=None,
    )


def _admit(monkeypatch, *, free: int, lag: int):
    manager = EngineSessionManager(
        bank=SessionBank(max_entries=8, max_bytes=12 * GIB, per_session_max_bytes=10 * GIB),
        idle_ttl_s=3600,
    )
    session = manager.get_or_create("anon-pi")
    assert session.try_begin_generation()
    machine = _LaggingMachine(active_gib=19.5, cache_gib=4.0, host_gib=2.8, lag=lag)
    _install(monkeypatch, machine, free=free)
    try:
        return srv._prefill_admission_shed(
            _state(manager),
            prompt_ids=list(range(20_000)),
            session_bank=manager.bank,
            session_id="anon-pi",
            prefill_chunk_tokens=2048,
        )
    finally:
        session.end_generation()


def _shortfall_free(monkeypatch, short_bytes: int) -> int:
    """The free pages at which the request, credited nothing for the pool,
    falls ``short_bytes`` under the Mac's abort floor."""

    probe_free = 1 * GIB
    monkeypatch.setenv("MTPLX_FOOTPRINT_RELEASE_WAIT_S", "0")
    receipt = _admit(monkeypatch, free=probe_free, lag=10**9)
    monkeypatch.delenv("MTPLX_FOOTPRINT_RELEASE_WAIT_S")
    assert receipt["refused"] is True
    return probe_free + int(receipt["system_shortfall_bytes_after"]) - short_bytes


def test_the_pool_the_guard_gives_back_is_credited_once_the_kernel_takes_it(monkeypatch):
    free = _shortfall_free(monkeypatch, short_bytes=GIB)
    receipt = _admit(monkeypatch, free=free, lag=3)
    assert receipt.get("refused") is not True
    assert "allocator_pool" in receipt["reclamation_steps"]
    assert receipt["footprint_release_wait_s"] >= 0.0
    # The freed pool is in neither line once it is back: no phantom
    # overhang is charged after reclamation.
    assert receipt["host_overhang_charged_bytes_after"] == 0


def test_without_the_wait_the_same_turn_is_refused(monkeypatch):
    """The 2026-10-27 refusal, kept reachable: the guard reads before the
    kernel has the memory back."""

    free = _shortfall_free(monkeypatch, short_bytes=GIB)
    monkeypatch.setenv("MTPLX_FOOTPRINT_RELEASE_WAIT_S", "0")
    receipt = _admit(monkeypatch, free=free, lag=3)
    assert receipt["refused"] is True
    assert "footprint_release_wait_s" not in receipt


def test_a_release_that_never_lands_waits_only_the_bound(monkeypatch):
    monkeypatch.setenv("MTPLX_FOOTPRINT_RELEASE_WAIT_S", "0.05")
    receipt = _admit(monkeypatch, free=1 * GIB, lag=10**9)
    assert receipt["refused"] is True
    assert 0.05 <= receipt["footprint_release_wait_s"] < 1.0


def test_nothing_pending_returns_at_once(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx, "get_active_memory", lambda: 10 * GIB)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 12 * GIB)
    assert srv._await_footprint_release(2 * GIB) is None
    assert srv._await_footprint_release(None) is None
