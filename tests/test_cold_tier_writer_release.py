"""The SSD writer lets a finished write's encoded payload go.

2026-10-29, M1 Max 64 GB, 27B: the writer loop's ``pending`` kept the last
written entry, 2.3 GB of encoded host bytes, alive while the thread waited
for the next write, and evicting an old entry pruned blob directories once
per blob (two path resolves and a failed rmdir each), so a write took
minutes while its payload sat in RAM. Together they kept 2-7 GB outside MLX
and the watchdog restarting the server every hour or two under pi.
"""

from __future__ import annotations

import gc
import weakref

from mtplx.cache_bank.cold_tier import PendingWrite, SessionBankColdTier
from tests.test_cold_tier_eviction_yield import seed_entry


class _Payload:
    """Stands in for one encoded tensor (bytes cannot be weakly referenced)."""


def test_a_written_payload_is_not_held_after_the_write(tmp_path, monkeypatch):
    tier = SessionBankColdTier(base_dir=tmp_path, mode="on")
    written = []
    monkeypatch.setattr(tier, "_write_pending", lambda pending: written.append(1) or True)
    payload = _Payload()
    ref = weakref.ref(payload)
    pending = PendingWrite(
        entry_id="e" * 64,
        token_ids=(1, 2, 3),
        metadata={"prefix_len": 3, "nbytes": 1},
        payload_spec=None,
        tensors={"tensor_00000000": payload},
    )
    try:
        tier._queue.put_nowait(pending)
        del payload, pending
        tier._queue.join()
        gc.collect()
        assert written == [1]
        assert ref() is None
    finally:
        tier.close()


def test_eviction_prunes_each_blob_directory_once_per_batch(tmp_path, monkeypatch):
    tier = SessionBankColdTier(base_dir=tmp_path, mode="on")
    _, tensors = seed_entry(tier, "victim", blobs=130)
    with tier._connect() as conn:
        row = conn.execute("SELECT * FROM entries").fetchone()
    with tier._base_lock:
        candidates = tier._retire_entry_row(row)
    pruned = []
    original = tier._prune_empty_parents
    monkeypatch.setattr(
        tier,
        "_prune_empty_parents",
        lambda path, *, stop_at: pruned.append(path) or original(path, stop_at=stop_at),
    )
    try:
        tier._delete_unreferenced_blobs(candidates, on_yield=lambda: None)
        assert all(not tier._blob_path(d).exists() for d in tensors)
        batches = (len(candidates) + 63) // 64
        assert len(pruned) <= batches * len({tier._blob_path(d).parent for d in tensors})
        assert len(pruned) < len(candidates)
        # Directories emptied by the deletion are gone.
        assert not any((tmp_path / "blobs").glob("*/*.bin"))
        assert not any(p.exists() for p in pruned)
    finally:
        tier.close()
