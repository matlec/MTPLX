"""M1: a turn drops the compact bank entry its own conversation restored by copy."""

from types import SimpleNamespace

import mlx.core as mx

from mtplx import generation as g
from mtplx.cache_state import COMPACT_KV_FORMAT, COMPACT_KV_FORMAT_KEY, CacheSnapshot
from mtplx.session_bank import SessionBank, SessionBankEntry


def _compact():
    return {COMPACT_KV_FORMAT_KEY: COMPACT_KV_FORMAT, "keys": mx.zeros((1, 2, 4, 8), mx.uint8)}


def _entry(session_id="s1"):
    return SessionBankEntry(
        token_ids=(1, 2, 3),
        token_hash="h",
        model_path="models/example",
        mtp_enabled=True,
        hidden_variant=None,
        cache_snapshot=CacheSnapshot(states=(_compact(), ("gdn",)), meta_states=((), ())),
        logits=None,
        hidden=None,
        session_id=session_id,
    )


def test_release_drops_the_entry_and_its_quantized_kv():
    bank = SessionBank(max_entries=4, max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    entry = _entry()
    bank._entries[entry.token_ids] = entry
    assert bank.release_restored_entry(entry, session_id="other") == 0
    assert entry.token_ids in bank._entries
    bank.release_restored_entry(entry, session_id="s1")
    assert entry.token_ids not in bank._entries
    assert entry.cache_snapshot.states == (None, ("gdn",))


def test_applies_only_to_a_clone_of_the_own_conversation(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_RELEASE_RESTORED_ENTRY", "1")
    restored = SimpleNamespace(entry=_entry(), restore_mode="clone")
    assert g._release_restored_entry_applies(restored, "s1")
    assert not g._release_restored_entry_applies(restored, "s2")
    assert not g._release_restored_entry_applies(SimpleNamespace(entry=_entry(), restore_mode="reference_lease"), "s1")
    monkeypatch.setenv("MTPLX_M1_RELEASE_RESTORED_ENTRY", "0")
    assert not g._release_restored_entry_applies(restored, "s1")


def test_admission_prices_the_released_source_out_of_decode_start():
    import tests.test_memguard_admission as base
    from mtplx.server import openai as srv
    from tests.test_memguard_admission import GIB, Q27_KV

    geometry = base.TestGrowthModel._geometry(base.TestGrowthModel(), live_bytes_per_token=Q27_KV)
    kwargs = dict(
        prompt_tokens=200_000, reused_tokens=199_950, restore_copies_prefix=True,
        layout="contiguous_then_repage", source_layout=None, output_tokens=256,
        publish=True, scratch_bytes=GIB, compact_direct=True,
    )
    kept = srv._admission_growth(geometry, **kwargs)
    released = srv._admission_growth(geometry, **kwargs, released_source_bytes=5 * GIB)
    assert released["decode_start_bytes"] == max(0, kept["decode_start_bytes"] - 5 * GIB)
