"""Restore candidates rank by where a restore actually resumes (session_bank)."""

from types import SimpleNamespace

from mtplx.session_bank import _achievable_restore_point


def _entry(prefix_len, boundaries, recurrent=True):
    def probe(matched):
        best = None
        for b in boundaries:
            if b <= matched and (best is None or b > best[0]):
                best = (b, None, None)
        return best

    return SimpleNamespace(
        has_recurrent=recurrent,
        prefix_len=prefix_len,
        gdn_boundaries=[(b, None) for b in boundaries],
        recurrent_boundary_at_or_below=probe,
    )


def test_hybrid_entry_ranks_at_its_checkpoint():
    long_run = _entry(219_502, [4096, 19_459, 65_794, 131_435])
    previous_turn = _entry(88_427, [4096, 19_459, 50_179, 61_952])
    assert _achievable_restore_point(long_run, 62_056, 8) == 19_459
    assert _achievable_restore_point(previous_turn, 61_952, 8) == 61_952
    ranked = sorted(
        [(long_run, 62_056), (previous_turn, 61_952)],
        key=lambda item: (_achievable_restore_point(item[0], item[1], 8), item[1]),
        reverse=True,
    )
    assert ranked[0][0] is previous_turn


def test_attention_only_and_unloaded_entries_rank_at_matched():
    assert _achievable_restore_point(_entry(1000, [10], recurrent=False), 900, 8) == 900
    assert _achievable_restore_point(_entry(1000, []), 900, 8) == 900


def test_no_checkpoint_below_ranks_last():
    assert _achievable_restore_point(_entry(1000, [950]), 900, 8) == 0
