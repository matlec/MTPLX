"""MTPLX_M1_RESIDENT_DECODE building blocks: device row selection, the
device-step tape replay and the rewind snapshot."""

import mlx.core as mx
import pytest

from mtplx import resident_decode as rd
from mtplx.gdn_capture import (
    _linear_gated_delta_from_conv_tape_replay,
    linear_gated_delta_from_conv_tape_replay_steps,
)


def test_select_row_matches_host_index():
    value = mx.random.normal((1, 2, 3, 5))
    for index in (0, 1):
        got = rd._select_row(value, mx.array(index, dtype=mx.int32))
        assert mx.array_equal(got, value[:, index]).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="metal kernel")
def test_device_step_tape_replay_is_bit_identical():
    B, T, Hk, Hv, Dk, Dv = 1, 2, 2, 4, 64, 64
    key_dim = Hk * Dk
    conv_dim = 2 * key_dim + Hv * Dv
    meta = {
        "conv_dim": conv_dim,
        "head_k_dim": Dk,
        "head_v_dim": Dv,
        "num_k_heads": Hk,
        "num_v_heads": Hv,
        "key_dim": key_dim,
    }
    conv_out = mx.random.normal((B, T, conv_dim)).astype(mx.bfloat16)
    tape = mx.random.normal((B, T, Hv, Dv)).astype(mx.float32)
    g = mx.random.uniform(0.5, 1.0, (B, T, Hv)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    # A corrupt device count is clamped to the tape instead of spinning.
    for steps, device_steps in ((2, 1 << 30), (2, 3)):
        ref = _linear_gated_delta_from_conv_tape_replay(
            tape, conv_out, g, state, meta, steps=steps
        )
        got = linear_gated_delta_from_conv_tape_replay_steps(
            tape, conv_out, g, state, meta, mx.array(device_steps, dtype=mx.int32)
        )
        assert mx.array_equal(ref, got).item()
    for steps in (1, 2):
        ref = _linear_gated_delta_from_conv_tape_replay(
            tape, conv_out, g, state, meta, steps=steps
        )
        got = linear_gated_delta_from_conv_tape_replay_steps(
            tape, conv_out, g, state, meta, mx.array(steps, dtype=mx.int32)
        )
        assert ref is not None and got is not None
        assert mx.array_equal(ref, got).item()


class _Attn:
    def __init__(self, offset):
        self.cache = [mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4)), mx.array(offset, dtype=mx.int32)]
        self.rollback_state = [None, None, None]

    def is_trimmable(self):
        return True


def test_device_commit_trims_attention_offset_by_rejection():
    for accepted, expected in ((1, 10), (0, 9)):
        entry = _Attn(10)
        rd.device_commit([entry], {}, mx.array(accepted, dtype=mx.int32))
        assert int(entry.cache[2].item()) == expected


def test_snapshot_restores_attention_offset():
    entry = _Attn(7)
    snap = rd._snapshot([entry], [])
    entry.cache[2] = mx.array(11, dtype=mx.int32)
    rd._restore(snap)
    assert int(entry.cache[2].item()) == 7


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_RESIDENT_DECODE", raising=False)
    assert rd.enabled() is False
    monkeypatch.setenv("MTPLX_M1_RESIDENT_MAX_CONTEXT", "1234")
    assert rd.max_context() == 1234


@pytest.mark.skipif(not mx.metal.is_available(), reason="metal kernel")
def test_inplace_row_writes_match_slice_update():
    from mtplx.cache_state import inplace_write_axis2, inplace_write_rows

    buf = mx.random.randint(-100, 100, (64, 2, 8)).astype(mx.int8)
    upd = mx.random.randint(-100, 100, (2, 2, 8)).astype(mx.int8)
    offset = mx.array(17, dtype=mx.int32)
    ref = mx.slice_update(buf, upd, offset, axes=(0,))
    mx.eval(ref)
    work = buf + mx.array(0, dtype=buf.dtype)
    mx.eval(work)
    done = inplace_write_rows(work, upd, offset)
    out = mx.depends([work], [done])[0]
    # A reader with no freshly produced input gets no barrier after the
    # writer (see inplace_write_rows); evaluate the write first.
    mx.eval(out)
    assert mx.array_equal(out, ref).item()

    buf = mx.random.normal((1, 2, 32, 16)).astype(mx.bfloat16)
    upd = mx.random.normal((1, 2, 2, 16)).astype(mx.bfloat16)
    ref = mx.slice_update(buf, upd, offset, axes=(2,))
    mx.eval(ref)
    work = buf + mx.array(0, dtype=buf.dtype)
    mx.eval(work)
    done = inplace_write_axis2(work, upd, offset)
    out = mx.depends([work], [done])[0]
    mx.eval(out)
    assert mx.array_equal(out, ref).item()


def test_inplace_row_writes_skip_offsets_past_the_buffer():
    from mtplx.cache_state import inplace_write_axis2, inplace_write_rows

    for offset in (63, 64, 1 << 20, -1):
        buf = mx.random.randint(-100, 100, (64, 2, 8)).astype(mx.int8)
        mx.eval(buf)
        work = buf + mx.array(0, dtype=buf.dtype)
        mx.eval(work)
        upd = mx.ones((2, 2, 8), dtype=mx.int8)
        done = inplace_write_rows(work, upd, mx.array(offset, dtype=mx.int32))
        mx.eval(done)
        assert done.item() == -1
        assert mx.array_equal(work, buf).item()

        buf = mx.random.normal((1, 2, 64, 16)).astype(mx.bfloat16)
        mx.eval(buf)
        work = buf + mx.array(0, dtype=buf.dtype)
        mx.eval(work)
        upd = mx.ones((1, 2, 2, 16), dtype=mx.bfloat16)
        done = inplace_write_axis2(work, upd, mx.array(offset, dtype=mx.int32))
        mx.eval(done)
        assert done.item() == -1
        assert mx.array_equal(work, buf).item()


def test_inplace_kv_off_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_RESIDENT_INPLACE_KV", raising=False)
    assert rd.inplace_kv_enabled() is False


def test_device_pick_matches_numpy_choice():
    import numpy as np

    from mtplx.qwen4_draft_device_chain import host_pick
    from mtplx.sampling import SparseDistribution

    rng = np.random.default_rng(3)
    for _ in range(50):
        ids = np.sort(rng.choice(1000, 20, replace=False))
        probs = rng.random(20)
        probs /= probs.sum()
        u = float(rng.random())
        host = host_pick(SparseDistribution(ids, probs, 1000), u)
        dev = int(
            rd.device_pick(
                mx.array(ids, dtype=mx.int32), mx.array(probs, dtype=mx.float32), mx.array(u, dtype=mx.float32)
            ).item()
        )
        assert dev == host


def test_device_round_decision_matches_host_residual():
    import numpy as np

    from mtplx.qwen4_draft_device_chain import host_pick
    from mtplx.sampling import SparseDistribution, residual_distribution

    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(60):
        t_ids = np.sort(rng.choice(200, (2, 20), replace=True), axis=1)
        t_ids = np.stack([np.sort(rng.choice(200, 20, replace=False)) for _ in range(2)])
        t_p = rng.random((2, 20))
        t_p /= t_p.sum(axis=1, keepdims=True)
        q_ids = np.sort(rng.choice(200, 20, replace=False))
        q_p = rng.random(20)
        q_p /= q_p.sum()
        d = int(q_ids[rng.integers(0, 20)])
        u_acc, u_post = float(rng.random()), float(rng.random())
        acc, bonus, corr = rd.device_round_decision(
            mx.array(t_ids, dtype=mx.int32), mx.array(t_p, dtype=mx.float32),
            mx.array(q_ids, dtype=mx.int32), mx.array(q_p, dtype=mx.float32),
            mx.array([[d]], dtype=mx.int32),
            mx.array(u_acc, dtype=mx.float32), mx.array(u_post, dtype=mx.float32),
        )
        target0 = SparseDistribution(t_ids[0], t_p[0], 200)
        q = SparseDistribution(q_ids, q_p, 200)
        p_d, q_d = target0.probability(d), q.probability(d)
        host_acc = u_acc <= min(1.0, p_d / q_d)
        if abs(u_acc - min(1.0, p_d / q_d)) < 1e-5:
            continue  # float32 edge: the host would rewind this round
        assert bool(int(acc.item())) == host_acc
        if host_acc:
            assert int(bonus.item()) == host_pick(SparseDistribution(t_ids[1], t_p[1], 200), u_post)
        else:
            assert int(corr.item()) == host_pick(residual_distribution(target0, q), u_post)
        checked += 1
    assert checked > 40


class _StockKV:
    """Stock-like KV container: python-int offset, trimmable."""

    def __init__(self, offset):
        self.offset = offset

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n
        return n


def test_validate_cache_refuses_stock_attention_before_anything_moves():
    stock = _StockKV(10)
    with pytest.raises(rd.ResidentUnsupported):
        rd.validate_cache([stock])
    assert stock.offset == 10
    rd.validate_cache([_Attn(5)])


def test_device_commit_trims_stock_containers_on_the_host():
    for accepted, expected in ((1, 10), (0, 9)):
        stock = _StockKV(10)
        rd.device_commit([stock], {}, mx.array(accepted, dtype=mx.int32))
        assert stock.offset == expected


def test_snapshot_restores_int_offsets():
    stock = _StockKV(7)
    snap = rd._snapshot([stock], [])
    stock.offset = 12
    rd._restore(snap)
    assert stock.offset == 7
