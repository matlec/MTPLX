"""MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN: the sampled draft chain with one eval per round.

The lane's claim is identity, not similarity: drafted tokens, proposal
distributions and the request generator's stream must be exactly those of the
serial sampled reader, for every seed, including the rounds where the host
does not confirm the device's prediction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx import qwen4_draft_device_chain as chain
from mtplx.fast_sampling import sparse_distribution_from_mlx_logits_relaxed_ties
from mtplx.sampling import SamplerConfig, SparseDistribution


VOCAB = 96
TOP_K = 8


class _StubHead:
    """The FR-Spec capture surface with an identity ranked table."""

    def __init__(self, ids: np.ndarray, vocab_rows: int):
        self._ids = mx.array(ids.astype(np.int32))
        self._vocab_rows = int(vocab_rows)
        self.armed = False

    def arm_prescatter_capture(self, enabled: bool) -> None:
        self.armed = bool(enabled)

    def take_prescatter_row(self, dense):
        return dense.reshape(-1, dense.shape[-1])[-1]


def _plan(*, rows: int = VOCAB, top_k: int = TOP_K, top_p: float = 0.95, temperature: float = 1.0):
    ids = np.arange(rows, dtype=np.int64)
    head = _StubHead(ids, rows)
    return chain.SampledChainPlan(
        head=head,
        ids_np=ids,
        ids_mx=head._ids,
        rows=rows,
        vocab_rows=rows,
        top_k=top_k,
        temperature=temperature,
        top_p=top_p,
    )


def test_host_pick_is_generator_choice_with_the_same_stream():
    seeds = np.random.default_rng(7)
    for trial in range(400):
        size = int(seeds.integers(1, 20))
        probs = seeds.random(size) ** 3
        probs[seeds.random(size) < 0.2] = 0.0
        if probs.sum() <= 0:
            probs[0] = 1.0
        ids = np.sort(seeds.choice(1000, size=size, replace=False))
        keep = probs > 0
        dist = SparseDistribution(ids[keep], probs[keep] / probs[keep].sum(), 1000)
        a = np.random.default_rng(trial)
        b = np.random.default_rng(trial)
        expected = int(a.choice(dist.token_ids, p=dist.probs))
        got = chain.host_pick(dist, float(b.random()))
        assert got == expected
        # One uniform each, so both generators sit at the same point.
        assert a.random() == b.random()


@pytest.mark.parametrize("top_p", [0.95, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_host_distribution_is_the_relaxed_tie_reader(top_p, temperature):
    plan = _plan(top_p=top_p, temperature=temperature)
    config = SamplerConfig(temperature=temperature, top_p=top_p, top_k=TOP_K)
    rng = np.random.default_rng(3)
    for _ in range(60):
        row = mx.array(rng.normal(0, 3, size=(1, 1, VOCAB)).astype(np.float32))
        local, vals, probs = chain.device_support(plan, row)
        mx.eval(local, vals, probs)
        mine = chain.host_distribution(
            plan, np.asarray(local), np.asarray(vals), np.asarray(probs)
        )
        stock = sparse_distribution_from_mlx_logits_relaxed_ties(row[0, -1], config)
        assert mine is not None and stock is not None
        assert np.array_equal(mine.token_ids, stock.token_ids)
        assert np.array_equal(mine.probs, stock.probs)


def test_device_prediction_agrees_with_the_host_except_at_cdf_edges():
    plan = _plan()
    rng = np.random.default_rng(11)
    disagreements = 0
    trials = 1500
    for _ in range(trials):
        row = mx.array(rng.normal(0, 2.5, size=(1, 1, VOCAB)).astype(np.float32))
        uniform = float(rng.random())
        local, vals, probs = chain.device_support(plan, row)
        predicted = chain.device_predict(
            plan, local, vals, probs, mx.array(uniform, dtype=mx.float32)
        )
        mx.eval(local, vals, probs, predicted)
        dist = chain.host_distribution(
            plan, np.asarray(local), np.asarray(vals), np.asarray(probs)
        )
        host = chain.host_pick(dist, uniform)
        if int(predicted.item()) != host:
            disagreements += 1
            cdf = np.cumsum(dist.probs)
            cdf /= cdf[-1]
            assert np.min(np.abs(cdf - uniform)) < 1e-5, (
                "a disagreement away from a cdf edge is a wrong device law"
            )
    assert disagreements <= max(3, trials // 200)


def test_ranked_table_maps_local_rows_to_real_ids():
    ids = np.arange(0, 2 * VOCAB, 2, dtype=np.int64)  # strictly ascending, sparse
    head = _StubHead(ids, 2 * VOCAB)
    plan = chain.SampledChainPlan(
        head=head, ids_np=ids, ids_mx=head._ids, rows=VOCAB, vocab_rows=2 * VOCAB,
        top_k=TOP_K, temperature=1.0, top_p=0.95,
    )
    rng = np.random.default_rng(5)
    for _ in range(50):
        row = mx.array(rng.normal(0, 3, size=(1, 1, VOCAB)).astype(np.float32))
        uniform = float(rng.random())
        token, dist = chain.serial_read(plan, row, uniform)
        assert dist is not None
        assert set(int(t) for t in dist.token_ids) <= set(int(i) for i in ids)
        assert token % 2 == 0
        assert dist.vocab_size == 2 * VOCAB


def test_claim_declines_without_a_live_frspec_head():
    rt = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace()))
    plan, reason = chain.claim(rt, SamplerConfig(temperature=1.0, top_p=0.95, top_k=20))
    assert plan is None and reason == "no_frspec_head"


# ---------------------------------------------------------------------------
# The generation loop
# ---------------------------------------------------------------------------


def _runtime():
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime

    table = np.random.default_rng(101).normal(0, 2.0, size=(VOCAB, VOCAB)).astype(np.float32)
    draft_table = (
        table + np.random.default_rng(202).normal(0, 0.7, size=(VOCAB, VOCAB))
    ).astype(np.float32)
    target = mx.array(table)
    draft = mx.array(draft_table)

    class Tokenizer:
        def decode(self, tokens, **_kwargs):
            return " ".join(str(int(token)) for token in tokens)

    class Model:
        def __init__(self):
            self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

        def make_cache(self):
            return []

        def make_mtp_cache(self):
            return []

        def _logits(self, input_ids):
            return mx.take(target, input_ids.astype(mx.int32), axis=0)

        def __call__(self, input_ids, *, cache=None, return_hidden=False, hidden_variant=None, **_kwargs):
            length = int(input_ids.shape[1])
            hidden = mx.zeros((1, length, 2), dtype=mx.float32)
            if return_hidden:
                return self._logits(input_ids), hidden
            return self._logits(input_ids)

        def mtp_forward(self, hidden_states, next_token_ids, *, mtp_cache=None, concat_order=None,
                        return_hidden=False, mtp_hidden_variant=None, position_offset=None):
            length = int(next_token_ids.shape[1])
            hidden = mx.zeros((1, length, 2), dtype=mx.float32)
            logits = mx.take(draft, next_token_ids.astype(mx.int32), axis=0)
            if return_hidden:
                return logits, hidden
            return logits

        def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
            return hidden_states

    model = Model()
    rt = MTPLXRuntime(
        model=model,
        tokenizer=Tokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=True,
        contract=MTPContract(),
    )
    rt.qwen4_relaxed_draft_ties = True

    def capture_stub(input_ids, cache=None, return_hidden=False, hidden_variant=None, capture_backend=None):
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if return_hidden:
            return model._logits(input_ids), hidden, {}
        return model._logits(input_ids), {}

    rt.forward_ar_capture = capture_stub
    return rt


@pytest.fixture(autouse=True)
def _events_are_recorded(monkeypatch):
    """The identity tests read per-draft events.  The serve fast path exports
    MTPLX_DROP_EVENTS=1 into the process environment, and a server test that
    runs earlier in a full suite leaves it there (found 2026-09-18: all of
    this file passed alone and seven cases failed inside the full run, with
    status, rounds and token identity intact and the event list empty)."""

    monkeypatch.delenv("MTPLX_DROP_EVENTS", raising=False)


def _generate(seed: int, max_tokens: int = 48):
    from mtplx.generation import generate_mtpk

    return generate_mtpk(
        _runtime(),
        [3],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=TOP_K),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        stop_token_ids=set(),
        seed=seed,
    )


def _fingerprint(out):
    return {
        "tokens": list(out.tokens),
        "drafted_by_depth": list(out.stats.drafted_by_depth or []),
        "accepted_by_depth": list(out.stats.accepted_by_depth or []),
        "verify_calls": out.stats.verify_calls,
    }


def _install(monkeypatch, *, predict=None):
    import mtplx.generation as generation

    monkeypatch.setenv("MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN", "1")
    plan = _plan()
    monkeypatch.setattr(
        generation._qwen4_sampled_chain, "claim", lambda rt, sampler: (plan, None)
    )
    if predict is not None:
        monkeypatch.setattr(generation._qwen4_sampled_chain, "device_predict", predict)
    return plan


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_chain_is_token_identical_to_the_serial_sampled_lane(monkeypatch, seed):
    # Off explicitly: unset, it defaults on for M1 when the in-place append is built.
    monkeypatch.setenv("MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN", "0")
    baseline = _generate(seed)
    assert baseline.stats.sampled_draft_chain_status == "off"
    _install(monkeypatch)
    on = _generate(seed)
    assert on.stats.sampled_draft_chain_status == "installed"
    assert on.stats.sampled_draft_chain_rounds > 0
    assert _fingerprint(on) == _fingerprint(baseline)
    assert any(
        draft.get("draft_core") == "sampled-chain"
        for event in (on.stats.events or [])
        for draft in event.get("drafts", [])
    )


@pytest.mark.parametrize("seed", [0, 5])
def test_an_unconfirmed_prediction_finishes_serially_with_the_same_tokens(monkeypatch, seed):
    monkeypatch.delenv("MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN", raising=False)
    baseline = _generate(seed)

    def always_wrong(plan, cand_local, cand_vals, cand_probs, uniform):
        # A token the host almost never picks: forces the cut on every round.
        return mx.array(VOCAB - 1, dtype=mx.int32) + 0 * mx.sum(cand_local)

    _install(monkeypatch, predict=always_wrong)
    on = _generate(seed)
    assert on.stats.sampled_draft_chain_cuts > 0
    assert _fingerprint(on) == _fingerprint(baseline)
    assert any(
        draft.get("draft_core") == "sampled-chain-serial"
        for event in (on.stats.events or [])
        for draft in event.get("drafts", [])
    )
