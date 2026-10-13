"""M1 MoE host-sync cuts (deferred depth-1 greedy draft, batched bonus argmax)
must leave the greedy stream and its receipts unchanged on the TinyMTP harness."""

from __future__ import annotations

import pytest

from tests.test_graphbank_compiled_verify import _tiny_mtpk_runtime

_KNOBS = ("MTPLX_M1_MOE_DEFERRED_DRAFT", "MTPLX_M1_MOE_BATCHED_BONUS")


def _run(mtp_token: int, max_tokens: int = 9):
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    rt, _model = _tiny_mtpk_runtime(mtp_token=mtp_token)
    return generate_mtpk(
        rt,
        [0],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        stop_token_ids=set(),
    )


def _fingerprint(out):
    return {
        "tokens": list(out.tokens),
        "drafted_by_depth": list(out.stats.drafted_by_depth or []),
        "accepted_by_depth": list(out.stats.accepted_by_depth or []),
        "verify_calls": out.stats.verify_calls,
        "draft_tokens": [
            draft.get("token")
            for event in (out.stats.events or [])
            for draft in event.get("drafts", [])
        ],
    }


def _cores(out):
    return {
        draft.get("draft_core")
        for event in (out.stats.events or [])
        for draft in event.get("drafts", [])
    }


@pytest.fixture
def _pinned(monkeypatch):
    monkeypatch.setenv("MTPLX_GREEDY_DRAFT_CHAIN", "1")
    monkeypatch.setenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "0")
    for knob in _KNOBS:
        monkeypatch.setenv(knob, "0")
    return monkeypatch


@pytest.mark.parametrize("mtp_token", [1, 2], ids=["all-accept", "all-reject"])
@pytest.mark.parametrize("knobs", [_KNOBS[:1], _KNOBS[1:], _KNOBS])
def test_sync_cuts_match_baseline(_pinned, mtp_token, knobs):
    baseline = _run(mtp_token)
    assert "greedy-chain" in _cores(baseline)
    for knob in knobs:
        _pinned.setenv(knob, "1")
    out = _run(mtp_token)
    assert _fingerprint(out) == _fingerprint(baseline)
    if "MTPLX_M1_MOE_DEFERRED_DRAFT" in knobs:
        assert _cores(out) == {"greedy-chain-deferred"}
    assert -1 not in _fingerprint(out)["draft_tokens"]


def test_deferred_draft_stays_off_past_depth_one(_pinned):
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    _pinned.setenv("MTPLX_M1_MOE_DEFERRED_DRAFT", "1")
    rt, _model = _tiny_mtpk_runtime(mtp_token=1)
    out = generate_mtpk(
        rt,
        [0],
        max_tokens=8,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        stop_token_ids=set(),
    )
    assert "greedy-chain-deferred" not in _cores(out)
