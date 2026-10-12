"""The sampled draft chain on draft logits that already are the compact row (the FR-Spec
legacy swap of the dense 27B family): claim_compact_logits."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx import qwen4_draft_device_chain as chain
from mtplx.sampling import SamplerConfig

VOCAB = 64
IDS = np.array([40, 3, 17, 60, 8, 25, 33, 1, 50, 12, 9, 44], dtype=np.int64)  # not ascending
SAMPLER = SamplerConfig(temperature=1.0, top_p=0.95, top_k=5)


def _plan():
    plan, reason = chain.claim_compact_logits(IDS, VOCAB, SAMPLER)
    assert reason is None
    return plan


def test_device_prediction_matches_the_host_pick_on_an_unsorted_table():
    plan = _plan()
    rng = np.random.default_rng(0)
    for _ in range(200):
        logits = mx.array(rng.normal(size=(1, 1, len(IDS))).astype(np.float32))
        support = chain.device_support(plan, logits)
        u = float(rng.random())
        predicted = int(chain.device_predict(plan, *support, mx.array(u, dtype=mx.float32)).item())
        q = chain.host_distribution(plan, *(np.asarray(a) for a in support))
        assert predicted == chain.host_pick(q, u)
        assert set(q.token_ids.tolist()) <= set(IDS.tolist())


def test_a_full_vocabulary_row_is_gathered_to_the_compact_row():
    plan = _plan()
    full = np.random.default_rng(1).normal(size=(1, 1, VOCAB)).astype(np.float32)
    compact = full[..., IDS]
    a = chain.device_support(plan, mx.array(full))
    b = chain.device_support(plan, mx.array(compact))
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    with pytest.raises(RuntimeError):
        chain.device_support(plan, mx.zeros((1, 1, 7)))


def test_claim_refuses_what_it_cannot_serve():
    assert chain.claim_compact_logits(IDS, 10, SAMPLER)[0] is None
    assert chain.claim_compact_logits(IDS, VOCAB, SamplerConfig(temperature=0.0, top_k=5))[0] is None
    assert chain.claim_compact_logits(IDS, VOCAB, SamplerConfig(temperature=1.0, top_k=0))[0] is None
