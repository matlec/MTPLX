"""The system edge: a checkpoint where the prompt's leading system turn ends.

An agent that compacts its conversation (omp's handoff, 2026-10-11) starts a
new one under the same system prompt. The new prompt matches the old
conversation's entry up to the end of the system turn and no further, so a
restore can only resume from a checkpoint at or below that point. At 200K the
grid had thinned the anchors below it down to 512: three compactions matched
20,783 / 20,848 / 21,023 tokens and each re-read the whole system prompt.
"""

from __future__ import annotations

from types import SimpleNamespace

from mtplx.checkpoint_anchors import AnchorPlan, retain_checkpoints
from mtplx.generation import (
    _checkpoint_sink,
    _sink_system_edge,
    _system_prefix_edge,
)

IM_START = 10
SYSTEM = [5]


class _Tokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_start|>"
        return IM_START

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert text == "system" and add_special_tokens is False
        return list(SYSTEM)


RT = SimpleNamespace(tokenizer=_Tokenizer())


def _plain(position: int):
    return (position, None, None)


def _positions(records) -> list[int]:
    return [int(record[0]) for record in records]


def _long_session(system_edge: int | None) -> list[int]:
    # The grid every 16,384 tokens from 512 (what a 200K omp session
    # captured), the system edge, and the prompt-end ladder.
    positions = list(range(512, 200_000, 16_384)) + [199_872, 199_936]
    if system_edge is not None:
        positions.append(system_edge)
    plan = AnchorPlan(
        record_count=8,
        prompt_end=199_999,
        restore_point=180_736,
        system_edge=system_edge,
    )
    return _positions(retain_checkpoints([_plain(p) for p in sorted(positions)], plan))


def test_the_system_edge_survives_a_200k_conversation():
    kept = _long_session(20_780)
    assert 20_780 in kept
    # A compacted conversation that matches 20,783 tokens resumes there.
    assert max(p for p in kept if p <= 20_783) == 20_780


def test_without_it_the_match_falls_back_to_the_first_checkpoint():
    # What the log showed: the 16,896 anchor is thinned away at 200K.
    kept = _long_session(None)
    assert max(p for p in kept if p <= 20_783) == 512


def test_the_edge_is_the_next_turns_im_start():
    prompt = [IM_START, *SYSTEM, 1, 2, 3, 11, IM_START, 7, 8, 9]
    assert _system_prefix_edge(RT, prompt) == 6


def test_no_edge_without_a_leading_system_turn():
    assert _system_prefix_edge(RT, [IM_START, 6, 1, 11, IM_START, 7, 8]) is None
    assert _system_prefix_edge(RT, [1, 2, 3]) is None
    assert _system_prefix_edge(RT, []) is None
    # The system turn is the whole prompt: nothing after it to resume into.
    assert _system_prefix_edge(RT, [IM_START, *SYSTEM, 1, 2, 11, IM_START]) is None
    assert _system_prefix_edge(SimpleNamespace(), [IM_START, *SYSTEM, 1]) is None


def test_the_kill_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_SYSTEM_EDGE_ANCHOR", "0")
    assert _system_prefix_edge(RT, [IM_START, *SYSTEM, 1, IM_START, 7, 8]) is None


def test_the_sink_carries_the_edge_to_the_prefill():
    sink = _checkpoint_sink([], session_bank=None, prompt_len=100, system_edge=40)
    assert sink.plan.system_edge == 40
    assert _sink_system_edge(sink) == 40
    assert _sink_system_edge([]) is None
    assert _sink_system_edge(None) is None
