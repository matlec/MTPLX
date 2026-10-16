"""The query-edge checkpoint anchor (checkpoint_anchors / generation._last_query_edge)."""

from types import SimpleNamespace

from mtplx.checkpoint_anchors import AnchorPlan, retain_checkpoints
from mtplx.generation import _last_query_edge

IM, USER, NL, TOOL, ASSIST = 1, 2, 3, 4, 5


class _Tok:
    def convert_tokens_to_ids(self, token):
        return IM

    def encode(self, text, add_special_tokens=False):
        return {"user\n": [USER, NL], "<tool_response>": [TOOL]}[text]


RT = SimpleNamespace(tokenizer=_Tok())


def test_edge_is_the_reply_to_the_last_real_user_message():
    ids = [IM, 9, 9, IM, USER, NL, 7, 7, IM, ASSIST, 8, IM, USER, NL, TOOL, 6, IM, ASSIST, 8]
    # the last user turn is a tool result; the query is the one at index 3
    assert _last_query_edge(RT, ids) == 8


def test_edge_at_the_generation_prompt_of_a_new_turn():
    ids = [IM, 9, IM, USER, NL, 7, 7, IM, ASSIST, NL]
    assert _last_query_edge(RT, ids) == 7


def test_no_user_turn():
    assert _last_query_edge(RT, [IM, 9, 9]) is None


def _records(positions):
    return [(p, {"s": SimpleNamespace(nbytes=100)}) for p in positions]


def test_retention_keeps_the_newest_checkpoint_at_or_before_the_query_edge():
    positions = [4096, 16384, 19459, 27651, 35618, 37435, 42183, 44035, 52000, 60000, 66000, 70300]
    plan = AnchorPlan(budget_bytes=8 * 100, prompt_end=70369, tail_backoff=64, query_edge=37500)
    kept = [int(r[0]) for r in retain_checkpoints(_records(positions), plan)]
    assert 37435 in kept
    plan_off = AnchorPlan(budget_bytes=8 * 100, prompt_end=70369, tail_backoff=64)
    assert 37435 not in [int(r[0]) for r in retain_checkpoints(_records(positions), plan_off)]
