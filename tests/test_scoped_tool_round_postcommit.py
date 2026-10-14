"""Scoped reasoning history: a tool-call turn's postcommit keeps its think.

Qwen3.6 templates keep ``<think>`` for the active round, and agent clients
(omp) echo the turn's reasoning on the tool continuation. The postcommit
history used to append the generated turn without its interior (an empty
think scaffold), so the generation-final check refused and every tool round
re-prefilled the whole response (2026-10-14 omp Kyoto run on
Qwen3.6-35B-A3B: a 12k-token write cost 25-70 s on the next call).

CPU only: the real Qwen3.6 tokenizer and chat template, no model load.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.server import openai as oa

MODEL_DIR = Path.home() / ".mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "config.json").exists(),
    reason="Qwen3.6-35B-A3B model pack not cached locally",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
SYSTEM = {"role": "system", "content": "You are a coding agent."}
U1 = {"role": "user", "content": "Count the lines of snake.html."}
THINK = "I should run wc -l on the file."
CALL = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": "wc -l snake.html"})},
    }
]


@pytest.fixture(scope="module")
def tok():
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    return _load_tokenizer_resilient(MODEL_DIR, config)


def _encode(tok, messages):
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=True,
        reasoning_effort="medium",
        strip_assistant_reasoning_history=False,
        scoped_reasoning_history=True,
        tools=TOOLS,
        tool_choice=None,
        template_observability={},
    )


def _history_ids(tok, monkeypatch, committed_stream_ids):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: True)
    monkeypatch.setattr(oa, "_reasoning_history_preserve_echo_active", lambda state: False)
    monkeypatch.setattr(oa, "_reasoning_effort_for_state", lambda *a, **kw: "medium")
    state = SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False, tool_prompt_mode="hybrid"),
        runtime=SimpleNamespace(tokenizer=tok),
    )
    ids, _splice = oa._history_ids_for_postcommit(
        state,
        messages=oa.ChatCompletionRequest(model="m", messages=[SYSTEM, U1]).messages,
        assistant_content="",
        assistant_tool_calls=CALL,
        thinking_enabled=True,
        reasoning_effort="medium",
        tool_specs=TOOLS,
        tool_prompt_mode="hybrid",
        committed_stream_ids=committed_stream_ids,
    )
    return ids


def _turn(tok):
    prompt = _encode(tok, [SYSTEM, U1])
    # What the client sends next: this turn's reasoning echoed, plus the result.
    next_prompt = _encode(
        tok,
        [
            SYSTEM,
            U1,
            {"role": "assistant", "content": "", "reasoning_content": THINK, "tool_calls": CALL},
            {"role": "tool", "tool_call_id": "call_1", "content": "412 snake.html"},
        ],
    )
    tail = tok.decode(next_prompt[len(prompt):])
    generated_text = tail[: tail.index("<|im_end|>") + len("<|im_end|>")]
    generated = oa._encode_rendered_chat_text(tok, generated_text)
    return prompt, generated, next_prompt


def test_scoped_tool_turn_history_carries_this_turns_think(tok, monkeypatch):
    prompt, generated, next_prompt = _turn(tok)
    final_ids = prompt + generated
    history_ids = _history_ids(tok, monkeypatch, final_ids)

    assert THINK in tok.decode(history_ids)
    # The generation boundary is a prefix of the retokenized history, so the
    # live KV can be banked without recompute ...
    assert history_ids[: len(final_ids)] == final_ids
    # ... and the client's next request extends it.
    assert next_prompt[: len(final_ids)] == final_ids


def test_scoped_without_committed_stream_keeps_empty_scaffold(tok, monkeypatch):
    history_ids = _history_ids(tok, monkeypatch, None)
    assert THINK not in tok.decode(history_ids)
