from __future__ import annotations

from vw_rag.prompting import (
    evaluate_prediction,
    normalize_answer,
    render_prompt,
    select_top,
    short_answer,
    token_positions,
)


class FakeTokenizer:
    name_or_path = "test/tokenizer"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return "<chat>" + "\n".join(message["content"] for message in messages) + "\n<assistant>"


def sample_row():
    return {
        "question": "Who wrote it?",
        "documents": [
            {"document_id": "D1", "title": "One", "text": "First passage."},
            {"document_id": "D2", "title": "Two", "text": "Second passage."},
            {"document_id": "D3", "title": "Three", "text": "Third passage."},
        ],
    }


def test_render_prompt_keeps_selected_documents_in_retrieval_order():
    prompt, spans = render_prompt(FakeTokenizer(), sample_row(), ["D3", "D1"])
    assert "[D1]" in prompt
    assert "[D2]" not in prompt
    assert "[D3]" in prompt
    assert prompt.index("[D1]") < prompt.index("[D3]")
    assert set(spans) == {"D1", "D3"}
    for document_id, expected in (("D1", "First passage."), ("D3", "Third passage.")):
        start, end = spans[document_id]
        assert prompt[start:end] == expected


def test_token_positions_uses_overlap_and_skips_zero_width_tokens():
    offsets = [(0, 0), (2, 5), (5, 8), (9, 12)]
    assert token_positions(offsets, {"D1": (4, 10)}) == {"D1": [1, 2, 3]}


def test_answer_normalization_and_metrics():
    assert normalize_answer("The Apple Corps!") == "apple corps"
    assert short_answer('Answer: "Apple Corps"\nextra') == "Apple Corps"
    assert evaluate_prediction("Apple Corps", ["the Apple Corps"])["em"] == 1.0
    assert evaluate_prediction("Apple", ["Apple Corps"])["f1"] == 2 / 3


def test_select_top_has_reproducible_tie_breaking():
    assert select_top(["D2", "D1", "D3"], [0.5, 0.5, 0.1], 2) == ["D1", "D2"]
