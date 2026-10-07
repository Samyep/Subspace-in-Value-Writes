from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "Answer the question using only the retrieved passages. "
    "Return only the short answer, without explanation or citations."
)


def render_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    kept_document_ids: Iterable[str] | None = None,
) -> tuple[str, dict[str, tuple[int, int]]]:
    kept = None if kept_document_ids is None else set(kept_document_ids)
    user = f"Question: {row['question']}\n\nRetrieved passages:\n"
    user_spans: dict[str, tuple[int, int]] = {}
    for document in row["documents"]:
        document_id = str(document["document_id"])
        if kept is not None and document_id not in kept:
            continue
        user += f"\n[{document_id}] Title: {document['title']}\nText: "
        start = len(user)
        user += str(document["text"])
        user_spans[document_id] = (start, len(user))
        user += "\n"
    user += "\nShort answer:"

    template_kwargs: dict[str, Any] = {}
    if "qwen3" in str(getattr(tokenizer, "name_or_path", "")).lower():
        template_kwargs["enable_thinking"] = False
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except Exception:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user}"}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    user_offset = prompt.find(user)
    if user_offset < 0:
        raise RuntimeError("Chat template did not preserve the user prompt verbatim")
    spans = {
        document_id: (start + user_offset, end + user_offset)
        for document_id, (start, end) in user_spans.items()
    }
    return prompt, spans


def token_positions(
    offsets: list[tuple[int, int]], spans: dict[str, tuple[int, int]]
) -> dict[str, list[int]]:
    return {
        document_id: [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < end and token_end > start and token_start != token_end
        ]
        for document_id, (start, end) in spans.items()
    }


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def short_answer(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = re.sub(r"^(short\s+answer|answer)\s*:\s*", "", line, flags=re.I)
    return line.strip().strip('"\'')


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def reference_answers(row: dict[str, Any]) -> list[str]:
    candidates = [row["answer"], *row.get("answer_aliases", [])]
    answers: list[str] = []
    seen: set[str] = set()
    for answer in candidates:
        key = normalize_answer(str(answer))
        if key not in seen:
            answers.append(str(answer))
            seen.add(key)
    return answers


def evaluate_prediction(prediction: str, references: list[str]) -> dict[str, float]:
    return {
        "em": max(exact_match(prediction, reference) for reference in references),
        "f1": max(token_f1(prediction, reference) for reference in references),
    }


def select_top(document_ids: list[str], scores: list[float], top_k: int) -> list[str]:
    order = sorted(
        range(len(document_ids)),
        key=lambda index: (-float(scores[index]), document_ids[index]),
    )
    return [document_ids[index] for index in order[:top_k]]
