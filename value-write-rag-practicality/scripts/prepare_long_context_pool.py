#!/usr/bin/env python3
"""Reconstruct and freeze the long-context MuSiQue retrieval pool.

The existing hybrid top-20 is kept byte-for-byte at the document level.  The
pool is extended with unused documents from the same deterministic BM25
ranking.  Evaluation labels never affect retrieval or ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.io_utils import portable_path, load_jsonl, write_json, write_jsonl
from vw_rag.prompting import normalize_answer, render_prompt


DEFAULT_DEPTHS = (20, 40, 80, 160)
TOKENIZER_MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen_14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama_3b": "unsloth/Llama-3.2-3B-Instruct",
}
WORD = re.compile(r"\w+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/musique_long_context_pool.jsonl"
    )
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--no-token-counts", action="store_true")
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def stable_digest(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=32).hexdigest()


def paragraph_key(paragraph: dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_space(paragraph["title"]),
        normalize_space(paragraph["paragraph_text"]),
    )


def build_corpus(dataset) -> tuple[list[dict[str, str]], dict[tuple[str, str], int]]:
    unique: dict[tuple[str, str], None] = {}
    for split in ("train", "validation"):
        for row in dataset[split]:
            for paragraph in row["paragraphs"]:
                unique[paragraph_key(paragraph)] = None
    keys = sorted(unique, key=lambda item: stable_digest(f"{item[0]}\0{item[1]}"))
    corpus = []
    for title, text in keys:
        digest = stable_digest(f"{title}\0{text}")
        corpus.append(
            {
                "corpus_document_id": f"M{digest[:16]}",
                "title": title,
                "text": text,
            }
        )
    return corpus, {key: index for index, key in enumerate(keys)}


class BM25Index:
    def __init__(self, corpus: list[dict[str, str]], k1: float = 1.2, b: float = 0.75):
        self.corpus = corpus
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            stop_words="english",
            dtype=np.float32,
        )
        texts = [f"{document['title']} {document['text']}" for document in corpus]
        counts = self.vectorizer.fit_transform(texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        self.average_document_length = float(lengths.mean())
        document_frequency = np.diff(counts.tocsc().indptr)
        inverse_frequency = np.log(
            1.0
            + (counts.shape[0] - document_frequency + 0.5)
            / (document_frequency + 0.5)
        ).astype(np.float32)
        row_indices = np.repeat(
            np.arange(counts.shape[0], dtype=np.int64), np.diff(counts.indptr)
        )
        term_frequency = counts.data.copy()
        length_norm = k1 * (1.0 - b + b * lengths[row_indices] / lengths.mean())
        counts.data = term_frequency * (k1 + 1.0) / (term_frequency + length_norm)
        self.matrix = counts.multiply(inverse_frequency).tocsr()
        self.k1 = k1
        self.b = b

    def rank(self, query: str, depth: int) -> tuple[list[int], dict[int, float]]:
        query_vector = self.vectorizer.transform([query])
        query_vector.data[:] = 1.0
        scores = (self.matrix @ query_vector.T).tocoo()
        order = np.lexsort((scores.row, -scores.data))
        indices = scores.row[order[:depth]].astype(int).tolist()
        score_map = {
            int(scores.row[position]): float(scores.data[position])
            for position in order[:depth]
        }
        if len(indices) < depth:
            selected = set(indices)
            for index in range(len(self.corpus)):
                if index not in selected:
                    indices.append(index)
                    score_map[index] = 0.0
                if len(indices) == depth:
                    break
        return indices, score_map


def build_title_map(corpus: list[dict[str, str]]) -> dict[tuple[str, ...], list[int]]:
    result: dict[tuple[str, ...], list[int]] = {}
    for index, document in enumerate(corpus):
        result.setdefault(tuple(tokens(document["title"])), []).append(index)
    return result


def linked_documents(
    source_index: int,
    corpus: list[dict[str, str]],
    title_map: dict[tuple[str, ...], list[int]],
) -> list[int]:
    words = tokens(corpus[source_index]["text"])
    candidates: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for start in range(len(words)):
        for length in range(1, min(8, len(words) - start) + 1):
            phrase = tuple(words[start : start + length])
            if length == 1 and (
                len(phrase[0]) < 5 or phrase[0] in ENGLISH_STOP_WORDS
            ):
                continue
            for candidate in title_map.get(phrase, []):
                if candidate == source_index or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append((start, -length, candidate))
    return [candidate for _, _, candidate in sorted(candidates)]


def hybrid_top20(
    initial: list[int],
    corpus: list[dict[str, str]],
    title_map: dict[tuple[str, ...], list[int]],
) -> tuple[list[int], dict[int, str]]:
    chosen: list[int] = []
    provenance: dict[int, str] = {}
    selected: set[int] = set()

    def add(index: int, source: str) -> None:
        if index not in selected:
            selected.add(index)
            chosen.append(index)
            provenance[index] = source

    for index in initial[:15]:
        add(index, "bm25")
    for candidate in linked_documents(initial[0], corpus, title_map):
        add(candidate, "title_link_from_rank_1")
        if len(chosen) >= 20:
            break
    for index in initial:
        if len(chosen) >= 20:
            break
        add(index, "bm25")
    if len(chosen) != 20:
        raise RuntimeError("Failed to reconstruct hybrid top-20")
    return chosen, provenance


def percentile_summary(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "max": int(array.max()),
    }


def main() -> None:
    args = parse_args()
    depths = tuple(int(item) for item in args.depths.split(",") if item)
    if depths != tuple(sorted(set(depths))) or depths[0] != 20:
        raise SystemExit("Depths must be sorted, unique, and begin at 20")

    from datasets import load_dataset

    dataset = load_dataset("bdsaglam/musique", "answerable")
    corpus, key_to_index = build_corpus(dataset)
    index = BM25Index(corpus)
    title_map = build_title_map(corpus)
    corpus_id_to_index = {
        document["corpus_document_id"]: offset
        for offset, document in enumerate(corpus)
    }
    source_rows = {str(row["id"]): row for row in dataset["validation"]}
    base_rows = load_jsonl(args.base_data)
    if len(base_rows) != 400:
        raise RuntimeError(f"Expected 400 base rows, found {len(base_rows)}")

    tokenizers: dict[str, Any] = {}
    if not args.no_token_counts:
        from transformers import AutoTokenizer

        for label, model_id in TOKENIZER_MODELS.items():
            tokenizers[label] = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    frozen_rows: list[dict[str, Any]] = []
    exact_prefixes = 0
    final_answer_matches = 0
    token_values: dict[str, dict[int, list[int]]] = {
        label: {depth: [] for depth in depths} for label in tokenizers
    }
    support_availability = {depth: [] for depth in depths}
    for base in sorted(base_rows, key=lambda row: int(row["split_rank"])):
        example_id = str(base["example_id"])
        source = source_rows[example_id]
        initial, initial_scores = index.rank(str(base["question"]), max(depths))
        initial_rank = {corpus_index: rank for rank, corpus_index in enumerate(initial, 1)}
        top20, provenance = hybrid_top20(initial, corpus, title_map)
        expected = [str(item["corpus_document_id"]) for item in base["documents"]]
        observed = [corpus[item]["corpus_document_id"] for item in top20]
        if observed != expected:
            raise RuntimeError(f"Hybrid top-20 mismatch for {example_id}")
        exact_prefixes += 1
        selected = set(top20)
        extended = top20 + [item for item in initial if item not in selected]
        extended = extended[: max(depths)]
        if len(extended) != max(depths):
            raise RuntimeError(f"Short extended ranking for {example_id}")

        by_paragraph_index = {int(item["idx"]): item for item in source["paragraphs"]}
        decomposition = list(source["question_decomposition"])
        final_answers = {
            normalize_answer(str(source["answer"])),
            *(normalize_answer(str(item)) for item in source.get("answer_aliases", [])),
        }
        if normalize_answer(str(decomposition[-1]["answer"])) not in final_answers:
            raise RuntimeError(f"Final decomposition answer mismatch for {example_id}")
        final_answer_matches += 1
        role_corpus_ids = []
        for step in decomposition:
            paragraph = by_paragraph_index[int(step["paragraph_support_idx"])]
            role_corpus_ids.append(
                corpus[key_to_index[paragraph_key(paragraph)]]["corpus_document_id"]
            )
        first_hop_corpus_id = role_corpus_ids[0]
        terminal_hop_corpus_id = role_corpus_ids[-1]
        support_corpus_ids = set(role_corpus_ids)

        documents = []
        for rank, corpus_index in enumerate(extended, 1):
            document = corpus[corpus_index]
            corpus_id = document["corpus_document_id"]
            if rank <= 20:
                original = base["documents"][rank - 1]
                if (
                    original["title"] != document["title"]
                    or original["text"] != document["text"]
                ):
                    raise RuntimeError(f"Base text mismatch for {example_id} D{rank}")
            roles = []
            if corpus_id == first_hop_corpus_id:
                roles.append("first")
            if corpus_id == terminal_hop_corpus_id:
                roles.append("terminal")
            documents.append(
                {
                    "document_id": f"D{rank}",
                    "corpus_document_id": corpus_id,
                    "retrieval_rank": rank,
                    "retrieval_source": provenance.get(corpus_index, "bm25_extension"),
                    "initial_bm25_rank": initial_rank.get(corpus_index),
                    "initial_bm25_score": float(initial_scores.get(corpus_index, 0.0)),
                    "title": document["title"],
                    "text": document["text"],
                    "is_support": corpus_id in support_corpus_ids,
                    "support_roles": roles,
                }
            )

        row = {
            "example_id": example_id,
            "source_split": "validation",
            "split": "evaluation",
            "split_rank": int(base["split_rank"]),
            "question": str(base["question"]),
            "answer": str(base["answer"]),
            "answer_aliases": list(base.get("answer_aliases", [])),
            "depths": list(depths),
            "documents": documents,
            "support_corpus_document_ids": sorted(support_corpus_ids),
            "first_hop_corpus_document_id": first_hop_corpus_id,
            "terminal_hop_corpus_document_id": terminal_hop_corpus_id,
            "prompt_tokens_by_model_and_depth": {},
        }
        for depth in depths:
            support_ids = [
                item["document_id"] for item in documents[:depth] if item["is_support"]
            ]
            row.setdefault("retrieved_support_document_ids_by_depth", {})[str(depth)] = support_ids
            support_availability[depth].append(len(support_ids) / 2)
            if tokenizers:
                prefix_row = dict(row)
                prefix_row["documents"] = documents[:depth]
                for label, tokenizer in tokenizers.items():
                    prompt, _ = render_prompt(tokenizer, prefix_row)
                    count = len(tokenizer.encode(prompt, add_special_tokens=False))
                    row["prompt_tokens_by_model_and_depth"].setdefault(label, {})[
                        str(depth)
                    ] = count
                    token_values[label][depth].append(count)
        frozen_rows.append(row)

    write_jsonl(args.output, frozen_rows)
    summary = {
        "source_dataset": "bdsaglam/musique:answerable",
        "dataset_fingerprints": {
            split: dataset[split]._fingerprint for split in ("train", "validation")
        },
        "corpus_documents": len(corpus),
        "base_data": portable_path(args.base_data),
        "selected_count": len(frozen_rows),
        "exact_hybrid20_prefixes": exact_prefixes,
        "final_decomposition_answer_matches": final_answer_matches,
        "depths": list(depths),
        "retrieval_extension": "frozen hybrid top-20 followed by unused BM25 ranks",
        "labels_used_for_retrieval_or_ordering": False,
        "support_recall_available_by_depth": {
            str(depth): float(np.mean(values))
            for depth, values in support_availability.items()
        },
        "prompt_token_distributions": {
            label: {
                str(depth): percentile_summary(values)
                for depth, values in by_depth.items()
            }
            for label, by_depth in token_values.items()
        },
        "output": portable_path(args.output),
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
