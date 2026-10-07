from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
disable_progress_bar()


DIRECTION_TITLE = "Direction 36: Evidence-routing regimes in language models"
SYSTEM_PROMPT = (
    "Select the two source-fact labels needed to answer the question. "
    "Return only the two labels in alphabetical order, like A,B."
)

FACT_LABELS = list("ABCDEF")
WINDOW_CANDIDATES = ["last_4", "last_8", "last_third", "last_half", "all_layers"]

# Historical wording variants are retained here only for experiment reproducibility.
# They are not the recommended paper-facing nomenclature for Direction 36.
PROMPT_FAMILY_SPECS: Dict[str, Dict[str, Dict[str, str]]] = {
    "pack_a": {
        "support_set": {
            "frame": "Exactly two source facts are sufficient for the question.",
            "query": "Which two fact labels are sufficient to answer the question?",
        },
        "working_set": {
            "frame": "Treat the useful evidence as a small working set selected from the candidate source facts.",
            "query": "Which two fact labels form the working set for this question?",
        },
        "active_set": {
            "frame": "Treat the useful evidence as an active set selected from the candidate source facts.",
            "query": "Which two fact labels are the active set for this question?",
        },
        "bridge_facts": {
            "frame": "The answer should be reachable through a two-fact bridge in the candidate source facts.",
            "query": "Which two fact labels jointly bridge the question to its answer?",
        },
    },
    "pack_b": {
        "support_set": {
            "frame": "There are exactly two source facts you need; the other facts are distractors.",
            "query": "Which two fact labels are enough to answer the question?",
        },
        "working_set": {
            "frame": "Only a two-fact working set should survive from the larger candidate pool.",
            "query": "Which two fact labels should enter the working set for this question?",
        },
        "active_set": {
            "frame": "Only two facts belong in the active set that resolves the question.",
            "query": "Which two fact labels belong to the active set for this question?",
        },
        "bridge_facts": {
            "frame": "The answer should come from combining one short bridge of two facts.",
            "query": "Which two fact labels make the required bridge?",
        },
    },
}

ITERATION_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "pack": "pack_a",
        "max_examples": 8,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "scan_windows": True,
        "window_name": None,
        "window_source_iteration": None,
    },
    2: {
        "pack": "pack_a",
        "max_examples": 20,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "scan_windows": False,
        "window_name": None,
        "window_source_iteration": 1,
    },
    3: {
        "pack": "pack_b",
        "max_examples": 20,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "scan_windows": False,
        "window_name": None,
        "window_source_iteration": 1,
    },
    4: {
        "pack": "pack_a",
        "max_examples": 20,
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "scan_windows": False,
        "window_name": None,
        "window_source_iteration": 1,
    },
    # --- Phase A (Writer-Readout Gap pivot, iterations 05+) ---
    5: {
        "pack": "pack_a",
        "max_examples": 20,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "scan_windows": True,
        "window_name": None,
        "window_source_iteration": None,
    },
    6: {
        "pack": "pack_a",
        "max_examples": 20,
        "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
        "scan_windows": True,
        "window_name": None,
        "window_source_iteration": None,
    },
}

# --- Phase C (Spotlight expansion, iteration 08+) ---
# Phase C.1: HotpotQA expansion. Model is specified via --model-id CLI override.
ITERATION_CONFIGS[8] = {
    "pack": "both",  # run both pack_a and pack_b
    "max_examples": 50,
    "model_id": None,  # MUST be overridden via --model-id
    "scan_windows": False,
    "window_name": None,
    "window_source_iteration": "per_model",
    "headline_families": ["active_set", "working_set"],
}

# Phase C.2: FEVER dataset. Model via --model-id CLI override.
ITERATION_CONFIGS[9] = {
    "pack": "both",
    "max_examples": 50,
    "model_id": None,  # MUST be overridden via --model-id
    "scan_windows": False,
    "window_name": None,
    "window_source_iteration": "per_model",
    "headline_families": ["active_set", "working_set"],
    "prompt_file": "fever_prompts.jsonl",  # Use FEVER prompts instead of HotpotQA
}

# Phase A window selections per model (locked, do not redesign)
PHASE_A_WINDOWS: Dict[str, str] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "last_8",
    "Qwen/Qwen2.5-3B-Instruct": "last_8",
    "Qwen/Qwen2.5-7B-Instruct": "last_4",
    "NousResearch/Hermes-3-Llama-3.1-8B": "last_8",
    "allenai/OLMo-2-1124-7B-Instruct": "last_8",  # OLMo-2 (32 layers, same depth as Llama)
}

# Phase B intervention configs
PHASE_B_CONFIG = {
    "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
    "pack": "pack_a",
    "max_examples": 20,
    "window_name": "last_8",
    "baseline_iteration": 6,
    "amplification_factors": [1.5, 2.0, 3.0],
}

PHASE_B_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llama_8b": PHASE_B_CONFIG,
    "qwen_1_5b": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "pack": "both",
        "max_examples": 50,
        "window_name": "last_8",
        "baseline_iteration": "08_qwen2_5_1_5b_instruct",  # iteration 8 with model slug
        "amplification_factors": [1.5, 2.0, 3.0],
    },
}


@dataclass
class EncodedStepPrompt:
    row: Dict[str, Any]
    prompt_text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    target_pos: int
    fact_positions: Dict[str, List[int]]
    fact_labels: List[str]
    target_label: str
    label_token_ids: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direction 36 custom experiment runner.")
    parser.add_argument("--workspace", required=True, help="Direction workspace path")
    parser.add_argument("--repo-root", default=None, help="Repository root path")
    parser.add_argument("--iteration", type=int, default=None, help="Iteration index for formal runs")
    parser.add_argument(
        "--prepare-prompts",
        action="store_true",
        help="Generate HotpotQA-based prompts into prompts.jsonl instead of running an iteration.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum number of base HotpotQA examples to keep when preparing prompts.",
    )
    parser.add_argument(
        "--phase-b",
        action="store_true",
        help="Run Phase B causal interventions instead of a standard iteration.",
    )
    parser.add_argument(
        "--prepare-fever-prompts",
        action="store_true",
        help="Generate FEVER-based prompts into fever_prompts.jsonl.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the model_id from ITERATION_CONFIGS (used for multi-model iterations like C.1).",
    )
    parser.add_argument(
        "--phase-b-config",
        type=str,
        default=None,
        help="Name of the Phase B config to use (e.g., 'qwen_1_5b'). Default uses PHASE_B_CONFIG.",
    )
    return parser.parse_args()


def infer_repo_root(workspace: Path, repo_root_arg: str | None) -> Path:
    if repo_root_arg:
        return Path(repo_root_arg).resolve()
    current = workspace.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "research_ideas.md").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError(f"Could not infer repo root from workspace={workspace}")


def load_plan(workspace: Path) -> Dict[str, Any]:
    return yaml.safe_load((workspace / "plan.yaml").read_text(encoding="utf-8"))


def ensure_repo_imports(repo_root: Path) -> None:
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    src_text = str(repo_root / "src")
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def jsonl_load(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_shuffle(items: Sequence[Any], seed_text: str) -> List[Any]:
    rng_seed = int.from_bytes(
        hashlib.blake2b(seed_text.encode("utf-8"), digest_size=8).digest(), "big"
    )
    rng = __import__("random").Random(rng_seed)
    out = list(items)
    rng.shuffle(out)
    return out


def compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids


def fact_line(fact: Dict[str, Any]) -> str:
    return f"{fact['label']}. [{fact['title']}] {fact['text']}"


def render_user_prompt(question: str, facts: Sequence[Dict[str, Any]], family_spec: Dict[str, str]) -> str:
    lines = [
        f"Question: {question.strip()}",
        family_spec["frame"],
        "Candidate source facts:",
        *(fact_line(fact) for fact in facts),
        family_spec["query"],
        "Return only the two labels in alphabetical order, like A,B.",
    ]
    return "\n".join(lines)


def dedup_support_pairs(row: Dict[str, Any]) -> List[Tuple[str, int]]:
    unique: List[Tuple[str, int]] = []
    seen = set()
    for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        key = (str(title), int(sent_id))
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def choose_distractors(
    *,
    support_pairs: Sequence[Tuple[str, int]],
    title_to_sentences: Dict[str, Sequence[str]],
) -> List[Tuple[str, int, str]]:
    support_titles = {title for title, _ in support_pairs}
    same_title: List[Tuple[str, int, str]] = []
    cross_title: List[Tuple[str, int, str]] = []

    support_set = set(support_pairs)
    for title, sentences in title_to_sentences.items():
        for sent_id, sentence in enumerate(sentences):
            key = (title, sent_id)
            text = str(sentence).strip()
            if key in support_set or len(text.split()) < 6:
                continue
            target_list = same_title if title in support_titles else cross_title
            target_list.append((title, sent_id, text))

    picked = cross_title[:3]
    if same_title:
        picked.extend(same_title[:1])
    if len(picked) < 4:
        pool = cross_title[3:] + same_title[1:]
        picked.extend(pool[: 4 - len(picked)])
    return picked[:4]


def _generate_from_split(
    *,
    split_name: str,
    tokenizer,
    max_examples: int,
    max_length: int,
    start_rank: int,
    seen_ids: set,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, set]:
    """Generate prompt rows from a single HotpotQA split.

    Returns (rows, audits, kept_count, updated_seen_ids).
    """
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)

    rows: List[Dict[str, Any]] = []
    kept_examples = 0
    audits: List[Dict[str, Any]] = []

    for item in dataset:
        if str(item["id"]) in seen_ids:
            continue
        support_pairs = dedup_support_pairs(item)
        if len(support_pairs) != 2:
            continue

        title_to_sentences = {
            str(title): [str(sentence) for sentence in sentences]
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"])
        }

        if not all(title in title_to_sentences and sent_id < len(title_to_sentences[title]) for title, sent_id in support_pairs):
            continue

        distractors = choose_distractors(
            support_pairs=support_pairs,
            title_to_sentences=title_to_sentences,
        )
        if len(distractors) < 4:
            continue

        fact_entries = [
            {
                "title": title,
                "sent_id": sent_id,
                "text": title_to_sentences[title][sent_id].strip(),
                "is_support": True,
            }
            for title, sent_id in support_pairs
        ] + [
            {
                "title": title,
                "sent_id": sent_id,
                "text": text,
                "is_support": False,
            }
            for title, sent_id, text in distractors
        ]

        fact_entries = stable_shuffle(fact_entries, f"{item['id']}::facts")
        for label, fact in zip(FACT_LABELS, fact_entries):
            fact["label"] = label

        support_labels = sorted([fact["label"] for fact in fact_entries if fact["is_support"]])
        if len(support_labels) != 2:
            continue

        base_rows: List[Dict[str, Any]] = []
        prompt_lengths: Dict[str, int] = {}
        valid_example = True

        for pack_name, family_specs in PROMPT_FAMILY_SPECS.items():
            for family_name, family_spec in family_specs.items():
                user_prompt = render_user_prompt(
                    question=str(item["question"]).strip(),
                    facts=fact_entries,
                    family_spec=family_spec,
                )
                chat_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_length = len(tokenizer.encode(chat_text, add_special_tokens=False))
                if prompt_length > max_length:
                    valid_example = False
                    break

                prompt_family = f"{family_name}_{pack_name}"
                prompt_lengths[prompt_family] = prompt_length
                base_rows.append(
                    {
                        "prompt_id": f"{item['id']}::{prompt_family}",
                        "example_id": str(item["id"]),
                        "example_rank": start_rank + kept_examples + 1,
                        "source_split": split_name,
                        "prompt_family": prompt_family,
                        "family_name": family_name,
                        "lexical_pack": pack_name,
                        "question": str(item["question"]).strip(),
                        "answer": str(item["answer"]).strip(),
                        "support_labels": support_labels,
                        "facts": fact_entries,
                        "prompt": chat_text,
                        "notes": "HotpotQA distractor example rendered into a labeled support-fact selection prompt for Direction 36.",
                    }
                )
            if not valid_example:
                break

        if not valid_example:
            continue

        rows.extend(base_rows)
        kept_examples += 1
        audits.append(
            {
                "example_id": str(item["id"]),
                "support_labels": support_labels,
                "prompt_lengths": prompt_lengths,
            }
        )
        seen_ids.add(str(item["id"]))
        if kept_examples >= max_examples:
            break

    return rows, audits, kept_examples, seen_ids


def generate_prompt_rows(
    *,
    tokenizer,
    max_examples: int,
    max_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate prompt rows, drawing from validation first then training if needed."""
    # Phase A used up to 20 examples from validation.
    # Phase C expansion (max_examples > 20) draws the first 20 from validation,
    # then fills the remainder from the training split.
    val_target = min(max_examples, 20)
    rows_val, audits_val, kept_val, seen_ids = _generate_from_split(
        split_name="validation",
        tokenizer=tokenizer,
        max_examples=val_target,
        max_length=max_length,
        start_rank=0,
        seen_ids=set(),
    )
    if kept_val < val_target:
        raise RuntimeError(f"Only generated {kept_val} validation examples, expected {val_target}.")

    all_rows = list(rows_val)
    all_audits = list(audits_val)
    total_kept = kept_val

    if max_examples > val_target:
        train_target = max_examples - val_target
        rows_train, audits_train, kept_train, _ = _generate_from_split(
            split_name="train",
            tokenizer=tokenizer,
            max_examples=train_target,
            max_length=max_length,
            start_rank=total_kept,
            seen_ids=seen_ids,
        )
        if kept_train < train_target:
            raise RuntimeError(
                f"Only generated {kept_train} training examples, expected {train_target}. "
                f"Total: {total_kept + kept_train}/{max_examples}."
            )
        all_rows.extend(rows_train)
        all_audits.extend(audits_train)
        total_kept += kept_train

    summary = {
        "base_example_count": total_kept,
        "prompt_count": len(all_rows),
        "prompt_families": sorted({row["prompt_family"] for row in all_rows}),
        "lexical_packs": sorted(PROMPT_FAMILY_SPECS.keys()),
        "max_prompt_length": max(
            max(example["prompt_lengths"].values()) for example in all_audits
        ),
        "splits_used": ["validation"] if max_examples <= 20 else ["validation", "train"],
        "validation_examples": kept_val,
        "training_examples": total_kept - kept_val,
        "audits": all_audits,
    }
    return all_rows, summary


def prepare_prompts(workspace: Path, repo_root: Path, max_examples: int) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    plan = load_plan(workspace)
    max_length = int(plan["experiment"].get("max_length", 768))
    model_id = str(plan["experiment"]["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    rows, summary = generate_prompt_rows(
        tokenizer=tokenizer,
        max_examples=max_examples,
        max_length=max_length,
    )
    prompt_path = Path(os.environ.get("D36_PREPARE_PROMPTS_OUT", str(workspace / "prompts.jsonl")))
    summary_path = Path(
        os.environ.get(
            "D36_PREPARE_PROMPTS_SUMMARY",
            str(workspace / "custom_outputs" / "prompt_generation_summary.json"),
        )
    )
    jsonl_write(prompt_path, rows)
    save_json(summary_path, summary)
    return {
        "workspace": str(workspace),
        "prompt_file": str(prompt_path),
        "summary_file": str(summary_path),
        "prompt_count": summary["prompt_count"],
        "base_example_count": summary["base_example_count"],
        "prompt_families": summary["prompt_families"],
    }


# --- FEVER prompt generation for Task C.2 ---

FEVER_SYSTEM_PROMPT = (
    "Select the two evidence labels that support or refute the claim. "
    "Return only the two labels in alphabetical order, like A,B."
)

FEVER_FAMILY_SPECS: Dict[str, Dict[str, Dict[str, str]]] = {
    "pack_a": {
        "active_set": {
            "frame": "Exactly two evidence sentences are relevant to verifying this claim.",
            "query": "Which two evidence labels verify the claim?",
        },
        "working_set": {
            "frame": "Treat the relevant evidence as a small working set selected from the candidate sentences.",
            "query": "Which two evidence labels form the working set for this claim?",
        },
    },
    "pack_b": {
        "active_set": {
            "frame": "There are exactly two evidence sentences you need; the other sentences are distractors.",
            "query": "Which two evidence labels are enough to verify the claim?",
        },
        "working_set": {
            "frame": "Only a two-sentence working set should survive from the larger candidate pool.",
            "query": "Which two evidence labels should enter the working set for this claim?",
        },
    },
}


def render_fever_user_prompt(claim: str, facts: Sequence[Dict[str, Any]], family_spec: Dict[str, str]) -> str:
    lines = [
        f"Claim: {claim.strip()}",
        family_spec["frame"],
        "Candidate evidence sentences:",
        *(fact_line(fact) for fact in facts),
        family_spec["query"],
        "Return only the two labels in alphabetical order, like A,B.",
    ]
    return "\n".join(lines)


def generate_fever_prompt_rows(
    *,
    tokenizer,
    max_examples: int,
    max_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate FEVER-based prompts for Task C.2."""
    ds = load_dataset("copenlu/fever_gold_evidence", split="train")

    # Build a distractor pool: evidence sentences from single-evidence claims
    distractor_pool: List[Dict[str, Any]] = []
    for item in ds:
        if len(item["evidence"]) == 1:
            ev = item["evidence"][0]
            text = str(ev[2]).strip()
            if len(text.split()) >= 8:
                distractor_pool.append({
                    "title": str(ev[0]),
                    "sent_id": int(ev[1]),
                    "text": text,
                })
        if len(distractor_pool) >= 10000:
            break

    rows: List[Dict[str, Any]] = []
    kept_examples = 0
    audits: List[Dict[str, Any]] = []

    for item in ds:
        if item["label"] not in ("SUPPORTS", "REFUTES"):
            continue
        if len(item["evidence"]) != 2:
            continue

        ev_pages = [str(ev[0]) for ev in item["evidence"]]
        if len(set(ev_pages)) != 2:
            continue

        ev_texts = [str(ev[2]).strip() for ev in item["evidence"]]
        if any(len(t.split()) < 8 for t in ev_texts):
            continue

        gold_entries = [
            {
                "title": str(ev[0]),
                "sent_id": int(ev[1]),
                "text": str(ev[2]).strip(),
                "is_support": True,
            }
            for ev in item["evidence"]
        ]

        # Pick 4 distractors from the pool, avoiding gold pages
        gold_page_set = set(ev_pages)
        candidates = [d for d in distractor_pool if d["title"] not in gold_page_set]
        if len(candidates) < 4:
            continue
        picked_distractors = stable_shuffle(candidates, f"fever_{item['id']}::distractors")[:4]

        distractor_entries = [
            {
                "title": d["title"],
                "sent_id": d["sent_id"],
                "text": d["text"],
                "is_support": False,
            }
            for d in picked_distractors
        ]

        fact_entries = gold_entries + distractor_entries
        fact_entries = stable_shuffle(fact_entries, f"fever_{item['id']}::facts")
        for label, fact in zip(FACT_LABELS, fact_entries):
            fact["label"] = label

        support_labels = sorted([f["label"] for f in fact_entries if f["is_support"]])
        if len(support_labels) != 2:
            continue

        claim_text = str(item["claim"]).strip()
        base_rows: List[Dict[str, Any]] = []
        prompt_lengths: Dict[str, int] = {}
        valid_example = True

        for pack_name, family_specs in FEVER_FAMILY_SPECS.items():
            for family_name, family_spec in family_specs.items():
                user_prompt = render_fever_user_prompt(
                    claim=claim_text,
                    facts=fact_entries,
                    family_spec=family_spec,
                )
                chat_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": FEVER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_length = len(tokenizer.encode(chat_text, add_special_tokens=False))
                if prompt_length > max_length:
                    valid_example = False
                    break

                prompt_family = f"{family_name}_{pack_name}"
                prompt_lengths[prompt_family] = prompt_length
                base_rows.append(
                    {
                        "prompt_id": f"fever_{item['id']}::{prompt_family}",
                        "example_id": f"fever_{item['id']}",
                        "example_rank": kept_examples + 1,
                        "source_split": "fever_train",
                        "prompt_family": prompt_family,
                        "family_name": family_name,
                        "lexical_pack": pack_name,
                        "question": claim_text,
                        "answer": str(item["label"]),
                        "support_labels": support_labels,
                        "facts": fact_entries,
                        "prompt": chat_text,
                        "dataset": "fever",
                        "fever_label": str(item["label"]),
                        "notes": "FEVER claim with gold evidence sentences rendered into a labeled evidence-selection prompt for direction 36 Task C.2.",
                    }
                )
            if not valid_example:
                break

        if not valid_example:
            continue

        rows.extend(base_rows)
        kept_examples += 1
        audits.append(
            {
                "example_id": f"fever_{item['id']}",
                "support_labels": support_labels,
                "fever_label": str(item["label"]),
                "prompt_lengths": prompt_lengths,
            }
        )
        if kept_examples >= max_examples:
            break

    if kept_examples < max_examples:
        raise RuntimeError(f"FEVER: only generated {kept_examples} examples, expected {max_examples}.")

    summary = {
        "dataset": "fever",
        "base_example_count": kept_examples,
        "prompt_count": len(rows),
        "prompt_families": sorted({row["prompt_family"] for row in rows}),
        "lexical_packs": sorted(FEVER_FAMILY_SPECS.keys()),
        "max_prompt_length": max(
            max(ex["prompt_lengths"].values()) for ex in audits
        ),
        "label_distribution": {
            "SUPPORTS": sum(1 for a in audits if a["fever_label"] == "SUPPORTS"),
            "REFUTES": sum(1 for a in audits if a["fever_label"] == "REFUTES"),
        },
        "audits": audits,
    }
    return rows, summary


def prepare_fever_prompts(workspace: Path, repo_root: Path, max_examples: int) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    plan = load_plan(workspace)
    max_length = int(plan["experiment"].get("max_length", 768))
    model_id = str(plan["experiment"]["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    rows, summary = generate_fever_prompt_rows(
        tokenizer=tokenizer,
        max_examples=max_examples,
        max_length=max_length,
    )
    fever_prompts_path = workspace / "fever_prompts.jsonl"
    jsonl_write(fever_prompts_path, rows)
    save_json(workspace / "custom_outputs" / "fever_prompt_generation_summary.json", summary)
    return {
        "workspace": str(workspace),
        "prompt_file": str(fever_prompts_path),
        "summary_file": str(workspace / "custom_outputs" / "fever_prompt_generation_summary.json"),
        "prompt_count": summary["prompt_count"],
        "base_example_count": summary["base_example_count"],
        "prompt_families": summary["prompt_families"],
    }


def rerender_prompt_for_model(row: Dict[str, Any], tokenizer, max_length: int) -> str | None:
    """Re-render a prompt row's chat text using a different tokenizer's chat template.

    Returns the new prompt text, or None if it exceeds max_length.
    """
    family_spec_key = row["family_name"]
    pack_key = row["lexical_pack"]
    is_fever = row.get("dataset") == "fever"

    if is_fever:
        family_spec = FEVER_FAMILY_SPECS[pack_key][family_spec_key]
        user_prompt = render_fever_user_prompt(
            claim=str(row["question"]).strip(),
            facts=row["facts"],
            family_spec=family_spec,
        )
        system_prompt = FEVER_SYSTEM_PROMPT
    else:
        family_spec = PROMPT_FAMILY_SPECS[pack_key][family_spec_key]
        user_prompt = render_user_prompt(
            question=str(row["question"]).strip(),
            facts=row["facts"],
            family_spec=family_spec,
        )
        system_prompt = SYSTEM_PROMPT

    chat_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_length = len(tokenizer.encode(chat_text, add_special_tokens=False))
    if prompt_length > max_length:
        return None
    return chat_text


def prompt_with_prefix(base_prompt: str, answer_prefix: str) -> str:
    return base_prompt + answer_prefix


def token_positions_for_spans(tokenizer, prompt: str, spans: Dict[str, Tuple[int, int]]) -> Dict[str, List[int]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    out: Dict[str, List[int]] = {}
    for name, (span_start, span_end) in spans.items():
        positions = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if int(tok_start) < span_end and int(tok_end) > span_start
        ]
        if not positions:
            raise RuntimeError(f"No token positions found for span {name}.")
        out[name] = positions
    return out


def resolve_label_token_ids(tokenizer, labels: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label in labels:
        token_id = None
        for text in [label, f" {label}", f"\n{label}"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                token_id = int(ids[0])
                if text == label:
                    break
        if token_id is None:
            raise RuntimeError(f"Could not resolve single-token id for label={label}")
        out[label] = token_id
    return out


def encode_step_prompt(
    *,
    row: Dict[str, Any],
    tokenizer,
    device: torch.device,
    target_label: str,
    answer_prefix: str,
) -> EncodedStepPrompt:
    prompt_text = prompt_with_prefix(str(row["prompt"]), answer_prefix)
    spans: Dict[str, Tuple[int, int]] = {}
    for fact in row["facts"]:
        line = fact_line(fact)
        start = prompt_text.index(line)
        spans[str(fact["label"])] = (start, start + len(line))

    fact_positions = token_positions_for_spans(tokenizer, prompt_text, spans)
    tokenized = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(device)
    position_ids = compute_position_ids(attention_mask).to(device)
    label_token_ids = resolve_label_token_ids(tokenizer, FACT_LABELS)
    return EncodedStepPrompt(
        row=row,
        prompt_text=prompt_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        target_pos=int(input_ids.shape[1] - 1),
        fact_positions=fact_positions,
        fact_labels=[str(fact["label"]) for fact in row["facts"]],
        target_label=target_label,
        label_token_ids=label_token_ids,
    )


def get_label_logits(logits: torch.Tensor, label_token_ids: Dict[str, int]) -> Dict[str, float]:
    return {label: float(logits[token_id].item()) for label, token_id in label_token_ids.items()}


def get_window_layers(num_layers: int, window_name: str) -> List[int]:
    if window_name == "last_4":
        start = max(0, num_layers - 4)
    elif window_name == "last_8":
        start = max(0, num_layers - 8)
    elif window_name == "last_third":
        start = max(0, math.floor(num_layers * 2 / 3))
    elif window_name == "last_half":
        start = max(0, math.floor(num_layers / 2))
    elif window_name == "all_layers":
        start = 0
    else:
        raise ValueError(f"Unknown window name: {window_name}")
    return list(range(start, num_layers))


def parse_json_stdout(stdout: str) -> Dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        raise ValueError("Empty stdout")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.rfind("{")
        if start < 0:
            raise
        return json.loads(stripped[start:])


def run_representative_decomposition(
    *,
    repo_root: Path,
    model_id: str,
    prompt_text: str,
    iteration: int,
    tag: str,
    target_label: str,
) -> str:
    cmd = [
        sys.executable,
        str(repo_root / "src" / "run_experiment.py"),
        "--model-id",
        model_id,
        "--prompt",
        prompt_text,
        "--experiment-name",
        f"direction_36_iter_{iteration:02d}_{tag}",
        "--run-id",
        f"direction36_{iteration:02d}_{tag}",
        "--target-token-text",
        target_label,
        "--dtype",
        "auto",
        "--device",
        "auto",
        "--top-k-next-tokens",
        "8",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Representative decomposition failed tag={tag} returncode={proc.returncode}\n"
            f"stdout={proc.stdout[-1500:]}\n"
            f"stderr={proc.stderr[-1500:]}"
        )
    parsed = parse_json_stdout(proc.stdout)
    return str(parsed["run_dir"])


def greedy_decode_two_labels(model, tokenizer, row: Dict[str, Any], device: torch.device) -> List[str]:
    label_token_ids = resolve_label_token_ids(tokenizer, FACT_LABELS)
    prediction: List[str] = []
    prefix = ""

    for _step in range(2):
        prompt_text = prompt_with_prefix(str(row["prompt"]), prefix)
        tokenized = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(device)
        position_ids = compute_position_ids(attention_mask).to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.logits[0, input_ids.shape[1] - 1].float().detach()
        label_logits = get_label_logits(logits, label_token_ids)
        predicted = max(label_logits.items(), key=lambda item: item[1])[0]
        prediction.append(predicted)
        prefix = "".join([prediction[0], ","]) if len(prediction) == 1 else prefix
    return prediction


def analyze_step_prompt(
    *,
    model,
    adapter,
    base_model,
    layers,
    loaded_device: torch.device,
    encoded: EncodedStepPrompt,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with torch.no_grad():
        outputs = model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            position_ids=encoded.position_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if outputs.hidden_states is None or outputs.attentions is None:
        raise RuntimeError("Model did not return hidden_states and attentions.")

    logits = outputs.logits[0, encoded.target_pos].float().detach()
    label_logits = get_label_logits(logits, encoded.label_token_ids)
    predicted_label = max(label_logits.items(), key=lambda item: item[1])[0]
    target_token_id = encoded.label_token_ids[encoded.target_label]

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Model has no output embeddings.")
    unembed_weight = output_embeddings.weight.detach().to(loaded_device).float()
    target_direction = unembed_weight[target_token_id]
    target_direction = target_direction / torch.clamp(target_direction.norm(), min=1e-8)

    fact_position_tensors = {
        label: torch.tensor(positions, device=loaded_device, dtype=torch.long)
        for label, positions in encoded.fact_positions.items()
    }
    support_set = set(encoded.row["support_labels"])

    records: List[Dict[str, Any]] = []

    for layer_idx, layer in enumerate(layers):
        resid_pre = outputs.hidden_states[layer_idx].detach()
        attn_input = adapter.attn_input(layer, resid_pre)
        layer_decomp = adapter.decompose_layer(layer, attn_input, encoded.position_ids, base_model=base_model)
        attn_probs = outputs.attentions[layer_idx][0].float()

        seq_len = int(encoded.input_ids.shape[1])
        layer_token_gate = torch.zeros(seq_len, device=loaded_device, dtype=torch.float32)
        layer_token_positive = torch.zeros(seq_len, device=loaded_device, dtype=torch.float32)
        layer_token_signed = torch.zeros(seq_len, device=loaded_device, dtype=torch.float32)
        layer_token_abs = torch.zeros(seq_len, device=loaded_device, dtype=torch.float32)

        q_heads = layer_decomp.q.shape[0]
        for q_head in range(q_heads):
            kv_head = adapter.q_to_kv_head(layer, q_head)
            alpha = attn_probs[q_head, encoded.target_pos, :].detach().float()
            v_src = layer_decomp.v[kv_head, :, :].detach().float()
            o_block = layer_decomp.o_weight[q_head].detach().float()
            writer = torch.matmul(v_src, o_block)
            writer_proj_target = torch.matmul(writer, target_direction)
            signed_contribution = alpha * writer_proj_target

            layer_token_gate += alpha
            layer_token_positive += signed_contribution.clamp(min=0)
            layer_token_signed += signed_contribution
            layer_token_abs += signed_contribution.abs()

        layer_gate = {}
        layer_gate_mean = {}
        layer_gate_peak = {}
        layer_positive = {}
        layer_positive_mean = {}
        layer_positive_peak = {}
        layer_signed = {}
        layer_abs_signed = {}
        total_gate = 0.0
        total_gate_peak = 0.0
        total_positive = 0.0
        total_positive_peak = 0.0
        for label in encoded.fact_labels:
            positions = fact_position_tensors[label]
            token_gate = layer_token_gate.index_select(0, positions)
            token_positive = layer_token_positive.index_select(0, positions)
            token_signed = layer_token_signed.index_select(0, positions)
            token_abs = layer_token_abs.index_select(0, positions)
            span_token_count = int(positions.numel())

            layer_gate[label] = float(token_gate.sum().item())
            layer_gate_mean[label] = float(token_gate.mean().item())
            layer_gate_peak[label] = float(token_gate.max().item())
            layer_positive[label] = float(token_positive.sum().item())
            layer_positive_mean[label] = float(token_positive.mean().item())
            layer_positive_peak[label] = float(token_positive.max().item())
            layer_signed[label] = float(token_signed.sum().item())
            layer_abs_signed[label] = float(token_abs.sum().item())
            total_gate += layer_gate[label]
            total_gate_peak += layer_gate_peak[label]
            total_positive += layer_positive[label]
            total_positive_peak += layer_positive_peak[label]

            records.append(
                {
                    "prompt_id": encoded.row["prompt_id"],
                    "example_id": encoded.row["example_id"],
                    "prompt_family": encoded.row["prompt_family"],
                    "family_name": encoded.row["family_name"],
                    "lexical_pack": encoded.row["lexical_pack"],
                    "step_index": 1 if encoded.target_label == encoded.row["support_labels"][0] else 2,
                    "target_label": encoded.target_label,
                    "predicted_label": predicted_label,
                    "layer": int(layer_idx),
                    "fact_label": label,
                    "is_support": bool(label in support_set),
                    "span_token_count": span_token_count,
                    "gate_mass": layer_gate[label],
                    "gate_mean": layer_gate_mean[label],
                    "gate_peak": layer_gate_peak[label],
                    "positive_contribution": layer_positive[label],
                    "positive_mean": layer_positive_mean[label],
                    "positive_peak": layer_positive_peak[label],
                    "signed_contribution": layer_signed[label],
                    "abs_signed_contribution": layer_abs_signed[label],
                    "gate_share": float(layer_gate[label] / total_gate) if total_gate > 0 else math.nan,
                    "gate_peak_share": float(layer_gate_peak[label] / total_gate_peak) if total_gate_peak > 0 else math.nan,
                    "positive_share": float(layer_positive[label] / total_positive) if total_positive > 0 else math.nan,
                    "positive_peak_share": float(layer_positive_peak[label] / total_positive_peak)
                    if total_positive_peak > 0
                    else math.nan,
                }
            )

    metadata = {
        "target_label": encoded.target_label,
        "predicted_label": predicted_label,
        "label_logits": label_logits,
        "num_layers": len(layers),
    }
    return records, metadata


def top_k_labels(score_map: Dict[str, float], k: int = 2) -> List[str]:
    ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    return [label for label, _ in ordered[:k]]


def compute_case_metrics(case_df: pd.DataFrame, selected_layers: Sequence[int]) -> Dict[str, Any]:
    selected_df = case_df[case_df["layer"].isin(selected_layers)].copy()
    if selected_df.empty:
        raise RuntimeError("Selected layer window produced no rows.")

    gold_support = sorted(selected_df[selected_df["is_support"]]["fact_label"].unique().tolist())
    gold_set = set(gold_support)

    gate_scores = selected_df.groupby("fact_label")["gate_peak"].sum().to_dict()
    contrib_scores = selected_df.groupby("fact_label")["positive_contribution"].sum().to_dict()
    gate_mass_scores = selected_df.groupby("fact_label")["gate_mass"].sum().to_dict()

    gate_top2 = top_k_labels(gate_scores, k=2)
    contrib_top2 = top_k_labels(contrib_scores, k=2)

    total_gate = float(sum(gate_scores.values()))
    total_gate_mass = float(sum(gate_mass_scores.values()))
    total_contrib = float(sum(contrib_scores.values()))
    gold_gate = float(sum(gate_scores[label] for label in gold_support))
    gold_gate_mass = float(sum(gate_mass_scores[label] for label in gold_support))
    gold_contrib = float(sum(contrib_scores[label] for label in gold_support))
    distractor_gate = [score for label, score in gate_scores.items() if label not in gold_set]
    distractor_contrib = [score for label, score in contrib_scores.items() if label not in gold_set]

    per_layer_gate_match: List[float] = []
    per_layer_contrib_match: List[float] = []
    for layer_idx in selected_layers:
        layer_scores = (
            selected_df[selected_df["layer"] == layer_idx]
            .groupby("fact_label")[["gate_peak", "positive_contribution"]]
            .sum()
            .to_dict("index")
        )
        layer_gate = {label: values["gate_peak"] for label, values in layer_scores.items()}
        layer_contrib = {label: values["positive_contribution"] for label, values in layer_scores.items()}
        per_layer_gate_match.append(float(set(top_k_labels(layer_gate, 2)) == gold_set))
        per_layer_contrib_match.append(float(set(top_k_labels(layer_contrib, 2)) == gold_set))

    prediction_labels = selected_df["decoded_prediction_set"].iloc[0]
    prediction_set = set(prediction_labels)
    prediction_correct = prediction_set == gold_set and len(prediction_labels) == 2

    if prediction_correct:
        failure_bucket = "correct"
    elif set(gate_top2) == gold_set:
        failure_bucket = "readout_failure"
    else:
        failure_bucket = "discovery_failure"

    # Writer-based failure taxonomy (pivot iterations 05+)
    writer_match = bool(set(contrib_top2) == gold_set)
    if prediction_correct:
        writer_failure_bucket = "correct"
    elif writer_match:
        writer_failure_bucket = "writer_readout_failure"
    else:
        writer_failure_bucket = "writer_discovery_failure"
    writer_readout_gap = float(writer_match) - float(prediction_correct)

    return {
        "prompt_id": str(selected_df["prompt_id"].iloc[0]),
        "example_id": str(selected_df["example_id"].iloc[0]),
        "prompt_family": str(selected_df["prompt_family"].iloc[0]),
        "family_name": str(selected_df["family_name"].iloc[0]),
        "lexical_pack": str(selected_df["lexical_pack"].iloc[0]),
        "gold_support_labels": gold_support,
        "decoded_prediction_set": prediction_labels,
        "prediction_set_correct": bool(prediction_correct),
        "gate_top2_labels": gate_top2,
        "contrib_top2_labels": contrib_top2,
        "gate_joint_support_match": bool(set(gate_top2) == gold_set),
        "contrib_joint_support_match": bool(set(contrib_top2) == gold_set),
        "gate_support_peak_share": float(gold_gate / total_gate) if total_gate > 0 else math.nan,
        "gate_support_raw_mass_share": float(gold_gate_mass / total_gate_mass) if total_gate_mass > 0 else math.nan,
        "contrib_support_mass_share": float(gold_contrib / total_contrib) if total_contrib > 0 else math.nan,
        "gate_support_margin": float(gold_gate / 2.0 - (sum(distractor_gate) / max(len(distractor_gate), 1))),
        "contrib_support_margin": float(
            gold_contrib / 2.0 - (sum(distractor_contrib) / max(len(distractor_contrib), 1))
        ),
        "layer_stable_gate_support_rate": float(sum(per_layer_gate_match) / len(per_layer_gate_match)),
        "layer_stable_contrib_support_rate": float(sum(per_layer_contrib_match) / len(per_layer_contrib_match)),
        "failure_bucket": failure_bucket,
        "writer_failure_bucket": writer_failure_bucket,
        "writer_readout_gap": writer_readout_gap,
    }


def summarize_case_metrics(case_df: pd.DataFrame) -> Dict[str, Any]:
    prompt_count = int(len(case_df))
    correct_count = int(case_df["prediction_set_correct"].sum())
    readout_failures = int((case_df["failure_bucket"] == "readout_failure").sum())
    discovery_failures = int((case_df["failure_bucket"] == "discovery_failure").sum())
    error_count = max(prompt_count - correct_count, 1)

    # Writer-readout gap metrics (pivot iterations 05+)
    contrib_rate = float(case_df["contrib_joint_support_match"].mean())
    pred_accuracy = float(case_df["prediction_set_correct"].mean())
    writer_readout_gap = contrib_rate - pred_accuracy
    writer_readout_failures = int((case_df["writer_failure_bucket"] == "writer_readout_failure").sum())
    writer_discovery_failures = int((case_df["writer_failure_bucket"] == "writer_discovery_failure").sum())

    return {
        "prompt_count": prompt_count,
        "prediction_set_accuracy": pred_accuracy,
        "gate_joint_support_match_rate": float(case_df["gate_joint_support_match"].mean()),
        "contrib_joint_support_match_rate": contrib_rate,
        "writer_readout_gap": writer_readout_gap,
        "mean_gate_support_peak_share": float(case_df["gate_support_peak_share"].mean()),
        "mean_contrib_support_mass_share": float(case_df["contrib_support_mass_share"].mean()),
        "mean_gate_support_margin": float(case_df["gate_support_margin"].mean()),
        "mean_contrib_support_margin": float(case_df["contrib_support_margin"].mean()),
        "mean_layer_stable_gate_support_rate": float(case_df["layer_stable_gate_support_rate"].mean()),
        "mean_layer_stable_contrib_support_rate": float(case_df["layer_stable_contrib_support_rate"].mean()),
        "correct_count": correct_count,
        "readout_failure_count": readout_failures,
        "discovery_failure_count": discovery_failures,
        "readout_failure_share_among_errors": float(readout_failures / error_count),
        "discovery_failure_share_among_errors": float(discovery_failures / error_count),
        "writer_readout_failure_count": writer_readout_failures,
        "writer_discovery_failure_count": writer_discovery_failures,
        "writer_readout_failure_share_among_errors": float(writer_readout_failures / error_count),
        "writer_discovery_failure_share_among_errors": float(writer_discovery_failures / error_count),
        "random_joint_match_baseline": float(1.0 / 15.0),
    }


def select_window_from_scan(layer_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    num_layers = int(layer_df["layer"].max()) + 1
    case_records: List[Dict[str, Any]] = []
    for window_name in WINDOW_CANDIDATES:
        selected_layers = get_window_layers(num_layers, window_name)
        case_rows = []
        for prompt_id, prompt_df in layer_df.groupby("prompt_id", sort=False):
            case_rows.append(compute_case_metrics(prompt_df, selected_layers))
        case_df = pd.DataFrame(case_rows)
        summary = summarize_case_metrics(case_df)
        summary["window_name"] = window_name
        summary["selected_layers"] = selected_layers
        case_records.append(summary)

    window_df = pd.DataFrame(case_records)
    window_df = window_df.sort_values(
        by=[
            "contrib_joint_support_match_rate",
            "mean_layer_stable_contrib_support_rate",
            "mean_contrib_support_mass_share",
            "prediction_set_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)
    best_name = str(window_df.iloc[0]["window_name"])
    return best_name, window_df


def load_selected_window_name(workspace: Path, source_iteration: int | None = None) -> str:
    source_iter = source_iteration if source_iteration is not None else 1
    summary_path = workspace / "custom_outputs" / f"iteration_{source_iter:02d}_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        name = str(payload.get("selected_window_name", "")).strip()
        if name:
            return name
    return "last_half"


def build_new_reader_explanation(
    *,
    iteration: int,
    summary: Dict[str, Any],
    selected_layers: Sequence[int],
) -> str:
    overall = summary["overall"]
    family_best = max(
        summary["by_family"].items(),
        key=lambda item: item[1]["gate_joint_support_match_rate"],
    )
    family_worst = min(
        summary["by_family"].items(),
        key=lambda item: item[1]["gate_joint_support_match_rate"],
    )

    return (
        f"# Iteration {iteration:02d} New Reader Explanation\n\n"
        f"CURRENT_CONTEXT_FOR_NEW_READER:\n"
        f"This is an early exploratory iteration for {DIRECTION_TITLE}. The final paper framing does not use the retired solver-style terminology. Instead, this run should be read as an early support-set recovery study: can internal late-layer signals identify the gold support facts before decoded behavior fully reflects them?\n\n"
        f"SPECIAL_SETUP_EXPLANATIONS:\n"
        f"The prompt is rendered as a labeled fact-selection task instead of a free-form QA answer so that the exact gold support set can be tested before any support label is emitted. Each case contains six candidate fact spans: two gold supporting facts and four distractors. For every prompt we run two teacher-forced answer steps: before the first gold label and before the second gold label after appending the first gold label plus a comma. At each layer we aggregate both raw gate mass and peak token gate within each fact span, plus positive writer contribution from the answer position onto each fact span, then ask whether the top-2 discovered facts equal the gold support set. The gate-based exact recovery metric uses peak token gate inside each fact span so that long distractor sentences do not win merely by accumulating many small attention weights. The selected late-layer window for this iteration is {list(selected_layers)}, which is the direct object used for support-set recovery rather than a loose salience visualization.\n\n"
        f"WHAT_WE_TESTED_THIS_ITERATION:\n"
        f"We tested whether the late-layer source-fact signal already matches the true support set before the model writes its support labels. The main metrics are exact top-2 gold-support recovery by gate and by writer contribution, support-mass share on the gold facts, and a failure taxonomy that separates discovery failure from readout failure. This iteration used model `{summary['model_id']}`, pack `{summary['pack']}`, and `{overall['prompt_count']}` prompts across the selected prompt families.\n\n"
        f"WHAT_THE_RESULTS_SUPPORT_RELATIVE_TO_FINAL_THESIS:\n"
        f"The strongest direct evidence here is the gate-based exact support recovery rate. Overall, the selected window reached gate top-2 support recovery `{overall['gate_joint_support_match_rate']:.3f}` with mean gold support peak-gate share `{overall['mean_gate_support_peak_share']:.3f}` and layer-stable gate support rate `{overall['mean_layer_stable_gate_support_rate']:.3f}`. Writer-based exact recovery was `{overall['contrib_joint_support_match_rate']:.3f}`. Among error cases, discovery failures accounted for `{overall['discovery_failure_share_among_errors']:.3f}` of errors, while readout failures accounted for `{overall['readout_failure_share_among_errors']:.3f}`. The best prompt family in this iteration was `{family_best[0]}` with gate recovery `{family_best[1]['gate_joint_support_match_rate']:.3f}`, while the weakest family was `{family_worst[0]}` at `{family_worst[1]['gate_joint_support_match_rate']:.3f}`.\n\n"
        f"DOES_THIS_SUPPORT_THE_FINAL_THESIS:\n"
        f"partial\n\n"
        f"WHY_THIS_IS_ARCHIVAL_AND_NOT_CANONICAL:\n"
        f"Even if the gate-based support recovery is materially above random, this iteration does not by itself establish the final paper claim. It predates the later cross-family routing-regime taxonomy, the repaired causal package, the natural-task bridge, and the robustness audits. On its own, it is best treated as exploratory support for pre-decoding support-set recovery rather than as a decisive result.\n\n"
        f"NEXT_STEP_FROM_THIS_ITERATION:\n"
        f"The next move is to test stability across wording packs and parameter scales, then ask whether writer-side recovery provides a cleaner and more general signature than gate-side discovery alone.\n"
    )


def build_pivot_new_reader_explanation(
    *,
    iteration: int,
    summary: Dict[str, Any],
    selected_layers: Sequence[int],
) -> str:
    """New reader explanation for pivot iterations (05+) targeting the writer-readout gap."""
    overall = summary["overall"]
    family_best = max(
        summary["by_family"].items(),
        key=lambda item: item[1]["contrib_joint_support_match_rate"],
    )
    family_worst = min(
        summary["by_family"].items(),
        key=lambda item: item[1]["contrib_joint_support_match_rate"],
    )

    return (
        f"# Iteration {iteration:02d} New Reader Explanation\n\n"
        f"CURRENT_CONTEXT_FOR_NEW_READER:\n"
        f"This iteration belongs to the writer-readout-gap consolidation phase for {DIRECTION_TITLE}. "
        f"Iterations 01-04 established an exploratory support-set benchmark; the canonical thesis now asks whether "
        f"late-layer attention value projections recover the gold support set more reliably than decoded behavior, "
        f"producing a writer-readout gap whose strength varies by model family.\n\n"
        f"SPECIAL_SETUP_EXPLANATIONS:\n"
        f"Same measurement protocol as iterations 01-04: HotpotQA distractor examples with gold supporting-fact annotations. "
        f"Six candidate fact spans (2 gold, 4 distractor). Two teacher-forced answer steps per prompt. "
        f"Gate and writer metrics aggregated over the selected late-layer window {list(selected_layers)}. "
        f"The new derived metric is `writer_readout_gap = contrib_joint_support_match_rate - prediction_set_accuracy`.\n\n"
        f"WHAT_WE_TESTED_THIS_ITERATION:\n"
        f"We tested whether the writer-readout gap persists on model `{summary['model_id']}` using pack `{summary['pack']}` "
        f"with `{overall['prompt_count']}` prompts. The key question is whether writer-side exact support recovery remains "
        f"high while decoded accuracy stays lower, producing a measurable gap.\n\n"
        f"WHAT_THE_RESULTS_SUPPORT_RELATIVE_TO_ORIGINAL_IDEA:\n"
        f"Writer-based exact top-2 support recovery: `{overall['contrib_joint_support_match_rate']:.3f}`. "
        f"Decoded prediction accuracy: `{overall['prediction_set_accuracy']:.3f}`. "
        f"**Writer-readout gap: `{overall['writer_readout_gap']:.3f}`**. "
        f"Gate-based exact recovery: `{overall['gate_joint_support_match_rate']:.3f}`. "
        f"Among error cases: writer-readout failures (writer found gold set but model decoded wrong) = "
        f"`{overall['writer_readout_failure_share_among_errors']:.3f}`, "
        f"writer-discovery failures = `{overall['writer_discovery_failure_share_among_errors']:.3f}`. "
        f"Best prompt family: `{family_best[0]}` (writer recovery `{family_best[1]['contrib_joint_support_match_rate']:.3f}`). "
        f"Weakest: `{family_worst[0]}` (`{family_worst[1]['contrib_joint_support_match_rate']:.3f}`).\n\n"
        f"DOES_THIS_SUPPORT_THE_PIVOTED_THESIS:\n"
        f"{'yes' if overall['writer_readout_gap'] >= 0.15 and overall['contrib_joint_support_match_rate'] >= 0.40 else 'partial' if overall['writer_readout_gap'] > 0.05 else 'no'}\n\n"
        f"PHASE_A_GATE_ASSESSMENT:\n"
        f"Writer exact recovery {'PASSES' if overall['contrib_joint_support_match_rate'] >= 0.50 else 'FAILS'} "
        f"the >= 0.50 threshold (actual: {overall['contrib_joint_support_match_rate']:.3f}). "
        f"Writer-readout gap {'PASSES' if overall['writer_readout_gap'] >= 0.15 else 'FAILS'} "
        f"the >= 0.15 threshold (actual: {overall['writer_readout_gap']:.3f}).\n"
    )


def render_iteration_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"# Iteration {summary['iteration']:02d} Custom Summary",
        "",
        f"- direction_title: {DIRECTION_TITLE}",
        f"- model_id: {summary['model_id']}",
        f"- lexical_pack: {summary['pack']}",
        f"- base_example_count: {summary['base_example_count']}",
        f"- prompt_count: {summary['overall']['prompt_count']}",
        f"- selected_window_name: {summary['selected_window_name']}",
        f"- selected_layers: {summary['selected_layers']}",
        f"- prediction_set_accuracy: {summary['overall']['prediction_set_accuracy']:.3f}",
        f"- gate_joint_support_match_rate: {summary['overall']['gate_joint_support_match_rate']:.3f}",
        f"- contrib_joint_support_match_rate: {summary['overall']['contrib_joint_support_match_rate']:.3f}",
        f"- writer_readout_gap: {summary['overall']['writer_readout_gap']:.3f}",
        f"- mean_gate_support_peak_share: {summary['overall']['mean_gate_support_peak_share']:.3f}",
        f"- mean_layer_stable_gate_support_rate: {summary['overall']['mean_layer_stable_gate_support_rate']:.3f}",
        f"- discovery_failure_share_among_errors: {summary['overall']['discovery_failure_share_among_errors']:.3f}",
        f"- readout_failure_share_among_errors: {summary['overall']['readout_failure_share_among_errors']:.3f}",
        "",
        "## Family Summary",
        "",
        "| family | prompt_count | prediction_acc | gate_top2_match | contrib_top2_match | writer_readout_gap | gate_peak_share | stable_gate | discovery_fail_share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family_name, family_summary in summary["by_family"].items():
        lines.append(
            "| "
            + family_name
            + f" | {family_summary['prompt_count']}"
            + f" | {family_summary['prediction_set_accuracy']:.3f}"
            + f" | {family_summary['gate_joint_support_match_rate']:.3f}"
            + f" | {family_summary['contrib_joint_support_match_rate']:.3f}"
            + f" | {family_summary['writer_readout_gap']:.3f}"
            + f" | {family_summary['mean_gate_support_peak_share']:.3f}"
            + f" | {family_summary['mean_layer_stable_gate_support_rate']:.3f}"
            + f" | {family_summary['discovery_failure_share_among_errors']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _model_slug(model_id: str) -> str:
    """Short filesystem-safe slug for a model_id."""
    name = model_id.split("/")[-1].lower()
    for ch in (" ", ".", "-"):
        name = name.replace(ch, "_")
    return name


def run_iteration(workspace: Path, repo_root: Path, iteration: int, model_id_override: str | None = None) -> Dict[str, Any]:
    iteration_cfg = ITERATION_CONFIGS.get(iteration)
    if iteration_cfg is None:
        raise RuntimeError(f"Unsupported iteration: {iteration}")

    ensure_repo_imports(repo_root)
    from src.hook_registry import resolve_adapter
    from src.load_model import load_model_and_tokenizer

    prompt_file = iteration_cfg.get("prompt_file", "prompts.jsonl")
    all_rows = jsonl_load(workspace / prompt_file)
    if not all_rows:
        raise RuntimeError(f"{prompt_file} is missing or empty. Run the appropriate --prepare-*-prompts first.")

    # Resolve model_id: CLI override > config
    if model_id_override:
        model_id = model_id_override
    elif iteration_cfg["model_id"] is not None:
        model_id = str(iteration_cfg["model_id"])
    else:
        raise RuntimeError(f"Iteration {iteration} requires --model-id override.")

    # Select rows: pack="both" includes all packs
    pack_filter = iteration_cfg["pack"]
    selected_rows = [
        row
        for row in all_rows
        if (pack_filter == "both" or row["lexical_pack"] == pack_filter)
        and int(row["example_rank"]) <= int(iteration_cfg["max_examples"])
    ]
    if not selected_rows:
        raise RuntimeError(f"No prompt rows found for iteration {iteration}.")
    loaded = load_model_and_tokenizer(
        model_id=model_id,
        trust_remote_code=True,
        dtype="auto",
        device="auto",
    )

    # Re-render prompts if the model uses a different chat template
    plan = load_plan(workspace)
    original_model_id = str(plan["experiment"]["model_id"])
    max_length = int(plan["experiment"].get("max_length", 768))
    if model_id != original_model_id:
        rerendered: List[Dict[str, Any]] = []
        for row in selected_rows:
            new_prompt = rerender_prompt_for_model(row, loaded.tokenizer, max_length)
            if new_prompt is None:
                continue
            row = dict(row)
            row["prompt"] = new_prompt
            rerendered.append(row)
        if not rerendered:
            raise RuntimeError(f"All prompts exceed max_length={max_length} after re-rendering for {model_id}.")
        selected_rows = rerendered
    adapter = resolve_adapter(loaded.model)
    base_model = adapter.get_base_model(loaded.model)
    layers = adapter.get_layers(base_model)
    num_layers = len(layers)

    layer_records: List[Dict[str, Any]] = []
    prediction_records: List[Dict[str, Any]] = []

    for row in selected_rows:
        decoded_labels = greedy_decode_two_labels(loaded.model, loaded.tokenizer, row, loaded.device)
        prediction_records.append(
            {
                "prompt_id": row["prompt_id"],
                "decoded_prediction_set": decoded_labels,
            }
        )
        first_gold, second_gold = row["support_labels"]
        step_specs = [(first_gold, ""), (second_gold, f"{first_gold},")]
        for target_label, answer_prefix in step_specs:
            encoded = encode_step_prompt(
                row=row,
                tokenizer=loaded.tokenizer,
                device=loaded.device,
                target_label=target_label,
                answer_prefix=answer_prefix,
            )
            records, metadata = analyze_step_prompt(
                model=loaded.model,
                adapter=adapter,
                base_model=base_model,
                layers=layers,
                loaded_device=loaded.device,
                encoded=encoded,
            )
            for record in records:
                record["decoded_prediction_set"] = decoded_labels
            layer_records.extend(records)

    layer_df = pd.DataFrame(layer_records)
    prediction_df = pd.DataFrame(prediction_records)
    if layer_df.empty:
        raise RuntimeError("No layer records were produced.")

    if iteration_cfg["scan_windows"]:
        selected_window_name, scan_df = select_window_from_scan(layer_df)
        scan_artifact_path = workspace / "custom_outputs" / f"iteration_{iteration:02d}_window_scan.csv"
        scan_df.to_csv(scan_artifact_path, index=False)
    elif iteration_cfg.get("window_name"):
        selected_window_name = str(iteration_cfg["window_name"])
        scan_df = pd.DataFrame()
    elif iteration_cfg.get("window_source_iteration") == "per_model":
        # Phase C: use locked Phase A window for each model
        if model_id in PHASE_A_WINDOWS:
            selected_window_name = PHASE_A_WINDOWS[model_id]
        else:
            raise RuntimeError(f"No Phase A window registered for {model_id}. Add to PHASE_A_WINDOWS.")
        scan_df = pd.DataFrame()
    else:
        source_iter = iteration_cfg.get("window_source_iteration")
        selected_window_name = load_selected_window_name(workspace, source_iter)
        scan_df = pd.DataFrame()

    selected_layers = get_window_layers(num_layers, selected_window_name)

    case_rows = []
    for prompt_id, prompt_df in layer_df.groupby("prompt_id", sort=False):
        case_rows.append(compute_case_metrics(prompt_df, selected_layers))
    case_df = pd.DataFrame(case_rows)

    family_summary: Dict[str, Any] = {}
    for family_name, family_df in case_df.groupby("prompt_family", sort=True):
        family_summary[str(family_name)] = summarize_case_metrics(family_df)

    overall_summary = summarize_case_metrics(case_df)

    # Compute headline-family metrics if configured
    headline_families = iteration_cfg.get("headline_families")
    headline_summary = None
    if headline_families:
        headline_mask = case_df["family_name"].isin(headline_families)
        headline_df = case_df[headline_mask]
        if not headline_df.empty:
            headline_summary = summarize_case_metrics(headline_df)
            # Add bootstrap CIs for headline metrics
            headline_writer_values = headline_df["contrib_joint_support_match"].astype(float).tolist()
            headline_decoded_values = headline_df["prediction_set_correct"].astype(float).tolist()
            headline_gap_values = headline_df["writer_readout_gap"].astype(float).tolist()
            headline_summary["bootstrap_ci"] = {
                "writer_recovery": dict(zip(
                    ["mean", "ci_lo", "ci_hi"],
                    bootstrap_ci(headline_writer_values),
                )),
                "decoded_accuracy": dict(zip(
                    ["mean", "ci_lo", "ci_hi"],
                    bootstrap_ci(headline_decoded_values),
                )),
                "writer_readout_gap": dict(zip(
                    ["mean", "ci_lo", "ci_hi"],
                    bootstrap_ci(headline_gap_values),
                )),
            }

    # Bootstrap CIs for overall metrics
    all_writer_values = case_df["contrib_joint_support_match"].astype(float).tolist()
    all_decoded_values = case_df["prediction_set_correct"].astype(float).tolist()
    all_gap_values = case_df["writer_readout_gap"].astype(float).tolist()
    overall_summary["bootstrap_ci"] = {
        "writer_recovery": dict(zip(
            ["mean", "ci_lo", "ci_hi"],
            bootstrap_ci(all_writer_values),
        )),
        "decoded_accuracy": dict(zip(
            ["mean", "ci_lo", "ci_hi"],
            bootstrap_ci(all_decoded_values),
        )),
        "writer_readout_gap": dict(zip(
            ["mean", "ci_lo", "ci_hi"],
            bootstrap_ci(all_gap_values),
        )),
    }

    summary = {
        "direction_title": DIRECTION_TITLE,
        "iteration": iteration,
        "model_id": model_id,
        "pack": iteration_cfg["pack"],
        "base_example_count": int(iteration_cfg["max_examples"]),
        "prompt_count": int(len(selected_rows)),
        "num_layers": num_layers,
        "selected_window_name": selected_window_name,
        "selected_layers": selected_layers,
        "overall": overall_summary,
        "by_family": family_summary,
    }
    if headline_summary is not None:
        summary["headline"] = headline_summary
        summary["headline_families"] = headline_families
    if not scan_df.empty:
        summary["window_scan"] = scan_df.to_dict(orient="records")

    # Use model slug in filenames for multi-model iterations (>= 8)
    slug = _model_slug(model_id) if model_id_override else ""
    suffix = f"_{slug}" if slug else ""

    custom_outputs = workspace / "custom_outputs"
    custom_outputs.mkdir(parents=True, exist_ok=True)
    layer_path = custom_outputs / f"iteration_{iteration:02d}{suffix}_fact_metrics.csv"
    case_path = custom_outputs / f"iteration_{iteration:02d}{suffix}_case_metrics.csv"
    family_path = custom_outputs / f"iteration_{iteration:02d}{suffix}_family_summary.csv"
    summary_json_path = custom_outputs / f"iteration_{iteration:02d}{suffix}_summary.json"
    summary_md_path = custom_outputs / f"iteration_{iteration:02d}{suffix}_summary.md"

    layer_df.to_csv(layer_path, index=False)
    case_df.to_csv(case_path, index=False)
    pd.DataFrame(
        [{"prompt_family": family_name, **metrics} for family_name, metrics in family_summary.items()]
    ).to_csv(family_path, index=False)
    save_json(summary_json_path, summary)
    summary_md_path.write_text(render_iteration_summary_markdown(summary), encoding="utf-8")

    analysis_dir = workspace / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    new_reader_name = f"iteration_{iteration:02d}{suffix}_new_reader_explanation.md" if suffix else f"iteration_{iteration:02d}_new_reader_explanation.md"
    new_reader_path = analysis_dir / new_reader_name
    explanation_builder = build_pivot_new_reader_explanation if iteration >= 5 else build_new_reader_explanation
    new_reader_path.write_text(
        explanation_builder(
            iteration=iteration,
            summary=summary,
            selected_layers=selected_layers,
        ),
        encoding="utf-8",
    )

    # Skip expensive representative decomposition for multi-model expansion runs
    run_dirs: List[str] = []
    if not model_id_override:
        representative_case = case_df.sort_values(
            by=["prediction_set_correct", "gate_joint_support_match", "gate_support_peak_share"],
            ascending=[False, False, False],
        ).iloc[0]
        representative_row = next(row for row in selected_rows if row["prompt_id"] == representative_case["prompt_id"])
        first_gold, second_gold = representative_row["support_labels"]
        run_dirs = [
            run_representative_decomposition(
                repo_root=repo_root,
                model_id=model_id,
                prompt_text=str(representative_row["prompt"]),
                iteration=iteration,
                tag="step1_support_selection",
                target_label=first_gold,
            ),
            run_representative_decomposition(
                repo_root=repo_root,
                model_id=model_id,
                prompt_text=prompt_with_prefix(str(representative_row["prompt"]), f"{first_gold},"),
                iteration=iteration,
                tag="step2_support_selection",
                target_label=second_gold,
            ),
        ]

    artifacts = [
        str(layer_path),
        str(case_path),
        str(family_path),
        str(summary_json_path),
        str(summary_md_path),
        str(new_reader_path),
    ]
    if not scan_df.empty:
        artifacts.append(str(scan_artifact_path))

    headline_tag = ""
    if headline_summary is not None:
        ci = headline_summary["bootstrap_ci"]
        headline_tag = (
            f"; headline_writer={ci['writer_recovery']['mean']:.3f} "
            f"[{ci['writer_recovery']['ci_lo']:.3f},{ci['writer_recovery']['ci_hi']:.3f}]"
            f"; headline_gap={ci['writer_readout_gap']['mean']:.3f} "
            f"[{ci['writer_readout_gap']['ci_lo']:.3f},{ci['writer_readout_gap']['ci_hi']:.3f}]"
        )

    notes = (
        f"model={model_id}; selected_window={selected_window_name}; "
        f"gate_joint_support_match_rate={overall_summary['gate_joint_support_match_rate']:.3f}; "
        f"contrib_joint_support_match_rate={overall_summary['contrib_joint_support_match_rate']:.3f}; "
        f"prediction_set_accuracy={overall_summary['prediction_set_accuracy']:.3f}; "
        f"writer_readout_gap={overall_summary['writer_readout_gap']:.3f}; "
        f"discovery_failure_share_among_errors={overall_summary['discovery_failure_share_among_errors']:.3f}"
        + headline_tag
    )

    return {
        "run_dirs": run_dirs,
        "artifacts": artifacts,
        "notes": notes,
        "summary": summary,
    }


def _get_v_proj(adapter, base_model, layer_idx: int):
    """Get the value projection module for a given layer via the adapter."""
    layers = adapter.get_layers(base_model)
    attn = adapter.get_layer_attention(layers[layer_idx])
    if hasattr(attn, "v_proj"):
        return attn.v_proj
    raise AttributeError(f"Cannot find v_proj on attention module at layer {layer_idx}")


def greedy_decode_two_labels_with_hooks(
    model, tokenizer, row: Dict[str, Any], device: torch.device,
    hook_layers: List[int], hook_positions: List[int],
    adapter, base_model,
    mode: str = "zero", scale: float = 1.0,
) -> List[str]:
    """Decode two labels with value projection hooks active during forward passes."""
    label_token_ids = resolve_label_token_ids(tokenizer, FACT_LABELS)
    prediction: List[str] = []
    prefix = ""

    # Register hooks
    hooks = []
    pos_tensor = torch.tensor(hook_positions, device=device, dtype=torch.long)

    def make_hook_fn(layer_idx: int):
        def hook_fn(module, inp, output):
            if mode == "zero":
                output = output.clone()
                output[:, pos_tensor, :] = 0.0
            elif mode == "scale":
                output = output.clone()
                output[:, pos_tensor, :] = output[:, pos_tensor, :] * scale
            return output
        return hook_fn

    for layer_idx in hook_layers:
        v_proj = _get_v_proj(adapter, base_model, layer_idx)
        h = v_proj.register_forward_hook(make_hook_fn(layer_idx))
        hooks.append(h)

    try:
        for _step in range(2):
            prompt_text = prompt_with_prefix(str(row["prompt"]), prefix)
            tokenized = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(device)
            position_ids = compute_position_ids(attention_mask).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
            logits = outputs.logits[0, input_ids.shape[1] - 1].float().detach()
            label_logits = get_label_logits(logits, label_token_ids)
            predicted = max(label_logits.items(), key=lambda item: item[1])[0]
            prediction.append(predicted)
            prefix = "".join([prediction[0], ","]) if len(prediction) == 1 else prefix
    finally:
        for h in hooks:
            h.remove()
    return prediction


def bootstrap_ci(values: List[float], n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute bootstrap mean and confidence interval."""
    import numpy as np
    arr = np.array(values)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    rng = np.random.RandomState(42)
    boot_means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    boot_means.sort()
    lo = boot_means[int(n_boot * alpha / 2)]
    hi = boot_means[int(n_boot * (1 - alpha / 2))]
    return mean, lo, hi


def run_phase_b(workspace: Path, repo_root: Path, config_name: str | None = None) -> Dict[str, Any]:
    """Run Phase B causal interventions on the specified model."""
    if config_name:
        cfg = PHASE_B_CONFIGS[config_name]
    else:
        cfg = PHASE_B_CONFIG
    ensure_repo_imports(repo_root)
    from src.hook_registry import resolve_adapter
    from src.load_model import load_model_and_tokenizer

    # Load baseline case metrics
    bl_iter = cfg['baseline_iteration']
    if isinstance(bl_iter, str):
        baseline_case_path = workspace / "custom_outputs" / f"iteration_{bl_iter}_case_metrics.csv"
    else:
        baseline_case_path = workspace / "custom_outputs" / f"iteration_{bl_iter:02d}_case_metrics.csv"
    baseline_df = pd.read_csv(baseline_case_path)

    # Load prompts and re-render for the intervention model
    all_rows = jsonl_load(workspace / "prompts.jsonl")
    pack_filter = cfg["pack"]
    selected_rows = [
        row for row in all_rows
        if (pack_filter == "both" or row["lexical_pack"] == pack_filter)
        and int(row["example_rank"]) <= int(cfg["max_examples"])
    ]

    model_id = str(cfg["model_id"])
    loaded = load_model_and_tokenizer(
        model_id=model_id,
        trust_remote_code=True,
        dtype="auto",
        device="auto",
    )
    adapter = resolve_adapter(loaded.model)
    base_model = adapter.get_base_model(loaded.model)
    layers = adapter.get_layers(base_model)
    num_layers = len(layers)

    # Re-render prompts for this model
    plan = load_plan(workspace)
    original_model_id = str(plan["experiment"]["model_id"])
    max_length = int(plan["experiment"].get("max_length", 768))
    if model_id != original_model_id:
        rerendered = []
        for row in selected_rows:
            new_prompt = rerender_prompt_for_model(row, loaded.tokenizer, max_length)
            if new_prompt is None:
                continue
            row = dict(row)
            row["prompt"] = new_prompt
            rerendered.append(row)
        selected_rows = rerendered

    # Determine intervention layers
    window_layers = get_window_layers(num_layers, cfg["window_name"])

    # Build prompt_id -> row mapping
    row_map = {row["prompt_id"]: row for row in selected_rows}

    # For each prompt, compute fact positions and identify intervention targets
    results: List[Dict[str, Any]] = []

    for _, baseline_row in baseline_df.iterrows():
        prompt_id = str(baseline_row["prompt_id"])
        if prompt_id not in row_map:
            continue
        row = row_map[prompt_id]

        gold_support = sorted(row["support_labels"])
        gold_set = set(gold_support)

        # Parse baseline decoded prediction
        baseline_labels_raw = baseline_row["decoded_prediction_set"]
        if isinstance(baseline_labels_raw, str):
            baseline_labels = eval(baseline_labels_raw)
        else:
            baseline_labels = list(baseline_labels_raw)
        baseline_correct = set(baseline_labels) == gold_set and len(baseline_labels) == 2

        # Parse writer-aligned facts (contrib_top2)
        writer_top2_raw = baseline_row["contrib_top2_labels"]
        if isinstance(writer_top2_raw, str):
            writer_top2 = eval(writer_top2_raw)
        else:
            writer_top2 = list(writer_top2_raw)
        writer_match = set(writer_top2) == gold_set

        # Compute fact positions in the prompt
        prompt_text = str(row["prompt"])
        spans: Dict[str, Tuple[int, int]] = {}
        for fact in row["facts"]:
            line = fact_line(fact)
            start = prompt_text.index(line)
            spans[str(fact["label"])] = (start, start + len(line))
        fact_positions = token_positions_for_spans(loaded.tokenizer, prompt_text, spans)

        # Identify writer-aligned fact positions (top-2 by writer)
        writer_positions: List[int] = []
        for label in writer_top2:
            writer_positions.extend(fact_positions[label])

        # Identify gold support fact positions
        gold_positions: List[int] = []
        for label in gold_support:
            gold_positions.extend(fact_positions[label])

        # Identify top-2 distractor facts by gate
        gate_top2_raw = baseline_row["gate_top2_labels"]
        if isinstance(gate_top2_raw, str):
            gate_top2 = eval(gate_top2_raw)
        else:
            gate_top2 = list(gate_top2_raw)
        distractor_labels = [l for l in gate_top2 if l not in gold_set]
        if not distractor_labels:
            # Fall back: use non-support facts
            distractor_labels = [f["label"] for f in row["facts"] if not f["is_support"]][:2]
        distractor_positions: List[int] = []
        for label in distractor_labels:
            distractor_positions.extend(fact_positions[label])

        result = {
            "prompt_id": prompt_id,
            "example_id": str(row["example_id"]),
            "prompt_family": str(row["prompt_family"]),
            "gold_support": gold_support,
            "writer_top2": writer_top2,
            "writer_match": writer_match,
            "baseline_correct": baseline_correct,
            "baseline_labels": baseline_labels,
            "is_gap_case": writer_match and not baseline_correct,
        }

        # B.1: Writer-aligned fact ablation (zero out value projections)
        ablated_labels = greedy_decode_two_labels_with_hooks(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=window_layers, hook_positions=writer_positions,
            adapter=adapter, base_model=base_model,
            mode="zero",
        )
        ablated_correct = set(ablated_labels) == gold_set and len(ablated_labels) == 2
        result["b1_ablation_labels"] = ablated_labels
        result["b1_ablation_correct"] = ablated_correct
        result["b1_ablation_accuracy_change"] = float(ablated_correct) - float(baseline_correct)

        # B.2: Writer-aligned fact amplification (only meaningful for gap cases)
        for factor in cfg["amplification_factors"]:
            amp_labels = greedy_decode_two_labels_with_hooks(
                loaded.model, loaded.tokenizer, row, loaded.device,
                hook_layers=window_layers, hook_positions=gold_positions,
                adapter=adapter, base_model=base_model,
                mode="scale", scale=factor,
            )
            amp_correct = set(amp_labels) == gold_set and len(amp_labels) == 2
            result[f"b2_amp_{factor:.1f}x_labels"] = amp_labels
            result[f"b2_amp_{factor:.1f}x_correct"] = amp_correct
            result[f"b2_amp_{factor:.1f}x_accuracy_change"] = float(amp_correct) - float(baseline_correct)

        # B.3: Distractor fact ablation (control)
        if distractor_positions:
            dist_ablated_labels = greedy_decode_two_labels_with_hooks(
                loaded.model, loaded.tokenizer, row, loaded.device,
                hook_layers=window_layers, hook_positions=distractor_positions,
                adapter=adapter, base_model=base_model,
                mode="zero",
            )
            dist_ablated_correct = set(dist_ablated_labels) == gold_set and len(dist_ablated_labels) == 2
            result["b3_distractor_ablation_labels"] = dist_ablated_labels
            result["b3_distractor_ablation_correct"] = dist_ablated_correct
            result["b3_distractor_ablation_accuracy_change"] = float(dist_ablated_correct) - float(baseline_correct)
        else:
            result["b3_distractor_ablation_labels"] = baseline_labels
            result["b3_distractor_ablation_correct"] = baseline_correct
            result["b3_distractor_ablation_accuracy_change"] = 0.0

        results.append(result)

    results_df = pd.DataFrame(results)

    # Compute aggregate statistics with bootstrap CIs
    correct_mask = results_df["baseline_correct"]
    gap_mask = results_df["is_gap_case"]
    incorrect_mask = ~results_df["baseline_correct"]

    summary = {"model_id": model_id, "prompt_count": len(results_df)}

    # B.1: Ablation effect on previously-correct prompts
    if correct_mask.sum() > 0:
        b1_changes = results_df.loc[correct_mask, "b1_ablation_accuracy_change"].tolist()
        mean, lo, hi = bootstrap_ci(b1_changes)
        summary["b1_ablation_on_correct"] = {
            "n": int(correct_mask.sum()),
            "mean_accuracy_change": mean,
            "ci_95_lo": lo,
            "ci_95_hi": hi,
            "significant": hi < 0,  # significant drop = CI entirely below zero
        }

    # B.1: Ablation effect on previously-incorrect (sanity check)
    if incorrect_mask.sum() > 0:
        b1_changes_inc = results_df.loc[incorrect_mask, "b1_ablation_accuracy_change"].tolist()
        mean_inc, lo_inc, hi_inc = bootstrap_ci(b1_changes_inc)
        summary["b1_ablation_on_incorrect"] = {
            "n": int(incorrect_mask.sum()),
            "mean_accuracy_change": mean_inc,
            "ci_95_lo": lo_inc,
            "ci_95_hi": hi_inc,
        }

    # B.2: Amplification effect on gap cases
    for factor in cfg["amplification_factors"]:
        col = f"b2_amp_{factor:.1f}x_accuracy_change"
        if gap_mask.sum() > 0:
            b2_changes = results_df.loc[gap_mask, col].tolist()
            mean_b2, lo_b2, hi_b2 = bootstrap_ci(b2_changes)
            summary[f"b2_amp_{factor:.1f}x_on_gap_cases"] = {
                "n": int(gap_mask.sum()),
                "mean_accuracy_change": mean_b2,
                "ci_95_lo": lo_b2,
                "ci_95_hi": hi_b2,
                "significant": lo_b2 > 0,  # significant gain = CI entirely above zero
            }
        # Also on all prompts
        b2_all = results_df[col].tolist()
        mean_all, lo_all, hi_all = bootstrap_ci(b2_all)
        summary[f"b2_amp_{factor:.1f}x_on_all"] = {
            "n": len(b2_all),
            "mean_accuracy_change": mean_all,
            "ci_95_lo": lo_all,
            "ci_95_hi": hi_all,
        }

    # B.3: Distractor ablation effect on all prompts
    b3_changes = results_df["b3_distractor_ablation_accuracy_change"].tolist()
    mean_b3, lo_b3, hi_b3 = bootstrap_ci(b3_changes)
    summary["b3_distractor_ablation_on_all"] = {
        "n": len(b3_changes),
        "mean_accuracy_change": mean_b3,
        "ci_95_lo": lo_b3,
        "ci_95_hi": hi_b3,
    }

    # B.3: On incorrect prompts
    if incorrect_mask.sum() > 0:
        b3_inc = results_df.loc[incorrect_mask, "b3_distractor_ablation_accuracy_change"].tolist()
        mean_b3i, lo_b3i, hi_b3i = bootstrap_ci(b3_inc)
        summary["b3_distractor_ablation_on_incorrect"] = {
            "n": len(b3_inc),
            "mean_accuracy_change": mean_b3i,
            "ci_95_lo": lo_b3i,
            "ci_95_hi": hi_b3i,
        }

    # Phase B gate assessment
    b1_sig = summary.get("b1_ablation_on_correct", {}).get("significant", False)
    b2_sig_any = any(
        summary.get(f"b2_amp_{f:.1f}x_on_gap_cases", {}).get("significant", False)
        for f in cfg["amplification_factors"]
    )
    summary["phase_b_gate"] = "PASS" if (b1_sig or b2_sig_any) else "FAIL"

    # Save artifacts
    slug = f"_{config_name}" if config_name else ""
    custom_outputs = workspace / "custom_outputs"
    custom_outputs.mkdir(parents=True, exist_ok=True)
    results_path = custom_outputs / f"phase_b{slug}_intervention_results.csv"
    summary_path = custom_outputs / f"phase_b{slug}_summary.json"
    results_df.to_csv(results_path, index=False)
    save_json(summary_path, summary)

    return {
        "artifacts": [str(results_path), str(summary_path)],
        "notes": (
            f"Phase B gate: {summary['phase_b_gate']}; "
            f"B1 ablation on correct: {summary.get('b1_ablation_on_correct', {}).get('mean_accuracy_change', 'N/A'):.3f}; "
            f"B2 amp 2.0x on gap: {summary.get('b2_amp_2.0x_on_gap_cases', {}).get('mean_accuracy_change', 'N/A'):.3f}; "
            f"B3 distractor on incorrect: {summary.get('b3_distractor_ablation_on_incorrect', {}).get('mean_accuracy_change', 'N/A'):.3f}"
        ),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    repo_root = infer_repo_root(workspace, args.repo_root)

    if args.prepare_prompts:
        result = prepare_prompts(workspace, repo_root, args.max_examples)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.prepare_fever_prompts:
        result = prepare_fever_prompts(workspace, repo_root, args.max_examples)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.phase_b:
        result = run_phase_b(workspace, repo_root, config_name=args.phase_b_config)
        print(json.dumps(
            {"artifacts": result["artifacts"], "notes": result["notes"]},
            ensure_ascii=False, indent=2,
        ))
        return

    if args.iteration is None:
        raise RuntimeError("--iteration is required unless --prepare-prompts or --phase-b is used.")

    result = run_iteration(workspace, repo_root, args.iteration, model_id_override=args.model_id)
    print(
        json.dumps(
            {
                "run_dirs": result["run_dirs"],
                "artifacts": result["artifacts"],
                "notes": result["notes"],
                "selected_window_name": result["summary"]["selected_window_name"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
