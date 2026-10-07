#!/usr/bin/env python3
"""Direction 36 — Reviewer Response Experiments.

Addresses the five consolidated reviewer concerns:
  1. Specificity controls: gold vs distractor vs random vs outside-window ablation
  2. Full layer sweep: heatmap of causal effects across all contiguous windows
  3. Readout baselines: attention-only, value-norm, signed-sum vs proposed method
  4. Label permutation: shuffle label-to-fact assignments to test construct validity
  5. Unconditional effect: reanalysis of existing P9 data (no GPU needed)
  6. Readout-causality coupling: stratify by writer score, check ablation correlation

Usage:
  python run_reviewer_controls.py --experiment specificity --cell qwen_7b
  python run_reviewer_controls.py --experiment layer_sweep --cell gemma_2b --sweep-examples 50
  python run_reviewer_controls.py --experiment readout_baselines --cell qwen_7b
  python run_reviewer_controls.py --experiment label_permutation --cell qwen_7b
  python run_reviewer_controls.py --experiment unconditional --all-cells
  python run_reviewer_controls.py --experiment coupling --cell qwen_7b
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random as pyrandom
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Cell configurations (from P9 canonical package)
# ---------------------------------------------------------------------------
CELL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemma_2b": {
        "model_id": "google/gemma-2-2b-it",
        "family": "Gemma", "size": "2B",
        "writer_window": list(range(16, 22)),  # L16-L21
        "total_layers": 26,
    },
    "granite_2b": {
        "model_id": "ibm-granite/granite-3.1-2b-instruct",
        "family": "Granite", "size": "2B",
        "writer_window": list(range(13, 21)),  # L13-L20
        "total_layers": 40,
    },
    "llama_3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "Llama", "size": "3B",
        "writer_window": list(range(14, 18)),  # L14-L17
        "total_layers": 28,
    },
    "qwen_7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "Qwen", "size": "7B",
        "writer_window": list(range(24, 28)),  # L24-L27
        "total_layers": 28,
    },
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "Mistral", "size": "7B",
        "writer_window": list(range(19, 23)),  # L19-L22
        "total_layers": 32,
    },
    "llama_8b": {
        "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
        "family": "Llama", "size": "8B",
        "writer_window": list(range(14, 25)),  # L14-L24
        "total_layers": 32,
    },
    "granite_8b": {
        "model_id": "ibm-granite/granite-3.1-8b-instruct",
        "family": "Granite", "size": "8B",
        "writer_window": list(range(26, 40)),  # L26-L39
        "total_layers": 40,
    },
    "gemma_9b": {
        "model_id": "google/gemma-2-9b-it",
        "family": "Gemma", "size": "9B",
        "writer_window": list(range(26, 30)),  # L26-L29
        "total_layers": 42,
    },
    "mistral_12b": {
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "family": "Mistral", "size": "12B",
        "writer_window": list(range(19, 25)),  # L19-L24
        "total_layers": 40,
    },
    "qwen_14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "family": "Qwen", "size": "14B",
        "writer_window": list(range(32, 48)),  # L32-L47
        "total_layers": 48,
    },
}

# Ordering by VRAM usage (smallest first)
CELL_ORDER = [
    "gemma_2b", "granite_2b", "llama_3b",
    "qwen_7b", "mistral_7b", "llama_8b", "granite_8b",
    "gemma_9b", "mistral_12b", "qwen_14b",
]

# Representative subset (one per family, prioritize smaller for speed)
REPRESENTATIVE_CELLS = ["gemma_2b", "granite_2b", "llama_3b", "qwen_7b", "mistral_7b"]

FACT_LABELS = list("ABCDEF")

WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parents[2]
OUTPUT_DIR = WORKSPACE / "custom_outputs" / "reviewer_controls"


def ensure_imports():
    for p in [str(REPO_ROOT), str(REPO_ROOT / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def jsonl_load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bootstrap_ci(values: List[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> Dict[str, float]:
    arr = np.array(values)
    mean = float(arr.mean())
    if len(arr) < 2:
        return {"mean": mean, "ci_lo": mean, "ci_hi": mean, "n": len(arr)}
    rng = np.random.RandomState(seed)
    boot = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    boot.sort()
    return {
        "mean": mean,
        "ci_lo": boot[int(n_boot * alpha / 2)],
        "ci_hi": boot[int(n_boot * (1 - alpha / 2))],
        "n": len(arr),
    }


def permutation_pvalue(values: List[float], n_perm: int = 10000, seed: int = 42) -> float:
    """One-sided sign-flip permutation test: H1 mean < 0."""
    arr = np.array(values)
    obs = arr.mean()
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(arr))
        if (arr * signs).mean() <= obs:
            count += 1
    return count / n_perm


def example_level_aggregate(df: pd.DataFrame, effect_col: str) -> Dict[str, float]:
    """Aggregate prompt-level effects to example level, then compute stats."""
    ex = df.groupby("example_id")[effect_col].mean().tolist()
    ci = bootstrap_ci(ex)
    ci["perm_p"] = permutation_pvalue(ex)
    return ci


# ---------------------------------------------------------------------------
# Prompt position identification
# ---------------------------------------------------------------------------
def fact_line(fact: Dict[str, Any]) -> str:
    return f"{fact['label']}. [{fact['title']}] {fact['text']}"


def get_span_positions(tokenizer, prompt_text: str, spans: Dict[str, Tuple[int, int]]) -> Dict[str, List[int]]:
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    out = {}
    for name, (s, e) in spans.items():
        positions = [i for i, (ts, te) in enumerate(offsets) if ts < e and te > s]
        out[name] = positions
    return out


def identify_all_positions(tokenizer, row: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    """Identify token positions for all span types needed by reviewer experiments."""
    facts = row["facts"]
    gold_labels = set(row["support_labels"])

    # Fact spans (text only, excluding label token and title brackets)
    fact_spans = {}
    gold_text_positions = []
    distractor_text_positions = []
    all_fact_positions = []

    for fact in facts:
        line = fact_line(fact)
        start = prompt_text.index(line)
        # Text-only span: skip "X. [Title] " prefix
        text_part = fact["text"]
        text_start = prompt_text.index(text_part, start)
        text_end = text_start + len(text_part)
        fact_spans[f"fact_{fact['label']}_text"] = (text_start, text_end)
        # Full line span (for position counting)
        fact_spans[f"fact_{fact['label']}_full"] = (start, start + len(line))

    positions = get_span_positions(tokenizer, prompt_text, fact_spans)

    for fact in facts:
        label = fact["label"]
        text_pos = positions[f"fact_{label}_text"]
        full_pos = positions[f"fact_{label}_full"]
        all_fact_positions.extend(full_pos)
        if label in gold_labels:
            gold_text_positions.extend(text_pos)
        else:
            distractor_text_positions.extend(text_pos)

    # Question span
    question = row["question"]
    q_start = prompt_text.index(question)
    q_spans = {"question": (q_start, q_start + len(question))}
    q_positions = get_span_positions(tokenizer, prompt_text, q_spans)

    # Random non-fact, non-question spans (matched token count to gold)
    all_occupied = set(all_fact_positions + q_positions["question"])
    total_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    free_positions = [i for i in range(total_tokens) if i not in all_occupied]
    n_gold = len(gold_text_positions)
    rng = pyrandom.Random(hash(row["prompt_id"]) & 0xFFFFFFFF)
    if len(free_positions) >= n_gold:
        random_positions = sorted(rng.sample(free_positions, n_gold))
    else:
        random_positions = free_positions

    return {
        "gold_text": sorted(gold_text_positions),
        "distractor_text": sorted(distractor_text_positions),
        "all_facts": sorted(set(all_fact_positions)),
        "question": sorted(q_positions["question"]),
        "random_matched": sorted(random_positions),
        "n_gold_tokens": n_gold,
        "n_distractor_tokens": len(distractor_text_positions),
        "n_total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Core ablation runner (reuses custom_experiment.py hook approach)
# ---------------------------------------------------------------------------
def resolve_label_token_ids(tokenizer, labels=FACT_LABELS) -> Dict[str, int]:
    out = {}
    for label in labels:
        token_id = None
        for text in [label, f" {label}", f"\n{label}"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                token_id = int(ids[0])
                if text == label:
                    break
        if token_id is None:
            raise RuntimeError(f"Cannot resolve single-token id for label={label}")
        out[label] = token_id
    return out


def compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 0)


def greedy_decode_with_ablation(
    model, tokenizer, row, device,
    hook_layers: List[int], hook_positions: List[int],
    adapter, base_model,
) -> Tuple[List[str], bool]:
    """Decode two labels with value projection zeroing. Returns (labels, correct)."""
    label_token_ids = resolve_label_token_ids(tokenizer)
    gold_set = set(row["support_labels"])
    prediction = []
    prefix = ""

    # Setup hooks
    hooks = []
    pos_tensor = torch.tensor(hook_positions, device=device, dtype=torch.long)

    def make_hook(layer_idx):
        def fn(module, inp, output):
            out = output.clone()
            # Clamp positions to actual sequence length
            valid = pos_tensor[pos_tensor < out.shape[1]]
            if valid.numel() > 0:
                out[:, valid, :] = 0.0
            return out
        return fn

    def _get_v_proj(layer_idx):
        layers = adapter.get_layers(base_model)
        attn = adapter.get_layer_attention(layers[layer_idx])
        if hasattr(attn, "v_proj"):
            return attn.v_proj
        raise AttributeError(f"No v_proj at layer {layer_idx}")

    for li in hook_layers:
        v_proj = _get_v_proj(li)
        hooks.append(v_proj.register_forward_hook(make_hook(li)))

    try:
        for step in range(2):
            prompt_text = row["prompt"] + prefix
            tok = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = tok["input_ids"].to(device)
            attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(device)
            pos_ids = compute_position_ids(attn_mask).to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            position_ids=pos_ids, use_cache=False, return_dict=True)
            logits = out.logits[0, -1].float().detach()
            label_logits = {l: float(logits[tid]) for l, tid in label_token_ids.items()}
            predicted = max(label_logits, key=label_logits.get)
            prediction.append(predicted)
            if step == 0:
                prefix = f"{prediction[0]},"
    finally:
        for h in hooks:
            h.remove()

    correct = set(prediction) == gold_set and len(prediction) == 2
    return prediction, correct


def greedy_decode_baseline(model, tokenizer, row, device) -> Tuple[List[str], bool]:
    """Decode two labels without any ablation."""
    label_token_ids = resolve_label_token_ids(tokenizer)
    gold_set = set(row["support_labels"])
    prediction = []
    prefix = ""
    for step in range(2):
        prompt_text = row["prompt"] + prefix
        tok = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(device)
        attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(device)
        pos_ids = compute_position_ids(attn_mask).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        position_ids=pos_ids, use_cache=False, return_dict=True)
        logits = out.logits[0, -1].float().detach()
        label_logits = {l: float(logits[tid]) for l, tid in label_token_ids.items()}
        predicted = max(label_logits, key=label_logits.get)
        prediction.append(predicted)
        if step == 0:
            prefix = f"{prediction[0]},"
    correct = set(prediction) == gold_set and len(prediction) == 2
    return prediction, correct


# ---------------------------------------------------------------------------
# Observational readout (for baselines experiment)
# ---------------------------------------------------------------------------
def compute_readout_scores(
    model, tokenizer, adapter, base_model, row, device,
    window_layers: List[int],
) -> Dict[str, Dict[str, float]]:
    """Compute multiple readout methods for a single prompt.

    Returns dict of method_name -> {fact_label: score}.
    Methods: positive_aligned (proposed), signed_sum, attention_only, value_norm.
    """
    gold_labels = row["support_labels"]
    facts = row["facts"]
    prompt_text = row["prompt"]

    # Get fact positions (text-only)
    fact_spans = {}
    for fact in facts:
        text = fact["text"]
        start = prompt_text.index(text, prompt_text.index(fact_line(fact)))
        fact_spans[fact["label"]] = (start, start + len(text))
    positions = get_span_positions(tokenizer, prompt_text, fact_spans)

    # Encode and get model outputs
    tok = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(device)
    pos_ids = compute_position_ids(attn_mask).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attn_mask, position_ids=pos_ids,
            output_hidden_states=True, output_attentions=True,
            use_cache=False, return_dict=True,
        )

    target_pos = int(input_ids.shape[1] - 1)

    # Get target label direction (first gold label)
    target_label = gold_labels[0]
    label_token_ids = resolve_label_token_ids(tokenizer)
    output_emb = model.get_output_embeddings()
    unembed = output_emb.weight.detach().to(device).float()
    target_dir = unembed[label_token_ids[target_label]]
    target_dir = target_dir / target_dir.norm().clamp(min=1e-8)

    layers = adapter.get_layers(base_model)

    # Initialize score accumulators
    positive_aligned = {l: 0.0 for l in FACT_LABELS}
    signed_sum = {l: 0.0 for l in FACT_LABELS}
    attention_only = {l: 0.0 for l in FACT_LABELS}
    value_norm = {l: 0.0 for l in FACT_LABELS}

    for layer_idx in window_layers:
        resid_pre = outputs.hidden_states[layer_idx].detach()
        attn_input = adapter.attn_input(layers[layer_idx], resid_pre)
        layer_decomp = adapter.decompose_layer(
            layers[layer_idx], attn_input, pos_ids, base_model=base_model
        )
        attn_probs = outputs.attentions[layer_idx][0].float()

        q_heads = layer_decomp.q.shape[0]
        for qh in range(q_heads):
            kvh = adapter.q_to_kv_head(layers[layer_idx], qh)
            alpha = attn_probs[qh, target_pos, :].detach().float()
            v_src = layer_decomp.v[kvh, :, :].detach().float()
            o_block = layer_decomp.o_weight[qh].detach().float()
            writer = torch.matmul(v_src, o_block)
            writer_proj = torch.matmul(writer, target_dir)
            signed_contrib = alpha * writer_proj
            v_norms = writer.norm(dim=-1)

            for label in FACT_LABELS:
                pos = positions.get(label, [])
                if not pos:
                    continue
                pt = torch.tensor(pos, device=device, dtype=torch.long)
                pt = pt[pt < alpha.shape[0]]
                if pt.numel() == 0:
                    continue

                sc = signed_contrib[pt]
                positive_aligned[label] += float(sc.clamp(min=0).sum().item())
                signed_sum[label] += float(sc.sum().item())
                attention_only[label] += float(alpha[pt].sum().item())
                value_norm[label] += float(v_norms[pt].sum().item())

    return {
        "positive_aligned": positive_aligned,
        "signed_sum": signed_sum,
        "attention_only": attention_only,
        "value_norm": value_norm,
    }


def top_k_labels(scores: Dict[str, float], k: int = 2) -> List[str]:
    return [l for l, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_for_cell(cell_key: str):
    """Load model, tokenizer, adapter, base_model for a cell."""
    ensure_imports()
    from src.hook_registry import resolve_adapter
    from src.load_model import load_model_and_tokenizer

    cfg = CELL_CONFIGS[cell_key]
    loaded = load_model_and_tokenizer(
        model_id=cfg["model_id"],
        trust_remote_code=True, dtype="auto", device="auto",
    )
    adapter = resolve_adapter(loaded.model)
    base_model = adapter.get_base_model(loaded.model)
    return loaded, adapter, base_model


def _rerender_prompt_safe(row: Dict[str, Any], tokenizer, max_length: int) -> Optional[str]:
    """Re-render a prompt for a different model's chat template, handling system-role fallback."""
    sys.path.insert(0, str(WORKSPACE))
    from custom_experiment import (
        render_user_prompt, PROMPT_FAMILY_SPECS, SYSTEM_PROMPT,
        FEVER_FAMILY_SPECS, FEVER_SYSTEM_PROMPT, render_fever_user_prompt,
    )

    family_spec_key = row["family_name"]
    pack_key = row["lexical_pack"]
    is_fever = row.get("dataset") == "fever"

    if is_fever:
        family_spec = FEVER_FAMILY_SPECS[pack_key][family_spec_key]
        user_prompt = render_fever_user_prompt(
            claim=str(row["question"]).strip(), facts=row["facts"], family_spec=family_spec,
        )
        system_prompt = FEVER_SYSTEM_PROMPT
    else:
        family_spec = PROMPT_FAMILY_SPECS[pack_key][family_spec_key]
        user_prompt = render_user_prompt(
            question=str(row["question"]).strip(), facts=row["facts"], family_spec=family_spec,
        )
        system_prompt = SYSTEM_PROMPT

    # Try with system role first
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback: prepend system to user message
        combined = f"{system_prompt}\n\n{user_prompt}"
        messages = [{"role": "user", "content": combined}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    prompt_length = len(tokenizer.encode(chat_text, add_special_tokens=False))
    if prompt_length > max_length:
        return None
    return chat_text


def load_prompts(cell_key: str, max_examples: int = 200) -> List[Dict[str, Any]]:
    """Load and re-render prompts for the given cell."""
    from custom_experiment import jsonl_load as ce_jsonl_load, load_plan
    from transformers import AutoTokenizer

    cfg = CELL_CONFIGS[cell_key]
    prompt_path = Path(os.environ.get("D36_PROMPTS_FILE", str(WORKSPACE / "prompts.jsonl")))
    rows = ce_jsonl_load(prompt_path)
    min_rank = int(os.environ.get("D36_EXAMPLE_RANK_MIN", "1"))
    max_rank = int(os.environ.get("D36_EXAMPLE_RANK_MAX", str(max_examples)))

    # Filter to headline families + both packs, up to max_examples
    selected = [
        r for r in rows
        if r["family_name"] in ("active_set", "working_set")
        and min_rank <= int(r["example_rank"]) <= max_rank
    ]

    # Re-render for model if needed
    plan = load_plan(WORKSPACE)
    orig_model = str(plan["experiment"]["model_id"])
    max_length = int(plan["experiment"].get("max_length", 768))

    if cfg["model_id"] != orig_model:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True, use_fast=True
        )
        rerendered = []
        for r in selected:
            new_prompt = _rerender_prompt_safe(r, tokenizer, max_length)
            if new_prompt is not None:
                r = dict(r)
                r["prompt"] = new_prompt
                rerendered.append(r)
        selected = rerendered

    return selected


# ===================================================================
# EXPERIMENT 1: Specificity Controls
# ===================================================================
def run_specificity_controls(cell_key: str, max_examples: int = 200):
    """Run matched ablation controls for a single model cell."""
    cfg = CELL_CONFIGS[cell_key]
    print(f"\n{'='*60}")
    print(f"SPECIFICITY CONTROLS: {cell_key} ({cfg['model_id']})")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    writer_window = cfg["writer_window"]
    total_layers = cfg["total_layers"]
    w_width = len(writer_window)

    # Matched-width early window (same number of layers, centered in early region)
    w_start = min(writer_window)
    early_center = w_start // 2
    early_start = max(0, early_center - w_width // 2)
    early_window = list(range(early_start, min(early_start + w_width, w_start)))
    if not early_window:
        early_window = list(range(min(w_width, w_start)))

    # Matched-width late window (after writer window, if layers exist)
    w_end = max(writer_window) + 1
    late_window = list(range(w_end, min(w_end + w_width, total_layers)))
    if not late_window:
        late_window = early_window  # fallback

    results = []
    for idx, row in enumerate(rows):
        if idx % 100 == 0:
            print(f"  Processing prompt {idx+1}/{len(rows)}...")

        pos = identify_all_positions(loaded.tokenizer, row, row["prompt"])

        # Baseline (no ablation)
        baseline_labels, baseline_correct = greedy_decode_baseline(
            loaded.model, loaded.tokenizer, row, loaded.device
        )

        result = {
            "prompt_id": row["prompt_id"],
            "example_id": row["example_id"],
            "prompt_family": row["prompt_family"],
            "baseline_correct": baseline_correct,
            "baseline_labels": baseline_labels,
            "n_gold_tokens": pos["n_gold_tokens"],
            "n_distractor_tokens": pos["n_distractor_tokens"],
        }

        # Condition 1: Gold facts in writer window (replicates P9)
        _, c1_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=writer_window, hook_positions=pos["gold_text"],
            adapter=adapter, base_model=base_model,
        )
        result["gold_writer_correct"] = c1_correct

        # Condition 2: Distractor facts in writer window
        _, c2_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=writer_window, hook_positions=pos["distractor_text"],
            adapter=adapter, base_model=base_model,
        )
        result["distractor_writer_correct"] = c2_correct

        # Condition 3: Random distractor subset in writer window
        # (length-matched: sample from distractor positions, same count as gold)
        rng = pyrandom.Random(hash(row["prompt_id"]) & 0xFFFFFFFF)
        n_gold = len(pos["gold_text"])
        if len(pos["distractor_text"]) >= n_gold:
            random_dist_pos = sorted(rng.sample(pos["distractor_text"], n_gold))
        else:
            random_dist_pos = pos["distractor_text"]
        _, c3_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=writer_window, hook_positions=random_dist_pos,
            adapter=adapter, base_model=base_model,
        )
        result["random_distractor_writer_correct"] = c3_correct

        # Condition 4: Gold facts in EARLY window (matched width)
        _, c4_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=early_window, hook_positions=pos["gold_text"],
            adapter=adapter, base_model=base_model,
        )
        result["gold_early_correct"] = c4_correct

        # Condition 5: Gold facts in LATE window (matched width, after writer)
        _, c5_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=late_window, hook_positions=pos["gold_text"],
            adapter=adapter, base_model=base_model,
        )
        result["gold_late_correct"] = c5_correct

        # Condition 6: Question tokens in writer window
        _, c6_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=writer_window, hook_positions=pos["question"],
            adapter=adapter, base_model=base_model,
        )
        result["question_writer_correct"] = c6_correct

        # Condition 7: ALL facts in writer window
        _, c7_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=writer_window, hook_positions=pos["all_facts"],
            adapter=adapter, base_model=base_model,
        )
        result["all_facts_writer_correct"] = c7_correct

        results.append(result)

    df = pd.DataFrame(results)

    # Compute effects (conditional on baseline correct)
    conditions = [
        "gold_writer", "distractor_writer", "random_distractor_writer",
        "gold_early", "gold_late", "question_writer", "all_facts_writer",
    ]
    summary = {"cell_key": cell_key, "model_id": cfg["model_id"],
               "family": cfg["family"], "size": cfg["size"],
               "writer_window": f"L{min(writer_window)}-L{max(writer_window)}",
               "early_window": f"L{min(early_window)}-L{max(early_window)}",
               "late_window": f"L{min(late_window)}-L{max(late_window)}",
               "n_prompts": len(df), "n_examples": df["example_id"].nunique(),
               "baseline_accuracy": float(df["baseline_correct"].mean())}

    correct_mask = df["baseline_correct"]
    n_correct = int(correct_mask.sum())
    summary["n_correct_prompts"] = n_correct

    for cond in conditions:
        col = f"{cond}_correct"
        # All-prompt effect
        all_changes = (df[col].astype(float) - df["baseline_correct"].astype(float)).tolist()
        summary[f"{cond}_all_prompt"] = bootstrap_ci(all_changes)
        summary[f"{cond}_all_prompt"]["accuracy"] = float(df[col].mean())

        # Conditional effect (baseline-correct only)
        if n_correct > 0:
            cond_changes = (df.loc[correct_mask, col].astype(float) - 1.0).tolist()
            summary[f"{cond}_conditional"] = bootstrap_ci(cond_changes)
            summary[f"{cond}_conditional"]["perm_p"] = permutation_pvalue(cond_changes)
            summary[f"{cond}_conditional"]["destruction_rate"] = float(
                1.0 - df.loc[correct_mask, col].mean()
            )

        # Example-level
        summary[f"{cond}_example_level"] = example_level_aggregate(
            df[correct_mask].assign(**{f"effect": lambda d, c=col: d[c].astype(float) - 1.0}),
            "effect"
        ) if n_correct > 0 else {}

    # Save
    out_dir = OUTPUT_DIR / "specificity" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "prompt_results.csv", index=False)
    save_json(out_dir / "summary.json", summary)
    print(f"  Saved to {out_dir}")
    print(f"  Baseline acc: {summary['baseline_accuracy']:.3f}")
    for cond in conditions:
        cond_data = summary.get(f"{cond}_conditional", {})
        print(f"  {cond}: destruction_rate={cond_data.get('destruction_rate', 'N/A'):.3f}, "
              f"mean={cond_data.get('mean', 'N/A'):.3f}")

    # Cleanup GPU
    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# EXPERIMENT 2: Layer Sweep
# ===================================================================
def run_layer_sweep(cell_key: str, max_examples: int = 50, window_width: int = 4):
    """Full layer sweep: ablate gold facts in sliding windows across all layers."""
    cfg = CELL_CONFIGS[cell_key]
    print(f"\n{'='*60}")
    print(f"LAYER SWEEP: {cell_key} ({cfg['model_id']})")
    print(f"Window width: {window_width}, Examples: {max_examples}")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    total_layers = cfg["total_layers"]
    canonical_window = cfg["writer_window"]

    # Generate all windows
    windows = []
    for start in range(total_layers - window_width + 1):
        windows.append(list(range(start, start + window_width)))

    print(f"Sweeping {len(windows)} windows across {total_layers} layers")

    # First pass: compute baselines and positions
    baselines = {}
    all_positions = {}
    for row in rows:
        bl_labels, bl_correct = greedy_decode_baseline(
            loaded.model, loaded.tokenizer, row, loaded.device
        )
        baselines[row["prompt_id"]] = {"labels": bl_labels, "correct": bl_correct}
        pos = identify_all_positions(loaded.tokenizer, row, row["prompt"])
        all_positions[row["prompt_id"]] = pos

    # Sweep windows
    window_results = []
    for wi, window in enumerate(windows):
        if wi % 5 == 0:
            print(f"  Window {wi+1}/{len(windows)} (L{window[0]}-L{window[-1]})...")

        changes = []
        for row in rows:
            bl = baselines[row["prompt_id"]]
            if not bl["correct"]:
                continue
            pos = all_positions[row["prompt_id"]]
            _, abl_correct = greedy_decode_with_ablation(
                loaded.model, loaded.tokenizer, row, loaded.device,
                hook_layers=window, hook_positions=pos["gold_text"],
                adapter=adapter, base_model=base_model,
            )
            changes.append(float(abl_correct) - 1.0)

        if changes:
            ci = bootstrap_ci(changes)
            ci["perm_p"] = permutation_pvalue(changes)
        else:
            ci = {"mean": 0, "ci_lo": 0, "ci_hi": 0, "n": 0, "perm_p": 1.0}

        window_results.append({
            "window_start": window[0],
            "window_end": window[-1],
            "window_str": f"L{window[0]}-L{window[-1]}",
            "is_canonical": window == canonical_window or set(window).issubset(set(canonical_window)),
            **ci,
        })

    sweep_df = pd.DataFrame(window_results)

    # Save
    out_dir = OUTPUT_DIR / "layer_sweep" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_dir / "sweep_results.csv", index=False)

    # Find peak window
    peak = sweep_df.loc[sweep_df["mean"].idxmin()]
    summary = {
        "cell_key": cell_key, "model_id": cfg["model_id"],
        "family": cfg["family"], "size": cfg["size"],
        "total_layers": total_layers,
        "window_width": window_width,
        "n_windows": len(windows),
        "canonical_window": f"L{min(canonical_window)}-L{max(canonical_window)}",
        "peak_window": peak["window_str"],
        "peak_effect": float(peak["mean"]),
        "peak_ci": [float(peak["ci_lo"]), float(peak["ci_hi"])],
        "canonical_in_peak_region": bool(
            abs(int(peak["window_start"]) - min(canonical_window)) <= window_width
        ),
        "n_prompts": len(rows),
        "n_correct_prompts": sum(1 for v in baselines.values() if v["correct"]),
    }
    save_json(out_dir / "summary.json", summary)
    print(f"  Peak window: {peak['window_str']} (effect={peak['mean']:.3f})")
    print(f"  Canonical window: {summary['canonical_window']}")
    print(f"  Canonical near peak: {summary['canonical_in_peak_region']}")

    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# EXPERIMENT 3: Readout Baselines
# ===================================================================
def run_readout_baselines(cell_key: str, max_examples: int = 200):
    """Compare proposed readout with attention-only, value-norm, signed-sum."""
    cfg = CELL_CONFIGS[cell_key]
    print(f"\n{'='*60}")
    print(f"READOUT BASELINES: {cell_key} ({cfg['model_id']})")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    layers = adapter.get_layers(base_model)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    writer_window = cfg["writer_window"]
    # Also test with early window and random window
    total_layers = cfg["total_layers"]
    early_window = list(range(min(4, total_layers)))
    rng = pyrandom.Random(42)
    random_window = sorted(rng.sample(range(total_layers), min(len(writer_window), total_layers)))

    results = []
    for idx, row in enumerate(rows):
        if idx % 100 == 0:
            print(f"  Processing {idx+1}/{len(rows)}...")

        gold_set = set(row["support_labels"])

        # Readout scores in writer window
        scores = compute_readout_scores(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, writer_window,
        )

        # Readout scores in random window
        scores_random = compute_readout_scores(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, random_window,
        )

        # Baseline decoded prediction
        bl_labels, bl_correct = greedy_decode_baseline(
            loaded.model, loaded.tokenizer, row, loaded.device
        )

        result = {
            "prompt_id": row["prompt_id"],
            "example_id": row["example_id"],
            "prompt_family": row["prompt_family"],
            "baseline_correct": bl_correct,
        }

        # Evaluate each readout method in writer window
        for method_name, method_scores in scores.items():
            top2 = top_k_labels(method_scores)
            result[f"{method_name}_writer_top2"] = top2
            result[f"{method_name}_writer_match"] = set(top2) == gold_set

        # Evaluate proposed method in random window (control)
        rand_top2 = top_k_labels(scores_random["positive_aligned"])
        result["positive_aligned_random_top2"] = rand_top2
        result["positive_aligned_random_match"] = set(rand_top2) == gold_set

        results.append(result)

    df = pd.DataFrame(results)

    methods = ["positive_aligned", "signed_sum", "attention_only", "value_norm"]
    summary = {
        "cell_key": cell_key, "model_id": cfg["model_id"],
        "family": cfg["family"], "size": cfg["size"],
        "n_prompts": len(df),
        "decoded_accuracy": float(df["baseline_correct"].mean()),
        "writer_window": f"L{min(writer_window)}-L{max(writer_window)}",
        "random_window": f"L{min(random_window)}-L{max(random_window)}",
    }
    for m in methods:
        col = f"{m}_writer_match"
        summary[f"{m}_writer_accuracy"] = float(df[col].mean())
    summary["positive_aligned_random_accuracy"] = float(
        df["positive_aligned_random_match"].mean()
    )

    out_dir = OUTPUT_DIR / "readout_baselines" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "prompt_results.csv", index=False)
    save_json(out_dir / "summary.json", summary)

    print(f"  Decoded accuracy: {summary['decoded_accuracy']:.3f}")
    for m in methods:
        print(f"  {m} (writer): {summary[f'{m}_writer_accuracy']:.3f}")
    print(f"  positive_aligned (random window): {summary['positive_aligned_random_accuracy']:.3f}")

    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# EXPERIMENT 4: Label Permutation
# ===================================================================
def run_label_permutation(cell_key: str, max_examples: int = 200, n_perms: int = 5):
    """Randomly permute label-to-fact assignments and re-run observational readout."""
    cfg = CELL_CONFIGS[cell_key]
    print(f"\n{'='*60}")
    print(f"LABEL PERMUTATION: {cell_key} ({cfg['model_id']})")
    print(f"N permutations: {n_perms}")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    layers = adapter.get_layers(base_model)

    # Import rendering functions
    sys.path.insert(0, str(WORKSPACE))
    from custom_experiment import (
        render_user_prompt, PROMPT_FAMILY_SPECS, SYSTEM_PROMPT,
    )

    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    writer_window = cfg["writer_window"]
    results = []

    for perm_idx in range(n_perms + 1):  # 0 = original, 1..n = permuted
        print(f"  Permutation {perm_idx}/{n_perms}...")
        rng = pyrandom.Random(42 + perm_idx)

        for row in rows:
            facts = row["facts"]
            gold_set_orig = set(row["support_labels"])

            if perm_idx == 0:
                # Original label assignment
                permuted_facts = facts
                gold_labels = row["support_labels"]
            else:
                # Shuffle labels but keep content in place
                perm_labels = list(FACT_LABELS)
                rng.shuffle(perm_labels)
                permuted_facts = []
                for i, fact in enumerate(facts):
                    pf = dict(fact)
                    pf["label"] = perm_labels[i]
                    permuted_facts.append(pf)
                gold_labels = sorted([pf["label"] for pf in permuted_facts if pf["is_support"]])

            # Re-render prompt with permuted labels
            family_spec = PROMPT_FAMILY_SPECS[row["lexical_pack"]][row["family_name"]]
            user_prompt = render_user_prompt(
                question=row["question"], facts=permuted_facts, family_spec=family_spec,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                chat_text = loaded.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                combined = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
                chat_text = loaded.tokenizer.apply_chat_template(
                    [{"role": "user", "content": combined}],
                    tokenize=False, add_generation_prompt=True,
                )

            # Build modified row
            mod_row = dict(row)
            mod_row["prompt"] = chat_text
            mod_row["facts"] = permuted_facts
            mod_row["support_labels"] = gold_labels

            # Decode
            bl_labels, bl_correct = greedy_decode_baseline(
                loaded.model, loaded.tokenizer, mod_row, loaded.device
            )

            # Readout
            scores = compute_readout_scores(
                loaded.model, loaded.tokenizer, adapter, base_model,
                mod_row, loaded.device, writer_window,
            )
            gold_set = set(gold_labels)
            pa_top2 = top_k_labels(scores["positive_aligned"])

            results.append({
                "prompt_id": row["prompt_id"],
                "example_id": row["example_id"],
                "perm_idx": perm_idx,
                "gold_labels": gold_labels,
                "decoded_correct": bl_correct,
                "readout_match": set(pa_top2) == gold_set,
                "readout_top2": pa_top2,
            })

    df = pd.DataFrame(results)

    # Summarize
    summary = {"cell_key": cell_key, "model_id": cfg["model_id"],
               "n_permutations": n_perms, "n_prompts_per_perm": len(rows)}

    for pidx in range(n_perms + 1):
        mask = df["perm_idx"] == pidx
        label = "original" if pidx == 0 else f"perm_{pidx}"
        summary[label] = {
            "decoded_accuracy": float(df.loc[mask, "decoded_correct"].mean()),
            "readout_accuracy": float(df.loc[mask, "readout_match"].mean()),
        }

    # Test: is readout accuracy stable across permutations?
    orig_acc = summary["original"]["readout_accuracy"]
    perm_accs = [summary[f"perm_{i}"]["readout_accuracy"] for i in range(1, n_perms + 1)]
    summary["readout_stability"] = {
        "original": orig_acc,
        "perm_mean": float(np.mean(perm_accs)),
        "perm_std": float(np.std(perm_accs)),
        "perm_min": float(np.min(perm_accs)),
        "perm_max": float(np.max(perm_accs)),
        "stable": bool(abs(orig_acc - np.mean(perm_accs)) < 0.05),
    }

    out_dir = OUTPUT_DIR / "label_permutation" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    save_json(out_dir / "summary.json", summary)
    print(f"  Original readout: {orig_acc:.3f}")
    print(f"  Permuted readout: {np.mean(perm_accs):.3f} +/- {np.std(perm_accs):.3f}")
    print(f"  Stable: {summary['readout_stability']['stable']}")

    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# EXPERIMENT 5: Unconditional Effect Reanalysis (no GPU needed)
# ===================================================================
def run_unconditional_reanalysis():
    """Reanalyze P9 data to report unconditional + conditional effects side by side."""
    print(f"\n{'='*60}")
    print("UNCONDITIONAL EFFECT REANALYSIS (from P9 data)")
    print(f"{'='*60}")

    p9_dir = WORKSPACE / "custom_outputs" / "phase_p9_scaleup"
    cross_model = json.loads((p9_dir / "p9_cross_model_summary.json").read_text())

    all_cells = []
    for cell in cross_model["cells"]:
        cell_key = cell["cell_key"]
        b1_path = p9_dir / f"{cell_key}_b1_results.csv"
        if not b1_path.exists():
            continue
        df = pd.read_csv(b1_path)

        n_total = len(df)
        n_correct = int(df["baseline_correct"].sum())
        n_abl_correct = int(df["ablation_correct"].sum())

        # All-prompt paired effect
        all_changes = (df["ablation_correct"].astype(float) - df["baseline_correct"].astype(float)).tolist()
        all_ci = bootstrap_ci(all_changes)
        all_ci["perm_p"] = permutation_pvalue(all_changes)

        # Conditional destruction (baseline correct only)
        correct_mask = df["baseline_correct"]
        if correct_mask.sum() > 0:
            cond_changes = (df.loc[correct_mask, "ablation_correct"].astype(float) - 1.0).tolist()
            cond_ci = bootstrap_ci(cond_changes)
            cond_ci["perm_p"] = permutation_pvalue(cond_changes)
            destruction_rate = 1.0 - df.loc[correct_mask, "ablation_correct"].mean()
        else:
            cond_ci = {"mean": 0, "ci_lo": 0, "ci_hi": 0, "n": 0, "perm_p": 1.0}
            destruction_rate = 0

        # Repair rate (baseline wrong → ablation correct)
        wrong_mask = ~df["baseline_correct"]
        if wrong_mask.sum() > 0:
            repair_rate = float(df.loc[wrong_mask, "ablation_correct"].mean())
        else:
            repair_rate = 0

        # Example-level effects
        ex_all = df.groupby("example_id").apply(
            lambda g: (g["ablation_correct"].astype(float) - g["baseline_correct"].astype(float)).mean()
        ).tolist()
        ex_all_ci = bootstrap_ci(ex_all)
        ex_all_ci["perm_p"] = permutation_pvalue(ex_all)

        row = {
            "cell_key": cell_key,
            "model_id": cell["model_id"],
            "family": cell["family"],
            "size": cell["size"],
            "window": cell["ablation_window"],
            "n_prompts": n_total,
            "baseline_accuracy": float(df["baseline_correct"].mean()),
            "ablation_accuracy": float(df["ablation_correct"].mean()),
            "unconditional_accuracy_drop": float(df["baseline_correct"].mean() - df["ablation_correct"].mean()),
            "all_prompt_effect": all_ci,
            "conditional_destruction_rate": destruction_rate,
            "conditional_effect": cond_ci,
            "repair_rate": repair_rate,
            "example_level_unconditional": ex_all_ci,
            "example_level_conditional": cell["example_level"],
            "n_flipped_to_wrong": cell["n_flipped_to_wrong"],
            "n_flipped_to_right": cell["n_flipped_to_right"],
        }
        all_cells.append(row)

        print(f"  {cell_key}: unconditional_drop={row['unconditional_accuracy_drop']:.3f}, "
              f"destruction_rate={destruction_rate:.3f}, "
              f"repair_rate={repair_rate:.3f}")

    summary = {"cells": all_cells}
    out_dir = OUTPUT_DIR / "unconditional"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "unconditional_reanalysis.json", summary)

    # Also make a clean table
    table_rows = []
    for c in all_cells:
        table_rows.append({
            "cell": c["cell_key"],
            "family": c["family"],
            "size": c["size"],
            "baseline_acc": f"{c['baseline_accuracy']:.3f}",
            "ablated_acc": f"{c['ablation_accuracy']:.3f}",
            "uncond_drop": f"{c['unconditional_accuracy_drop']:.3f}",
            "cond_destruction": f"{c['conditional_destruction_rate']:.3f}",
            "repair_rate": f"{c['repair_rate']:.3f}",
            "ex_uncond_mean": f"{c['example_level_unconditional']['mean']:.3f}",
            "ex_cond_mean": f"{c['example_level_conditional']['mean']:.3f}",
            "flipped_wrong": c["n_flipped_to_wrong"],
            "flipped_right": c["n_flipped_to_right"],
        })
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(out_dir / "unconditional_table.csv", index=False)
    print(f"\n  Saved to {out_dir}")

    return summary


# ===================================================================
# EXPERIMENT 6: Readout-Causality Coupling
# ===================================================================
def run_coupling_analysis(cell_key: str, max_examples: int = 200):
    """Stratify examples by writer readout score; check if ablation damage correlates."""
    cfg = CELL_CONFIGS[cell_key]
    print(f"\n{'='*60}")
    print(f"READOUT-CAUSALITY COUPLING: {cell_key} ({cfg['model_id']})")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    writer_window = cfg["writer_window"]
    results = []

    for idx, row in enumerate(rows):
        if idx % 100 == 0:
            print(f"  Processing {idx+1}/{len(rows)}...")

        gold_set = set(row["support_labels"])
        pos = identify_all_positions(loaded.tokenizer, row, row["prompt"])

        # Baseline
        bl_labels, bl_correct = greedy_decode_baseline(
            loaded.model, loaded.tokenizer, row, loaded.device
        )

        # Readout scores
        scores = compute_readout_scores(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, writer_window,
        )
        pa_scores = scores["positive_aligned"]

        # Writer readout margin: gold score minus best distractor score
        gold_score = sum(pa_scores[l] for l in row["support_labels"])
        dist_scores = [pa_scores[l] for l in FACT_LABELS if l not in gold_set]
        best_dist = max(dist_scores) if dist_scores else 0
        writer_margin = gold_score / 2.0 - best_dist

        pa_top2 = top_k_labels(pa_scores)
        readout_match = set(pa_top2) == gold_set

        # Ablation (gold facts in writer window)
        _, abl_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            hook_layers=writer_window, hook_positions=pos["gold_text"],
            adapter=adapter, base_model=base_model,
        )

        results.append({
            "prompt_id": row["prompt_id"],
            "example_id": row["example_id"],
            "baseline_correct": bl_correct,
            "ablation_correct": abl_correct,
            "readout_match": readout_match,
            "writer_margin": writer_margin,
            "gold_score": gold_score,
            "ablation_damage": float(bl_correct) - float(abl_correct),
        })

    df = pd.DataFrame(results)

    # Stratify by writer margin quartiles
    df["margin_quartile"] = pd.qcut(df["writer_margin"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    quartile_stats = {}
    for q, qdf in df.groupby("margin_quartile"):
        correct_in_q = qdf["baseline_correct"]
        if correct_in_q.sum() > 0:
            damage = qdf.loc[correct_in_q, "ablation_damage"].tolist()
            quartile_stats[str(q)] = bootstrap_ci(damage)
            quartile_stats[str(q)]["n_correct"] = int(correct_in_q.sum())
        else:
            quartile_stats[str(q)] = {"mean": 0, "n_correct": 0}

    # Readout-correct vs readout-incorrect stratification
    readout_groups = {}
    for label, mask in [("readout_correct", df["readout_match"]),
                        ("readout_incorrect", ~df["readout_match"])]:
        subset = df[mask & df["baseline_correct"]]
        if len(subset) > 0:
            damage = subset["ablation_damage"].tolist()
            readout_groups[label] = bootstrap_ci(damage)
            readout_groups[label]["n"] = len(subset)

    summary = {
        "cell_key": cell_key, "model_id": cfg["model_id"],
        "n_prompts": len(df),
        "margin_quartile_damage": quartile_stats,
        "readout_stratified_damage": readout_groups,
        "overall_correlation": float(df[df["baseline_correct"]]["writer_margin"].corr(
            df[df["baseline_correct"]]["ablation_damage"]
        )) if df["baseline_correct"].sum() > 1 else None,
    }

    out_dir = OUTPUT_DIR / "coupling" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    save_json(out_dir / "summary.json", summary)

    print(f"  Margin quartile damage:")
    for q, s in quartile_stats.items():
        print(f"    {q}: mean_damage={s.get('mean', 0):.3f} (n_correct={s.get('n_correct', 0)})")
    if readout_groups:
        for g, s in readout_groups.items():
            print(f"  {g}: mean_damage={s.get('mean', 0):.3f} (n={s.get('n', 0)})")
    print(f"  Correlation: {summary['overall_correlation']}")

    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# Main
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Direction 36 reviewer response experiments")
    p.add_argument("--experiment", required=True,
                   choices=["specificity", "layer_sweep", "readout_baselines",
                            "label_permutation", "unconditional", "coupling",
                            "all_gpu", "all_analysis"])
    p.add_argument("--cell", type=str, default=None,
                   help="Cell key (e.g., qwen_7b). Use 'all' for all cells.")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--sweep-examples", type=int, default=50,
                   help="Number of examples for layer sweep (default 50)")
    p.add_argument("--sweep-width", type=int, default=4,
                   help="Window width for layer sweep")
    p.add_argument("--n-perms", type=int, default=5,
                   help="Number of label permutations")
    return p.parse_args()


def main():
    args = parse_args()

    if args.experiment == "unconditional":
        run_unconditional_reanalysis()
        return

    if args.experiment == "all_analysis":
        run_unconditional_reanalysis()
        return

    if args.experiment == "all_gpu":
        # Run all GPU experiments on all cells in order
        for cell in CELL_ORDER:
            print(f"\n\n{'#'*70}")
            print(f"# Processing {cell}")
            print(f"{'#'*70}")
            try:
                run_specificity_controls(cell, args.max_examples)
            except Exception as e:
                print(f"  ERROR in specificity for {cell}: {e}")
            if cell in REPRESENTATIVE_CELLS:
                try:
                    run_layer_sweep(cell, args.sweep_examples, args.sweep_width)
                except Exception as e:
                    print(f"  ERROR in layer_sweep for {cell}: {e}")
                try:
                    run_readout_baselines(cell, args.max_examples)
                except Exception as e:
                    print(f"  ERROR in readout_baselines for {cell}: {e}")
                try:
                    run_label_permutation(cell, min(args.max_examples, 50), args.n_perms)
                except Exception as e:
                    print(f"  ERROR in label_permutation for {cell}: {e}")
                try:
                    run_coupling_analysis(cell, args.max_examples)
                except Exception as e:
                    print(f"  ERROR in coupling for {cell}: {e}")
        return

    # Single cell experiments
    cells = CELL_ORDER if args.cell == "all" else [args.cell]

    for cell in cells:
        if cell not in CELL_CONFIGS:
            print(f"Unknown cell: {cell}. Available: {list(CELL_CONFIGS.keys())}")
            continue

        if args.experiment == "specificity":
            run_specificity_controls(cell, args.max_examples)
        elif args.experiment == "layer_sweep":
            run_layer_sweep(cell, args.sweep_examples, args.sweep_width)
        elif args.experiment == "readout_baselines":
            run_readout_baselines(cell, args.max_examples)
        elif args.experiment == "label_permutation":
            run_label_permutation(cell, min(args.max_examples, 50), args.n_perms)
        elif args.experiment == "coupling":
            run_coupling_analysis(cell, args.max_examples)


if __name__ == "__main__":
    main()
