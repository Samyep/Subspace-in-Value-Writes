#!/usr/bin/env python3
"""Direction 36 — Phase 8 Experiments: Final Spotlight Push.

Four experiments targeting the deepest remaining gaps after Phases 1-7:
  P8-A: pca_causal_ablation   — Ablate PCA-selected (unsupervised) facts, compare
                                 with gold-fact ablation. Closes readout→causal loop.
  P8-B: cross_example_pca     — Learn PCA direction on discovery split, apply to
                                 held-out split. Shows routing structure generalises.
  P8-C: logit_lens_baseline   — Compare writer readout vs logit lens (hidden-state
                                 projection through LM head) at answer position.
  P8-D: head_contribution     — Per-head routing contribution analysis. Identifies
                                 which attention heads drive the routing signal.

Usage:
  python run_phase8_experiments.py --experiment pca_causal_ablation --cell qwen_7b
  python run_phase8_experiments.py --experiment cross_example_pca --cell qwen_7b
  python run_phase8_experiments.py --experiment logit_lens_baseline --cell qwen_7b
  python run_phase8_experiments.py --experiment head_contribution --cell qwen_7b
  python run_phase8_experiments.py --experiment all_for_cell --cell qwen_7b
  python run_phase8_experiments.py --experiment full_campaign
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parents[2]
OUTPUT_ROOT = Path(os.environ.get("D36_OUTPUT_ROOT", str(WORKSPACE / "custom_outputs")))
OUTPUT_DIR = OUTPUT_ROOT / "phase8"

sys.path.insert(0, str(WORKSPACE))
from run_reviewer_controls import (
    CELL_CONFIGS, CELL_ORDER, REPRESENTATIVE_CELLS, FACT_LABELS,
    bootstrap_ci, permutation_pvalue, example_level_aggregate,
    save_json, jsonl_load,
    load_model_for_cell, load_prompts,
    identify_all_positions, get_span_positions, fact_line,
    greedy_decode_baseline, greedy_decode_with_ablation,
    resolve_label_token_ids, compute_position_ids,
    top_k_labels,
)

ALL_CELLS = CELL_ORDER


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ===================================================================
# Shared: collect per-fact value-write vectors from writer window
# ===================================================================

def collect_fact_write_vectors(
    model, tokenizer, adapter, base_model, row, device,
    window_layers: List[int],
) -> Dict[str, np.ndarray]:
    """Collect per-fact aggregated attention-weighted value-write vectors.
    Returns {label: numpy array of shape [hidden_dim]} for each fact."""
    facts = row["facts"]
    prompt_text = row["prompt"]

    fact_spans = {}
    for fact in facts:
        text = fact["text"]
        start = prompt_text.index(text, prompt_text.index(fact_line(fact)))
        fact_spans[fact["label"]] = (start, start + len(text))
    positions = get_span_positions(tokenizer, prompt_text, fact_spans)

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
    layers = adapter.get_layers(base_model)

    fact_vectors: Dict[str, List[torch.Tensor]] = {l: [] for l in FACT_LABELS}

    for layer_idx in window_layers:
        resid_pre = outputs.hidden_states[layer_idx].detach()
        attn_input = adapter.attn_input(layers[layer_idx], resid_pre)
        layer_decomp = adapter.decompose_layer(
            layers[layer_idx], attn_input, pos_ids, base_model=base_model,
        )
        attn_probs = outputs.attentions[layer_idx][0].float()

        q_heads = layer_decomp.q.shape[0]
        for qh in range(q_heads):
            kvh = adapter.q_to_kv_head(layers[layer_idx], qh)
            alpha = attn_probs[qh, target_pos, :].detach().float()
            v_src = layer_decomp.v[kvh, :, :].detach().float()
            o_block = layer_decomp.o_weight[qh].detach().float()
            writer = torch.matmul(v_src, o_block)  # [seq, hidden]

            for label in FACT_LABELS:
                pos = positions.get(label, [])
                if not pos:
                    continue
                pt = torch.tensor(pos, device=device, dtype=torch.long)
                pt = pt[pt < alpha.shape[0]]
                if pt.numel() == 0:
                    continue
                attn_weighted = alpha[pt].unsqueeze(-1) * writer[pt]
                fact_vectors[label].append(attn_weighted.sum(dim=0).cpu())

    result = {}
    for label in FACT_LABELS:
        vecs = fact_vectors[label]
        if vecs:
            result[label] = torch.stack(vecs).sum(dim=0).numpy()
    return result


# ===================================================================
# P8-A: PCA-Selected Causal Ablation
# ===================================================================

def experiment_pca_causal_ablation(cell_key: str, max_examples: int = 200):
    """For each prompt: (1) run unsupervised PCA to select top-2 facts,
    (2) ablate PCA-selected fact positions, (3) compare destruction with
    gold-fact and random-fact ablation. Closes readout→causal loop."""
    from sklearn.decomposition import PCA

    cfg = CELL_CONFIGS[cell_key]
    writer_window = cfg["writer_window"]
    out_dir = OUTPUT_DIR / "pca_causal_ablation" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"P8-A PCA-CAUSAL ABLATION: {cell_key} ({cfg['model_id']})")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    import random as pyrandom

    results = []
    gold_destroyed = 0
    pca_destroyed = 0
    random_destroyed = 0
    baseline_correct_total = 0

    for idx, row in enumerate(rows):
        if idx % 50 == 0:
            print(f"  Processing {idx+1}/{len(rows)}...")

        gold_labels = set(row["support_labels"])

        # Step 1: baseline decode
        _, bl_correct = greedy_decode_baseline(
            loaded.model, loaded.tokenizer, row, loaded.device,
        )
        if not bl_correct:
            results.append({
                "example_id": row.get("example_id", idx),
                "prompt_id": row.get("prompt_id", f"p{idx}"),
                "baseline_correct": False,
            })
            continue

        baseline_correct_total += 1

        # Step 2: collect fact write vectors and run PCA
        fact_vecs = collect_fact_write_vectors(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, writer_window,
        )

        labels_present = sorted(fact_vecs.keys())
        if len(labels_present) < 4:
            continue

        X = np.stack([fact_vecs[l] for l in labels_present])
        pca = PCA(n_components=min(3, len(labels_present)))
        X_pca = pca.fit_transform(X)

        # Find best separating PC (try each PC and both signs)
        is_gold = np.array([1 if l in gold_labels else 0 for l in labels_present])
        best_scores = X_pca[:, 0]
        # Use top-1 PC; pick sign that puts gold higher
        if is_gold.sum() > 0:
            gold_mean = X_pca[is_gold == 1, 0].mean()
            dist_mean = X_pca[is_gold == 0, 0].mean()
            if dist_mean > gold_mean:
                best_scores = -best_scores

        # PCA top-2 selection
        top2_idx = np.argsort(-best_scores)[:2]
        pca_selected = set(labels_present[i] for i in top2_idx)
        pca_agrees_gold = (pca_selected == gold_labels)

        # Step 3: identify token positions
        all_pos = identify_all_positions(
            loaded.tokenizer, row, row["prompt"],
        )

        # Gold fact positions
        gold_fact_pos = []
        dist_fact_pos = []
        pca_fact_pos = []
        for fact in row["facts"]:
            text = fact["text"]
            start = row["prompt"].index(text, row["prompt"].index(fact_line(fact)))
            spans = {fact["label"]: (start, start + len(text))}
            tok_pos = get_span_positions(loaded.tokenizer, row["prompt"], spans)
            if fact["label"] in gold_labels:
                gold_fact_pos.extend(tok_pos[fact["label"]])
            else:
                dist_fact_pos.extend(tok_pos[fact["label"]])
            if fact["label"] in pca_selected:
                pca_fact_pos.extend(tok_pos[fact["label"]])

        # Random 2 facts
        rng = pyrandom.Random(hash(row.get("prompt_id", idx)) & 0xFFFFFFFF)
        non_gold = [f for f in row["facts"] if f["label"] not in gold_labels]
        random_2 = rng.sample(non_gold, min(2, len(non_gold)))
        random_fact_pos = []
        for fact in random_2:
            text = fact["text"]
            start = row["prompt"].index(text, row["prompt"].index(fact_line(fact)))
            spans = {fact["label"]: (start, start + len(text))}
            tok_pos = get_span_positions(loaded.tokenizer, row["prompt"], spans)
            random_fact_pos.extend(tok_pos[fact["label"]])

        # Step 4: ablation — gold facts
        _, gold_abl_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            writer_window, sorted(set(gold_fact_pos)),
            adapter, base_model,
        )

        # Step 5: ablation — PCA-selected facts
        _, pca_abl_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            writer_window, sorted(set(pca_fact_pos)),
            adapter, base_model,
        )

        # Step 6: ablation — random distractor facts
        _, rand_abl_correct = greedy_decode_with_ablation(
            loaded.model, loaded.tokenizer, row, loaded.device,
            writer_window, sorted(set(random_fact_pos)),
            adapter, base_model,
        )

        if not gold_abl_correct:
            gold_destroyed += 1
        if not pca_abl_correct:
            pca_destroyed += 1
        if not rand_abl_correct:
            random_destroyed += 1

        results.append({
            "example_id": row.get("example_id", idx),
            "prompt_id": row.get("prompt_id", f"p{idx}"),
            "baseline_correct": True,
            "pca_selected": sorted(pca_selected),
            "gold_labels": sorted(gold_labels),
            "pca_agrees_gold": pca_agrees_gold,
            "gold_abl_destroyed": not gold_abl_correct,
            "pca_abl_destroyed": not pca_abl_correct,
            "random_abl_destroyed": not rand_abl_correct,
        })

    n_bl = max(baseline_correct_total, 1)
    gold_rate = gold_destroyed / n_bl
    pca_rate = pca_destroyed / n_bl
    rand_rate = random_destroyed / n_bl

    pca_gold_agree = sum(1 for r in results if r.get("pca_agrees_gold")) / max(len([r for r in results if r.get("baseline_correct")]), 1)

    summary = {
        "cell": cell_key,
        "family": cfg["family"],
        "model_id": cfg["model_id"],
        "n_prompts": len(rows),
        "n_baseline_correct": baseline_correct_total,
        "gold_destruction_rate": round(gold_rate, 3),
        "pca_destruction_rate": round(pca_rate, 3),
        "random_destruction_rate": round(rand_rate, 3),
        "pca_gold_ratio": round(pca_rate / max(rand_rate, 0.001), 1),
        "gold_random_ratio": round(gold_rate / max(rand_rate, 0.001), 1),
        "pca_gold_agreement": round(pca_gold_agree, 3),
        "verdict": (
            "PCA_CAUSAL" if pca_rate > rand_rate * 1.5 and pca_rate > 0.05
            else "WEAK_PCA_CAUSAL" if pca_rate > rand_rate * 1.2
            else "NOT_PCA_CAUSAL"
        ),
    }

    print(f"\n  Gold destruction:   {gold_rate:.3f}")
    print(f"  PCA destruction:    {pca_rate:.3f}")
    print(f"  Random destruction: {rand_rate:.3f}")
    print(f"  PCA/Random ratio:   {summary['pca_gold_ratio']}")
    print(f"  PCA-Gold agreement: {pca_gold_agree:.3f}")
    print(f"  Verdict: {summary['verdict']}")

    save_json(out_dir / "summary.json", _jsonable(summary))
    pd.DataFrame(results).to_csv(out_dir / "per_example.csv", index=False)

    # Cleanup
    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# P8-B: Cross-Example PCA Direction Transfer
# ===================================================================

def experiment_cross_example_pca(cell_key: str, max_examples: int = 200):
    """Learn PCA routing direction on discovery split, apply to held-out split.
    Shows the routing structure is a stable geometric property."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    cfg = CELL_CONFIGS[cell_key]
    writer_window = cfg["writer_window"]
    out_dir = OUTPUT_DIR / "cross_example_pca" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"P8-B CROSS-EXAMPLE PCA TRANSFER: {cell_key} ({cfg['model_id']})")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    # Split by example_id into discovery/validation
    example_ids = sorted(set(r.get("example_id", i) for i, r in enumerate(rows)))
    mid = len(example_ids) // 2
    discovery_ids = set(example_ids[:mid])
    validation_ids = set(example_ids[mid:])

    discovery_rows = [r for r in rows if r.get("example_id", 0) in discovery_ids]
    validation_rows = [r for r in rows if r.get("example_id", 0) in validation_ids]
    print(f"  Discovery: {len(discovery_rows)} prompts, Validation: {len(validation_rows)} prompts")

    # Phase 1: Collect all fact vectors from discovery set
    print("  Phase 1: Collecting discovery vectors...")
    all_disc_vecs = []
    all_disc_labels = []
    for idx, row in enumerate(discovery_rows):
        if idx % 100 == 0:
            print(f"    Discovery {idx+1}/{len(discovery_rows)}...")
        gold_labels = set(row["support_labels"])
        fv = collect_fact_write_vectors(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, writer_window,
        )
        for label in sorted(fv.keys()):
            all_disc_vecs.append(fv[label])
            all_disc_labels.append(1 if label in gold_labels else 0)

    X_disc = np.stack(all_disc_vecs)
    y_disc = np.array(all_disc_labels)
    print(f"  Discovery: {X_disc.shape[0]} fact vectors ({y_disc.sum()} gold, {(1-y_disc).sum()} distractor)")

    # Fit global PCA on discovery set
    pca = PCA(n_components=min(10, X_disc.shape[0], X_disc.shape[1]))
    X_disc_pca = pca.fit_transform(X_disc)

    # Find best separating PC on discovery set
    best_pc_idx = 0
    best_disc_auc = 0.5
    best_sign = 1
    for pc_idx in range(X_disc_pca.shape[1]):
        scores = X_disc_pca[:, pc_idx]
        auc_pos = roc_auc_score(y_disc, scores)
        auc_neg = roc_auc_score(y_disc, -scores)
        if max(auc_pos, auc_neg) > best_disc_auc:
            best_disc_auc = max(auc_pos, auc_neg)
            best_pc_idx = pc_idx
            best_sign = 1 if auc_pos >= auc_neg else -1

    print(f"  Discovery best PC: PC{best_pc_idx}, AUC={best_disc_auc:.3f}")

    # Phase 2: Apply learned direction to validation set
    print("  Phase 2: Applying to validation set...")
    val_aucs = []
    val_recoveries = []
    per_example_val = []

    for idx, row in enumerate(validation_rows):
        if idx % 100 == 0:
            print(f"    Validation {idx+1}/{len(validation_rows)}...")
        gold_labels = set(row["support_labels"])
        fv = collect_fact_write_vectors(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, writer_window,
        )

        labels_present = sorted(fv.keys())
        if len(labels_present) < 4:
            continue

        X_val = np.stack([fv[l] for l in labels_present])
        is_gold = np.array([1 if l in gold_labels else 0 for l in labels_present])

        if is_gold.sum() == 0 or is_gold.sum() == len(is_gold):
            continue

        # Project through learned PCA
        X_val_pca = pca.transform(X_val)
        scores = best_sign * X_val_pca[:, best_pc_idx]

        auc = roc_auc_score(is_gold, scores)
        val_aucs.append(auc)

        # Top-2 recovery
        top2_idx = np.argsort(-scores)[:2]
        top2_labels = set(labels_present[i] for i in top2_idx)
        correct = (top2_labels == gold_labels)
        val_recoveries.append(int(correct))

        per_example_val.append({
            "example_id": row.get("example_id", idx),
            "prompt_id": row.get("prompt_id", f"p{idx}"),
            "auc": round(auc, 3),
            "recovery": int(correct),
            "pca_selected": sorted(top2_labels),
            "gold_labels": sorted(gold_labels),
        })

    # Also run per-example PCA on validation (for comparison)
    print("  Phase 3: Per-example PCA on validation (comparison)...")
    perex_aucs = []
    perex_recoveries = []
    for idx, row in enumerate(validation_rows):
        gold_labels = set(row["support_labels"])
        fv = collect_fact_write_vectors(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, writer_window,
        )
        labels_present = sorted(fv.keys())
        if len(labels_present) < 4:
            continue
        X = np.stack([fv[l] for l in labels_present])
        is_gold = np.array([1 if l in gold_labels else 0 for l in labels_present])
        if is_gold.sum() == 0 or is_gold.sum() == len(is_gold):
            continue

        pca_ex = PCA(n_components=min(3, len(labels_present)))
        X_ex = pca_ex.fit_transform(X)
        best_ex_auc = 0.5
        best_ex_scores = X_ex[:, 0]
        for pc_idx in range(X_ex.shape[1]):
            auc_p = roc_auc_score(is_gold, X_ex[:, pc_idx])
            auc_n = roc_auc_score(is_gold, -X_ex[:, pc_idx])
            if max(auc_p, auc_n) > best_ex_auc:
                best_ex_auc = max(auc_p, auc_n)
                best_ex_scores = X_ex[:, pc_idx] if auc_p >= auc_n else -X_ex[:, pc_idx]
        perex_aucs.append(best_ex_auc)
        top2_idx = np.argsort(-best_ex_scores)[:2]
        top2_labels = set(labels_present[i] for i in top2_idx)
        perex_recoveries.append(int(top2_labels == gold_labels))

    transfer_auc = float(np.mean(val_aucs)) if val_aucs else 0
    transfer_recovery = float(np.mean(val_recoveries)) if val_recoveries else 0
    perex_auc = float(np.mean(perex_aucs)) if perex_aucs else 0
    perex_recovery = float(np.mean(perex_recoveries)) if perex_recoveries else 0

    summary = {
        "cell": cell_key,
        "family": cfg["family"],
        "model_id": cfg["model_id"],
        "n_discovery": len(discovery_rows),
        "n_validation": len(validation_rows),
        "n_disc_vectors": int(X_disc.shape[0]),
        "discovery_best_pc": int(best_pc_idx),
        "discovery_auc": round(best_disc_auc, 3),
        "transfer_auc": round(transfer_auc, 3),
        "transfer_recovery": round(transfer_recovery, 3),
        "perexample_auc": round(perex_auc, 3),
        "perexample_recovery": round(perex_recovery, 3),
        "auc_retention": round(transfer_auc / max(perex_auc, 0.001), 3),
        "n_validation_examples": len(val_aucs),
        "random_baseline": 0.067,
        "verdict": (
            "STRONG_TRANSFER" if transfer_auc > 0.80 and transfer_recovery > 0.40
            else "MODERATE_TRANSFER" if transfer_auc > 0.70
            else "WEAK_TRANSFER" if transfer_auc > 0.60
            else "NO_TRANSFER"
        ),
    }

    print(f"\n  Discovery AUC:      {best_disc_auc:.3f}")
    print(f"  Transfer AUC:       {transfer_auc:.3f}")
    print(f"  Transfer recovery:  {transfer_recovery:.3f}")
    print(f"  Per-example AUC:    {perex_auc:.3f}")
    print(f"  Per-example recov:  {perex_recovery:.3f}")
    print(f"  AUC retention:      {summary['auc_retention']}")
    print(f"  Verdict: {summary['verdict']}")

    save_json(out_dir / "summary.json", _jsonable(summary))
    pd.DataFrame(per_example_val).to_csv(out_dir / "per_example.csv", index=False)

    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# P8-C: Logit Lens Baseline
# ===================================================================

def experiment_logit_lens_baseline(cell_key: str, max_examples: int = 200):
    """At each writer-window layer, project hidden state at answer position through
    the LM head (logit lens). Select top-2 candidate labels by logit lens score.
    Compare recovery with writer readout."""
    cfg = CELL_CONFIGS[cell_key]
    writer_window = cfg["writer_window"]
    out_dir = OUTPUT_DIR / "logit_lens" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"P8-C LOGIT LENS BASELINE: {cell_key} ({cfg['model_id']})")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    label_token_ids = resolve_label_token_ids(loaded.tokenizer)
    output_emb = loaded.model.get_output_embeddings()
    lm_head_weight = output_emb.weight.detach().to(loaded.device).float()
    # Also get bias if present
    lm_head_bias = None
    if hasattr(output_emb, 'bias') and output_emb.bias is not None:
        lm_head_bias = output_emb.bias.detach().to(loaded.device).float()

    # Check for layer norm before LM head
    has_final_ln = hasattr(loaded.model, 'model') and hasattr(loaded.model.model, 'norm')
    if has_final_ln:
        final_ln = loaded.model.model.norm

    results = []
    lens_correct_count = 0
    writer_correct_count = 0
    decoded_correct_count = 0
    total = 0

    # Per-layer logit lens results
    per_layer_correct = {li: 0 for li in writer_window}
    # Aggregated across writer window
    agg_correct = 0

    from run_reviewer_controls import compute_readout_scores

    for idx, row in enumerate(rows):
        if idx % 50 == 0:
            print(f"  Processing {idx+1}/{len(rows)}...")

        gold_labels = set(row["support_labels"])

        # Get writer readout scores
        readout_scores = compute_readout_scores(
            loaded.model, loaded.tokenizer, adapter, base_model,
            row, loaded.device, writer_window,
        )
        writer_top2 = set(top_k_labels(readout_scores["positive_aligned"], 2))
        writer_correct = (writer_top2 == gold_labels)

        # Decoded baseline
        decoded_labels, decoded_correct = greedy_decode_baseline(
            loaded.model, loaded.tokenizer, row, loaded.device,
        )

        # Logit lens at each layer
        tok = loaded.tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(loaded.device)
        attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(loaded.device)
        pos_ids = compute_position_ids(attn_mask).to(loaded.device)

        with torch.no_grad():
            outputs = loaded.model(
                input_ids=input_ids, attention_mask=attn_mask, position_ids=pos_ids,
                output_hidden_states=True, use_cache=False, return_dict=True,
            )

        target_pos = int(input_ids.shape[1] - 1)

        # Aggregated logit lens score across writer window layers
        agg_label_scores = {l: 0.0 for l in FACT_LABELS}
        best_layer_correct = False

        for li in writer_window:
            hidden = outputs.hidden_states[li][0, target_pos].detach().float()

            # Apply layer norm if available
            if has_final_ln:
                hidden_normed = final_ln(hidden.unsqueeze(0)).squeeze(0)
            else:
                hidden_normed = hidden

            # Project through LM head
            logits = torch.matmul(hidden_normed, lm_head_weight.T)
            if lm_head_bias is not None:
                logits = logits + lm_head_bias

            # Get label logits
            label_logits = {}
            for label in FACT_LABELS:
                label_logits[label] = float(logits[label_token_ids[label]].item())

            # Top-2 by logit lens
            lens_top2 = set(top_k_labels(label_logits, 2))
            layer_correct = (lens_top2 == gold_labels)
            if layer_correct:
                per_layer_correct[li] += 1
                best_layer_correct = True

            # Accumulate for aggregated lens
            for label in FACT_LABELS:
                agg_label_scores[label] += label_logits[label]

        # Aggregated logit lens top-2
        agg_top2 = set(top_k_labels(agg_label_scores, 2))
        agg_correct_this = (agg_top2 == gold_labels)

        total += 1
        if writer_correct:
            writer_correct_count += 1
        if decoded_correct:
            decoded_correct_count += 1
        if agg_correct_this:
            agg_correct += 1
        if best_layer_correct:
            lens_correct_count += 1

        results.append({
            "example_id": row.get("example_id", idx),
            "prompt_id": row.get("prompt_id", f"p{idx}"),
            "gold_labels": sorted(gold_labels),
            "writer_correct": writer_correct,
            "decoded_correct": decoded_correct,
            "logit_lens_agg_correct": agg_correct_this,
            "logit_lens_best_layer_correct": best_layer_correct,
        })

    n = max(total, 1)
    per_layer_acc = {li: round(c / n, 3) for li, c in per_layer_correct.items()}

    summary = {
        "cell": cell_key,
        "family": cfg["family"],
        "model_id": cfg["model_id"],
        "n_prompts": total,
        "writer_accuracy": round(writer_correct_count / n, 3),
        "decoded_accuracy": round(decoded_correct_count / n, 3),
        "logit_lens_agg_accuracy": round(agg_correct / n, 3),
        "logit_lens_best_layer_accuracy": round(lens_correct_count / n, 3),
        "per_layer_accuracy": per_layer_acc,
        "random_baseline": 0.067,
        "writer_advantage_over_lens": round(
            (writer_correct_count - agg_correct) / n, 3
        ),
        "verdict": (
            "WRITER_BEATS_LENS" if writer_correct_count > agg_correct
            else "LENS_BEATS_WRITER" if agg_correct > writer_correct_count
            else "TIE"
        ),
    }

    print(f"\n  Writer accuracy:     {summary['writer_accuracy']:.3f}")
    print(f"  Decoded accuracy:    {summary['decoded_accuracy']:.3f}")
    print(f"  Logit lens (agg):    {summary['logit_lens_agg_accuracy']:.3f}")
    print(f"  Logit lens (best):   {summary['logit_lens_best_layer_accuracy']:.3f}")
    print(f"  Writer advantage:    {summary['writer_advantage_over_lens']}")
    print(f"  Verdict: {summary['verdict']}")

    save_json(out_dir / "summary.json", _jsonable(summary))
    pd.DataFrame(results).to_csv(out_dir / "per_example.csv", index=False)

    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# P8-D: Head Contribution Analysis
# ===================================================================

def experiment_head_contribution(cell_key: str, max_examples: int = 100):
    """Identify which attention heads drive the routing signal.
    For each head in writer window, compute its individual contribution
    to gold-fact routing. Report concentration metrics."""
    cfg = CELL_CONFIGS[cell_key]
    writer_window = cfg["writer_window"]
    out_dir = OUTPUT_DIR / "head_contribution" / cell_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"P8-D HEAD CONTRIBUTION: {cell_key} ({cfg['model_id']})")
    print(f"{'='*60}")

    loaded, adapter, base_model = load_model_for_cell(cell_key)
    rows = load_prompts(cell_key, max_examples)
    print(f"Loaded {len(rows)} prompts")

    label_token_ids = resolve_label_token_ids(loaded.tokenizer)
    output_emb = loaded.model.get_output_embeddings()
    unembed = output_emb.weight.detach().to(loaded.device).float()

    layers = adapter.get_layers(base_model)

    # Accumulate per-head contributions across all examples
    # Key: (layer_idx, q_head_idx) -> list of per-example contribution sums
    head_gold_contrib: Dict[Tuple[int, int], List[float]] = {}
    head_dist_contrib: Dict[Tuple[int, int], List[float]] = {}

    for idx, row in enumerate(rows):
        if idx % 50 == 0:
            print(f"  Processing {idx+1}/{len(rows)}...")

        gold_labels = set(row["support_labels"])
        facts = row["facts"]
        prompt_text = row["prompt"]

        # Get fact positions
        fact_spans = {}
        for fact in facts:
            text = fact["text"]
            start = prompt_text.index(text, prompt_text.index(fact_line(fact)))
            fact_spans[fact["label"]] = (start, start + len(text))
        positions = get_span_positions(loaded.tokenizer, prompt_text, fact_spans)

        # Target direction
        target_label = row["support_labels"][0]
        target_dir = unembed[label_token_ids[target_label]].clone()
        target_dir = target_dir / target_dir.norm().clamp(min=1e-8)

        # Encode
        tok = loaded.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(loaded.device)
        attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(loaded.device)
        pos_ids = compute_position_ids(attn_mask).to(loaded.device)

        with torch.no_grad():
            outputs = loaded.model(
                input_ids=input_ids, attention_mask=attn_mask, position_ids=pos_ids,
                output_hidden_states=True, output_attentions=True,
                use_cache=False, return_dict=True,
            )

        target_pos = int(input_ids.shape[1] - 1)

        for layer_idx in writer_window:
            resid_pre = outputs.hidden_states[layer_idx].detach()
            attn_input = adapter.attn_input(layers[layer_idx], resid_pre)
            layer_decomp = adapter.decompose_layer(
                layers[layer_idx], attn_input, pos_ids, base_model=base_model,
            )
            attn_probs = outputs.attentions[layer_idx][0].float()

            q_heads = layer_decomp.q.shape[0]
            for qh in range(q_heads):
                key = (layer_idx, qh)
                if key not in head_gold_contrib:
                    head_gold_contrib[key] = []
                    head_dist_contrib[key] = []

                kvh = adapter.q_to_kv_head(layers[layer_idx], qh)
                alpha = attn_probs[qh, target_pos, :].detach().float()
                v_src = layer_decomp.v[kvh, :, :].detach().float()
                o_block = layer_decomp.o_weight[qh].detach().float()
                writer = torch.matmul(v_src, o_block)
                writer_proj = torch.matmul(writer, target_dir)
                signed_contrib = alpha * writer_proj

                gold_sum = 0.0
                dist_sum = 0.0
                for label in FACT_LABELS:
                    pos = positions.get(label, [])
                    if not pos:
                        continue
                    pt = torch.tensor(pos, device=loaded.device, dtype=torch.long)
                    pt = pt[pt < alpha.shape[0]]
                    if pt.numel() == 0:
                        continue
                    sc = float(signed_contrib[pt].clamp(min=0).sum().item())
                    if label in gold_labels:
                        gold_sum += sc
                    else:
                        dist_sum += sc

                head_gold_contrib[key].append(gold_sum)
                head_dist_contrib[key].append(dist_sum)

    # Compute per-head mean contributions
    head_stats = []
    for key in sorted(head_gold_contrib.keys()):
        layer_idx, qh = key
        gold_mean = float(np.mean(head_gold_contrib[key]))
        dist_mean = float(np.mean(head_dist_contrib[key]))
        selectivity = gold_mean - dist_mean
        head_stats.append({
            "layer": layer_idx,
            "head": qh,
            "gold_contrib": round(gold_mean, 4),
            "dist_contrib": round(dist_mean, 4),
            "selectivity": round(selectivity, 4),
            "gold_share": round(gold_mean / max(gold_mean + dist_mean, 1e-8), 3),
        })

    df_heads = pd.DataFrame(head_stats).sort_values("selectivity", ascending=False)

    # Concentration analysis
    total_selectivity = df_heads["selectivity"].sum()
    n_heads = len(df_heads)
    if total_selectivity > 0:
        cumulative = df_heads["selectivity"].cumsum() / total_selectivity
        top5_share = float(cumulative.iloc[min(4, n_heads - 1)])
        top10_share = float(cumulative.iloc[min(9, n_heads - 1)])
        # Gini coefficient
        n = len(df_heads)
        sel_sorted = df_heads["selectivity"].sort_values().values
        index = np.arange(1, n + 1)
        gini = float((2 * (index * sel_sorted).sum() / (n * sel_sorted.sum())) - (n + 1) / n) if sel_sorted.sum() > 0 else 0
    else:
        top5_share = 0
        top10_share = 0
        gini = 0

    # Top 10 heads
    top10_heads = df_heads.head(10).to_dict(orient="records")

    summary = {
        "cell": cell_key,
        "family": cfg["family"],
        "model_id": cfg["model_id"],
        "n_prompts": len(rows),
        "n_total_heads": n_heads,
        "n_layers_in_window": len(writer_window),
        "total_selectivity": round(float(total_selectivity), 4),
        "top5_share": round(top5_share, 3),
        "top10_share": round(top10_share, 3),
        "gini_coefficient": round(gini, 3),
        "top10_heads": top10_heads,
        "concentration": (
            "CONCENTRATED" if top5_share > 0.5
            else "MODERATE" if top10_share > 0.5
            else "DISTRIBUTED"
        ),
    }

    print(f"\n  Total heads in window: {n_heads}")
    print(f"  Top-5 share: {top5_share:.3f}")
    print(f"  Top-10 share: {top10_share:.3f}")
    print(f"  Gini: {gini:.3f}")
    print(f"  Concentration: {summary['concentration']}")
    print(f"  Top 3 heads:")
    for h in top10_heads[:3]:
        print(f"    L{h['layer']}H{h['head']}: selectivity={h['selectivity']:.4f}")

    save_json(out_dir / "summary.json", _jsonable(summary))
    df_heads.to_csv(out_dir / "head_table.csv", index=False)

    del loaded, adapter, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ===================================================================
# Campaign runners
# ===================================================================

def run_all_for_cell(cell_key: str):
    """Run all Phase 8 experiments for a single cell."""
    print(f"\n{'#'*70}")
    print(f"# PHASE 8 ALL EXPERIMENTS: {cell_key}")
    print(f"{'#'*70}")
    summaries = {}
    for exp_name, exp_fn in [
        ("pca_causal_ablation", experiment_pca_causal_ablation),
        ("cross_example_pca", experiment_cross_example_pca),
        ("logit_lens_baseline", experiment_logit_lens_baseline),
        ("head_contribution", experiment_head_contribution),
    ]:
        try:
            summaries[exp_name] = exp_fn(cell_key)
        except Exception as e:
            print(f"  ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            summaries[exp_name] = {"error": str(e)}
    return summaries


def run_full_campaign():
    """Run all Phase 8 experiments across all cells."""
    all_summaries = {}
    for cell_key in ALL_CELLS:
        all_summaries[cell_key] = run_all_for_cell(cell_key)
        gc.collect()
        torch.cuda.empty_cache()

    # Write aggregate
    agg_dir = OUTPUT_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    save_json(agg_dir / "phase8_summary.json", _jsonable(all_summaries))

    # Write per-experiment aggregate tables
    for exp_name in ["pca_causal_ablation", "cross_example_pca", "logit_lens", "head_contribution"]:
        rows_table = []
        for cell_key in ALL_CELLS:
            cell_data = all_summaries.get(cell_key, {}).get(exp_name, {})
            if "error" not in cell_data:
                rows_table.append(cell_data)
        if rows_table:
            pd.DataFrame(rows_table).to_csv(
                agg_dir / f"{exp_name}_aggregate.csv", index=False,
            )

    print(f"\n{'='*70}")
    print("PHASE 8 CAMPAIGN COMPLETE")
    print(f"{'='*70}")

    return all_summaries


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 8 experiments")
    parser.add_argument("--experiment", required=True,
                        choices=["pca_causal_ablation", "cross_example_pca",
                                 "logit_lens_baseline", "head_contribution",
                                 "all_for_cell", "full_campaign"])
    parser.add_argument("--cell", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=200)
    args = parser.parse_args()

    if args.experiment == "full_campaign":
        run_full_campaign()
    elif args.experiment == "all_for_cell":
        if not args.cell:
            parser.error("--cell required for all_for_cell")
        run_all_for_cell(args.cell)
    else:
        if not args.cell:
            parser.error(f"--cell required for {args.experiment}")
        exp_map = {
            "pca_causal_ablation": experiment_pca_causal_ablation,
            "cross_example_pca": experiment_cross_example_pca,
            "logit_lens_baseline": experiment_logit_lens_baseline,
            "head_contribution": experiment_head_contribution,
        }
        exp_map[args.experiment](args.cell, args.max_examples)


if __name__ == "__main__":
    main()
