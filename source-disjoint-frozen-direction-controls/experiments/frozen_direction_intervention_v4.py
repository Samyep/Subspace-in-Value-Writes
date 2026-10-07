#!/usr/bin/env python3
"""Source-disjoint frozen-PC1 interventions with audited displacement controls.

This is an independent execution, not a continuation of historical phase14.
Fit: Hotpot source ranks 1--200; evaluation: ranks 201--400. No fit labels
choose a component or sign. Only baseline-correct evaluation prompts receive
interventions. All outcomes, including repeated labels, are retained. V4 keeps
every previously accepted edit unchanged. If the original numerical criterion
fails, it certifies the adjacent FP32 amplitudes on the execution device and
uses the nearest reachable BF16 edit along the same vector. Original-criterion
violations remain explicitly flagged; there is no tolerance increase.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP = ROOT / 'dependencies/controlled_task'
DEFAULT_RAG = ROOT.parent / 'value-write-rag-practicality'
DEFAULT_LEGACY = ROOT / 'dependencies/legacy'
CELLS = {
    'llama_3b': {'model_id': 'unsloth/Llama-3.2-3B-Instruct', 'window': list(range(14, 18)),
                 'revision': '006f5dcd1393c3add266de40994ba96225e9689d',
                 'canonical_model_id': 'meta-llama/Llama-3.2-3B-Instruct',
                 'canonical_revision': '0cb88a4f764b7a12671c53f0838cd831a0843b95'},
    'llama_8b': {'model_id': 'NousResearch/Hermes-3-Llama-3.1-8B', 'window': list(range(14, 25)),
                 'revision': '896ea440e5a9e6070e3d8a2774daf2b481ab425b'},
    'qwen_7b': {'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'window': list(range(24, 28)),
                'revision': 'a09a35458c702b33eeacc393d103063234e8bc28'},
    'qwen_14b': {'model_id': 'Qwen/Qwen2.5-14B-Instruct', 'window': list(range(32, 48)),
                 'revision': 'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'},
    'mistral_7b': {'model_id': 'mistralai/Mistral-7B-Instruct-v0.3', 'window': list(range(19, 23)),
                   'revision': 'c170c708c41dac9275d15a8fff4eca08d52bab71'},
    'mistral_12b': {'model_id': 'mistralai/Mistral-Nemo-Instruct-2407', 'window': list(range(19, 25)),
                    'revision': '04d8a90549d23fc6bd7f642064003592df51e9b3'},
}
LABELS = list('ABCDEF')
REPLICATES = 3
BASE_SEED = 2991137
MATCH_RTOL = 0.005
MATCH_OUTPUT_NORM_FLOOR = 1e-6



def stable_id(value: Any) -> str:
    return hashlib.blake2b(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode(), digest_size=16).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def source_split(rows: list[dict], smoke: int = 0) -> tuple[list[dict], list[dict]]:
    selected = [r for r in rows if r['family_name'] in {'active_set', 'working_set'}
                and r['lexical_pack'] in {'pack_a', 'pack_b'}]
    assert len(selected) == 1600, f'Expected 1600 prompts, got {len(selected)}'
    assert len({r['prompt_id'] for r in selected}) == 1600
    expected_variants = {(f, p) for f in ('active_set', 'working_set') for p in ('pack_a', 'pack_b')}
    grouped = defaultdict(list)
    for r in selected:
        grouped[r['example_id']].append(r)
        assert [f['label'] for f in r['facts']] == LABELS
        assert len(r['support_labels']) == len(set(r['support_labels'])) == 2
    assert len(grouped) == 400
    assert {int(r['example_rank']) for r in selected} == set(range(1, 401))
    for variants in grouped.values():
        assert len(variants) == 4
        assert len({v['example_rank'] for v in variants}) == 1
        assert {(v['family_name'], v['lexical_pack']) for v in variants} == expected_variants
    fit = [r for r in selected if int(r['example_rank']) <= 200]
    evaluate = [r for r in selected if int(r['example_rank']) > 200]
    assert len(fit) == len(evaluate) == 800
    assert not ({r['example_id'] for r in fit} & {r['example_id'] for r in evaluate})
    if smoke:
        assert 1 <= smoke <= 10, 'Smoke allows 1--10 sources on each side'
        fit = [r for r in fit if int(r['example_rank']) <= smoke]
        evaluate = [r for r in evaluate if int(r['example_rank']) <= 200 + smoke]
    return fit, evaluate


def category(prediction: list[str], support: list[str]) -> str:
    assert len(prediction) == 2 and all(p in LABELS for p in prediction)
    if prediction[0] == prediction[1]:
        return 'repeated_label'
    return 'correct' if set(prediction) == set(support) else 'wrong_distinct_pair'


@lru_cache(maxsize=1)
def _released_seed_schedule() -> dict[tuple[str, str, int], int]:
    """Load the fixed seeds used by the released evaluation when available."""
    root = Path(__file__).resolve().parents[1] / 'results/frozen_direction/main_v4'
    schedule = {}
    for path in sorted(root.glob('*/per_prompt.jsonl')):
        cell = path.parent.name
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for control in row.get('random', []):
                key = (cell, row['prompt_id'], int(control['replicate']))
                seed = int(control['seed'])
                if key in schedule and schedule[key] != seed:
                    raise RuntimeError(f'Conflicting released seed for {key}')
                schedule[key] = seed
    return schedule


def random_seed(cell: str, prompt_id: str, replicate: int) -> int:
    released = _released_seed_schedule().get((cell, prompt_id, replicate))
    if released is not None:
        return released
    material = f'd36-path6-frozen-pc1|{BASE_SEED}|{cell}|{prompt_id}|{replicate}'
    return int.from_bytes(hashlib.blake2b(material.encode(), digest_size=8).digest(), 'big')


def random_unit(hidden_size: int, seed: int, device: torch.device) -> torch.Tensor:
    vector = np.random.default_rng(seed).standard_normal(hidden_size)
    vector /= np.linalg.norm(vector)
    result = torch.tensor(vector, dtype=torch.float32, device=device)
    return result / result.norm()


def project_remove(before: torch.Tensor, unit: torch.Tensor) -> tuple[torch.Tensor, dict]:
    original = before.float()
    delta = -torch.dot(original, unit) * unit
    after = (original + delta).to(before.dtype)
    actual = float(torch.linalg.vector_norm(after.float() - original))
    return after, {'requested_norm': float(delta.norm()), 'achieved_norm': actual,
                   'original_output_norm': float(original.norm())}


def _legacy_equal_norm_displacement(before: torch.Tensor, unit: torch.Tensor, target: float) -> tuple[torch.Tensor, dict]:
    """Choose a scalar before BF16 cast, matching the resulting displacement.

    The Gaussian unit vector is fixed; only its scalar is adjusted to compensate
    for rounding. Target is the *actual* PC1 displacement at this layer/step.
    This does not preserve the attention output's norm and is not a projection.
    The declared absolute resolution floor is not an exact-match guarantee or
    a mathematical bound on BF16 rounding. It handles staircase-unattainable
    relative matches only after completing the fixed scalar search.
    """
    original = before.float()
    original_norm = float(original.norm())
    absolute_floor = MATCH_OUTPUT_NORM_FLOOR * original_norm
    if target == 0.0:
        return before.clone(), {'requested_norm': 0.0, 'achieved_norm': 0.0,
                                'relative_norm_error': 0.0, 'absolute_norm_error': 0.0,
                                'output_relative_norm_error': 0.0,
                                'absolute_tolerance_floor': absolute_floor,
                                'used_absolute_tolerance': False,
                                'original_output_norm': original_norm,
                                'scale': 0.0, 'rounding_steps': 0}
    unit = unit / unit.norm()
    def candidate(scale: float):
        value = (original + scale * unit).to(before.dtype)
        achieved = float(torch.linalg.vector_norm(value.float() - original))
        return value, achieved
    low, high = 0.0, target
    after, achieved = candidate(high)
    steps = 1
    best = (abs(achieved - target), after, achieved, high)
    while achieved < target and steps < 16:
        low, high = high, 2 * high
        after, achieved = candidate(high)
        steps += 1
        if abs(achieved - target) < best[0]:
            best = (abs(achieved - target), after, achieved, high)
    for _ in range(24):
        if best[0] <= MATCH_RTOL * target:
            break
        middle = (low + high) / 2
        after, achieved = candidate(middle)
        steps += 1
        if abs(achieved - target) < best[0]:
            best = (abs(achieved - target), after, achieved, middle)
        if achieved < target:
            low = middle
        else:
            high = middle
    error, after, achieved, scale = best
    tolerance = max(MATCH_RTOL * target, absolute_floor)
    if error > tolerance:
        raise RuntimeError(f'BF16 displacement norm cannot meet declared tolerance: target={target}, achieved={achieved}, '
                           f'error={error}, tolerance={tolerance}, original_norm={original_norm}')
    return after, {'requested_norm': target, 'achieved_norm': achieved,
                   'relative_norm_error': error / target, 'absolute_norm_error': error,
                   'output_relative_norm_error': error / max(original_norm, 1e-30),
                   'absolute_tolerance_floor': absolute_floor,
                   'used_absolute_tolerance': error > MATCH_RTOL * target,
                   'scale': scale, 'rounding_steps': steps, 'original_output_norm': original_norm}


def _certify_nearest_quantized_edit(before: torch.Tensor, normalized_unit: torch.Tensor,
                                    target: float, original_norm: float) -> tuple[torch.Tensor, dict]:
    """Find the nearest norm on the fixed-vector quantized scalar staircase.

    Every candidate is evaluated on before.device using the original FP32
    multiply/add, BF16 assignment, and FP32 norm. Binary search is over ordered
    positive FP32 bit patterns and terminates only at adjacent amplitudes that
    straddle the target. The already-normalized vector is never changed.
    """
    if target <= 0 or not np.isfinite(target):
        raise ValueError('Certification requires a positive finite target')
    if not torch.isfinite(before).all() or not torch.isfinite(normalized_unit).all():
        raise RuntimeError('Non-finite activation or direction in norm certification')
    if not float(normalized_unit.float().norm()) > 0:
        raise RuntimeError('Zero direction in norm certification')
    original = before.float()
    tolerance = max(MATCH_RTOL * target, MATCH_OUTPUT_NORM_FLOOR * original_norm)
    evaluations = 0
    def to_bits(value):
        return int(np.asarray(np.float32(value)).view(np.uint32))
    def from_bits(bits):
        return float(np.asarray(np.uint32(bits)).view(np.float32))
    def candidate(scale):
        nonlocal evaluations
        scalar = float(np.float32(scale))
        after = (original + scalar * normalized_unit).to(before.dtype)
        achieved = float(torch.linalg.vector_norm(after.float() - original))
        evaluations += 1
        return after, achieved, scalar

    lower_bits = 0
    upper_scalar = max(float(np.float32(target)), float(np.finfo(np.float32).tiny))
    upper_after, upper_norm, upper_scalar = candidate(upper_scalar)
    expansions = 0
    while upper_norm < target:
        lower_bits = to_bits(upper_scalar)
        upper_scalar = float(np.float32(upper_scalar * 2))
        if not np.isfinite(upper_scalar):
            raise RuntimeError('No finite scalar bracket for norm certification')
        upper_after, upper_norm, upper_scalar = candidate(upper_scalar)
        expansions += 1
        if expansions > 150:
            raise RuntimeError('Norm certification bracket expansion did not converge')
    upper_bits = to_bits(upper_scalar)
    iterations = 0
    while upper_bits - lower_bits > 1:
        middle_bits = (lower_bits + upper_bits) // 2
        _, achieved, _ = candidate(from_bits(middle_bits))
        if achieved < target:
            lower_bits = middle_bits
        else:
            upper_bits = middle_bits
        iterations += 1
        if iterations > 32:
            raise RuntimeError('FP32 bit-bracket search exceeded 32 iterations')
    lower_after, lower_norm, lower_scalar = candidate(from_bits(lower_bits))
    upper_after, upper_norm, upper_scalar = candidate(from_bits(upper_bits))
    if upper_bits - lower_bits != 1 or not lower_norm < target <= upper_norm:
        raise RuntimeError('Norm certification did not establish adjacent amplitudes straddling the target')
    # Stable lower endpoint for an exact tie; no RNG, vector resampling or
    # coordinate correction enters the fallback.
    select_lower = abs(lower_norm - target) <= abs(upper_norm - target)
    after, achieved, scalar = ((lower_after, lower_norm, lower_scalar) if select_lower
                               else (upper_after, upper_norm, upper_scalar))
    error = abs(achieved - target)
    original_pass = error <= tolerance
    def endpoint(after_value, norm, scale, bits):
        return {'scale': scale, 'scalar_float32_bits': bits, 'achieved_norm': norm,
                'achieved_norm_fp64': float(torch.linalg.vector_norm((after_value.float() - original).double())),
                'absolute_norm_error': abs(norm - target),
                'original_tolerance_passed': abs(norm - target) <= tolerance}
    bracket = {'lower': endpoint(lower_after, lower_norm, lower_scalar, lower_bits),
               'upper': endpoint(upper_after, upper_norm, upper_scalar, upper_bits),
               'selected_endpoint': 'lower' if select_lower else 'upper',
               'tie_rule': 'lower', 'norm_gap': upper_norm - lower_norm,
               'adjacent_scalar_bits_difference': upper_bits - lower_bits,
               'straddles_target': True, 'device': str(before.device),
               'candidate_dtype': str(before.dtype), 'iterations': iterations,
               'expansions': expansions, 'candidate_evaluations': evaluations,
               'original_combined_tolerance': tolerance,
               'original_tolerance_attainable': original_pass}
    return after, {'requested_norm': target, 'achieved_norm': achieved,
                   'relative_norm_error': error / target, 'absolute_norm_error': error,
                   'output_relative_norm_error': error / max(original_norm, 1e-30),
                   'absolute_tolerance_floor': MATCH_OUTPUT_NORM_FLOOR * original_norm,
                   'used_absolute_tolerance': error > MATCH_RTOL * target and original_pass,
                   'scale': scalar, 'original_output_norm': original_norm,
                   'original_tolerance_passed': original_pass,
                   'certified_fallback_used': True, 'quantization_limited': not original_pass,
                   'certified_bracket': bracket, 'certified_search_steps': evaluations}


def equal_norm_displacement(before: torch.Tensor, unit: torch.Tensor, target: float) -> tuple[torch.Tensor, dict]:
    """Keep original successes; transparently flag certified nearest fallbacks."""
    try:
        after, record = _legacy_equal_norm_displacement(before, unit, target)
    except RuntimeError as error:
        if not str(error).startswith('BF16 displacement norm cannot meet declared tolerance:'):
            raise
        frame = error.__traceback__
        while frame is not None and frame.tb_frame.f_code.co_name != '_legacy_equal_norm_displacement':
            frame = frame.tb_next
        if frame is None:
            raise RuntimeError('Original norm failure has no matcher frame') from error
        saved = frame.tb_frame.f_locals
        # The exact vector used by the failed original call is reused, including
        # its original FP32 normalization rounding on this device.
        after, record = _certify_nearest_quantized_edit(before, saved['unit'], target,
                                                       saved['original_norm'])
        record['rounding_steps'] = saved['steps']
        record['legacy_failure'] = str(error)
        del frame, saved
        return after, record
    record.update(original_tolerance_passed=True, certified_fallback_used=False,
                  quantization_limited=False)
    return after, record


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode(loaded, row: dict, label_ids: dict, writer_window: list[int],
           direction: torch.Tensor | None = None, norm_schedule: dict | None = None) -> dict:
    """Same two constrained A--F decisions and comma prefix as legacy decoder."""
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    records, prediction, margins = [], [], []
    current_step = 0
    handles = []
    def make_hook(layer_index):
        def hook(_module, _args, output):
            first = output[0] if isinstance(output, tuple) else output
            edited = first.clone()
            before = first[0, -1]
            if norm_schedule is None:
                after, record = project_remove(before, direction)
            else:
                after, record = equal_norm_displacement(before, direction, norm_schedule[(current_step, layer_index)])
            edited[0, -1] = after
            record.update(step=current_step, layer=layer_index)
            records.append(record)
            return (edited,) + output[1:] if isinstance(output, tuple) else edited
        return hook
    if direction is not None:
        for layer_index in writer_window:
            handles.append(model.model.layers[layer_index].self_attn.register_forward_hook(make_hook(layer_index)))
    synchronize(device)
    started = time.perf_counter()
    try:
        for current_step in range(2):
            prefix = '' if not prediction else prediction[0] + ','
            encoded = tokenizer(row['prompt'] + prefix, return_tensors='pt', add_special_tokens=False)
            input_ids = encoded['input_ids'].to(device)
            mask = encoded.get('attention_mask', torch.ones_like(input_ids)).to(device)
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 0)
            output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True)
            logits = output.logits[0, -1].float()
            scores = {label: float(logits[token_id]) for label, token_id in label_ids.items()}
            winner = max(scores, key=scores.get)
            sorted_scores = sorted(scores.values(), reverse=True)
            prediction.append(winner)
            margins.append(sorted_scores[0] - sorted_scores[1])
            del output
    finally:
        for handle in handles:
            handle.remove()
    synchronize(device)
    if direction is not None:
        assert len(records) == 2 * len(writer_window)
        assert len({(r['step'], r['layer']) for r in records}) == len(records)
    return {'prediction': prediction, 'category': category(prediction, row['support_labels']),
            'gold_hit_count': len(set(prediction) & set(row['support_labels'])),
            'label_logit_margins': margins, 'seconds': time.perf_counter() - started,
            'perturbations': records}


def fit_pc1(vectors: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit only the leading covariance eigenvector; labels never enter here."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    centered = vectors.astype(np.float64)
    centered -= centered.mean(axis=1, keepdims=True)
    matrix = centered.reshape(-1, centered.shape[-1])
    # Numerically subtract the remaining pooled mean (theoretical value zero).
    matrix -= matrix.mean(axis=0)
    dimension = matrix.shape[1]
    operator = LinearOperator((dimension, dimension), matvec=lambda v: matrix.T @ (matrix @ v), dtype=np.float64)
    initial = np.random.default_rng(BASE_SEED).standard_normal(dimension)
    values, components = eigsh(operator, k=1, which='LA', v0=initial, tol=1e-9, maxiter=3000)
    unit = components[:, 0]
    # Deterministic storage orientation, independent of labels. Removal is sign invariant.
    if unit[np.argmax(np.abs(unit))] < 0:
        unit *= -1
    value = float(values[0])
    residual = float(np.linalg.norm(operator @ unit - value * unit) / max(value, 1e-30))
    if residual > 1e-6:
        raise RuntimeError(f'Unconverged PC1 eigensolver: {residual}')
    component = unit.astype(np.float32)
    component /= np.linalg.norm(component)
    return component, {'n_prompts': len(vectors), 'n_facts': len(matrix), 'hidden_size': dimension,
                       'component_index': 1, 'gold_labels_used': False,
                       'eigenvalue_unnormalized': value, 'explained_variance_ratio': value / float(np.square(matrix).sum()),
                       'eigen_residual_relative': residual}


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        backup = path.with_name(path.name + '.interrupted-tail-' + str(time.time_ns()))
        backup.write_bytes(data)
        data = data[:data.rfind(b'\n') + 1]
        path.write_bytes(data)
    return [json.loads(line) for line in data.splitlines()]


def append_result(path: Path, result: dict) -> None:
    with path.open('a') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def load_mistral_tokenizer(snapshot: Path, cell: str):
    """Use the public tokenizer API, respecting each checkpoint's tokenizer.

    Mistral-7B's canonical tokenizer uses Metaspace; the large-vocabulary NeMo
    tokenizer uses Split + ByteLevel with its supported public regex option.
    Validate the serialized raw backend, never a wrapper's cosmetic flag.
    """
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    assert cell in {'mistral_7b', 'mistral_12b'}
    apply_regex_fix = cell == 'mistral_12b'
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True,
        trust_remote_code=False, fix_mistral_regex=apply_regex_fix)
    backend = tokenizer.backend_tokenizer
    serialized = json.loads(backend.to_str())
    canonical = json.loads((snapshot / 'tokenizer.json').read_text())
    observed_pre = serialized['pre_tokenizer']
    canonical_pre = canonical['pre_tokenizer']
    if cell == 'mistral_7b':
        if observed_pre != canonical_pre or observed_pre.get('type') != 'Metaspace':
            raise RuntimeError('Mistral-7B canonical Metaspace pre-tokenizer was changed')
    else:
        parts = observed_pre.get('pretokenizers', [])
        if not (observed_pre.get('type') == 'Sequence' and len(parts) == 2
                and parts[0].get('type') == 'Split' and parts[1].get('type') == 'ByteLevel'
                and getattr(backend, 'fix_mistral_regex', False)):
            raise RuntimeError('NeMo raw tokenizer backend did not receive the public regex correction')
        expected_regex = (r'[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|'
                          r'[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|'
                          r'\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+')
        if parts[0].get('pattern', {}).get('Regex') != expected_regex:
            raise RuntimeError('NeMo backend regex differs from the current public correction')
    probes = ['A. [Title] First fact.', 'Two  spaces and\ta tab.\nNext line.',
              'Select two facts. Unicode: café, 東京. Digits: 12345.']
    canonical_backend = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    probe_audits = []
    for text in probes:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise RuntimeError(f'{cell} tokenizer changed round-trip text: {text!r} -> {decoded!r}')
        canonical_ids = canonical_backend.encode(text, add_special_tokens=False).ids
        same_ids = encoded == canonical_ids
        if cell == 'mistral_7b' and not same_ids:
            raise RuntimeError('Mistral-7B tokenization differs from its canonical tokenizer asset')
        probe_audits.append({'text': text, 'roundtrip_exact': True,
                             'canonical_token_ids_match': same_ids, 'token_count': len(encoded)})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, {'cell': cell, 'loader': 'AutoTokenizer.from_pretrained public API',
                       'fix_mistral_regex_argument': apply_regex_fix,
                       'raw_backend_fix_flag': getattr(backend, 'fix_mistral_regex', None),
                       'canonical_pre_tokenizer': canonical_pre, 'observed_pre_tokenizer': observed_pre,
                       'canonical_pre_tokenizer_preserved': observed_pre == canonical_pre,
                       'probe_audits': probe_audits}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', choices=tuple(CELLS), required=True)
    p.add_argument('--output', type=Path, required=True, help='Per-cell output; use a separate directory for smoke')
    p.add_argument('--experiment-root', type=Path, default=DEFAULT_EXP)
    p.add_argument('--rag-root', type=Path, default=DEFAULT_RAG)
    p.add_argument('--legacy-root', type=Path, default=DEFAULT_LEGACY)
    p.add_argument('--data', type=Path)
    p.add_argument('--protocol-file', type=Path, default=Path(__file__).with_name('PROTOCOL_v4.md'),
                   help='Parent protocol file included in the immutable run manifest')
    p.add_argument('--revision', help='Must equal the predeclared per-model commit; defaults to that pinned revision')
    p.add_argument('--smoke', type=int, default=0, help='1--10 sources on each side; zero is complete 200+200-source protocol')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-prompt-tokens', type=int, choices=[768], default=768,
                   help='Inherited 768-token input cutoff; exclusions are enumerated, never truncated')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    torch.manual_seed(BASE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    for root in (args.rag_root, args.experiment_root, args.legacy_root):
        sys.path.insert(0, str(root.resolve()))
    from huggingface_hub import snapshot_download
    import transformers
    from run_reviewer_controls import (greedy_decode_baseline, resolve_label_token_ids,
                                      get_span_positions, fact_line, _rerender_prompt_safe)
    from vw_rag.model_loading import load_model_and_tokenizer
    from vw_rag.value_write import DocumentFeatureExtractor

    data_path = args.data or ROOT / 'data/hotpot_400_prompts.jsonl'
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    fit_rows, eval_rows = source_split(rows, args.smoke)
    cfg = CELLS[args.cell]
    if args.revision and args.revision != cfg['revision']:
        raise RuntimeError('This fixed protocol does not permit substituting a model revision')
    snapshot = Path(snapshot_download(cfg['model_id'], revision=cfg['revision'], local_files_only=True))
    if args.cell == 'llama_3b':
        verification = Path(__file__).with_name('llama3b_checkpoint_verification.json')
        if not verification.exists():
            raise RuntimeError('Llama checkpoint verification artifact is required before loading the mirrored weights')
    identity = {
        'version': 'path6-six-model-source-disjoint-pc1-displacement-v4', 'cell': args.cell,
        'model_id': cfg['model_id'], 'resolved_snapshot': snapshot.name,
        'canonical_model_id': cfg.get('canonical_model_id', cfg['model_id']),
        'canonical_revision': cfg.get('canonical_revision', cfg['revision']),
        'model_snapshot_files': {str(p.relative_to(snapshot)): {'bytes': p.stat().st_size}
                                 for p in sorted(snapshot.rglob('*')) if p.is_file()},
        'writer_window': cfg['window'], 'dtype': 'bfloat16', 'attention_backend': 'eager',
        'model_loader': 'vw_rag.model_loading',
        'tokenizer_loader': ('AutoTokenizer public API, canonical Metaspace, fix_mistral_regex=False' if args.cell == 'mistral_7b'
                             else 'AutoTokenizer public API, verified raw Split+ByteLevel, fix_mistral_regex=True' if args.cell == 'mistral_12b'
                             else 'vw_rag.model_loading'),
        'prompt_renderer': 'run_reviewer_controls._rerender_prompt_safe (original system-role fallback)',
        'fit_source_ranks': [1, 200], 'eval_source_ranks': [201, 400],
        'fit_example_ids': sorted({r['example_id'] for r in fit_rows}),
        'eval_example_ids': sorted({r['example_id'] for r in eval_rows}),
        'fit_prompt_ids': [r['prompt_id'] for r in fit_rows], 'eval_prompt_ids': [r['prompt_id'] for r in eval_rows],
        'smoke_sources_per_side': args.smoke, 'random_replicates': REPLICATES, 'seed': BASE_SEED,
        'random_seed_definition': 'released fixed seed schedule, with a deterministic fallback for prompts outside the release',
        'random_direction': 'isotropic Gaussian unit vector fixed across layers and both steps, independently per prompt and replicate',
        'norm_schedule': 'actual BF16 displacement of PC1 rollout at matching layer and decoding step; random rollout can follow a different first label',
        'norm_relative_tolerance': MATCH_RTOL,
        'norm_absolute_output_fraction': MATCH_OUTPUT_NORM_FLOOR,
        'norm_tolerance_rule': 'max(0.005 * target_edit_norm, 1e-6 * current_attention_output_norm); absolute floor only after fixed scalar search',
        'norm_quantization_policy': 'Keep every original passing edit unchanged; on original failure certify adjacent FP32 scalar amplitudes on the same device and same normalized vector, choose the nearest actual norm with lower tie; preserve flags for original-bound violations',
        'norm_controls_are_approximate': True,
        'max_prompt_tokens': args.max_prompt_tokens,
        'fit_centering': 'within prompt across six text-only fact vectors, then numerical pooled recentering',
        'fit_component': 'first covariance eigenvector by variance alone; no gold component/sign selection',
        'intervention_site': 'full self-attention module output, final current decoding position, every frozen writer layer, both restricted steps',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__},
    }
    run_id = stable_id(identity)
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive lock also prevents accidental simultaneous appends during resume.
    import fcntl
    lock = (args.output / '.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['run_id'] != run_id:
            raise RuntimeError('Resume refused: code, protocol, input, snapshot, or environment changed')
    else:
        atomic_json(manifest_path, {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(), 'identity': identity})
    print(json.dumps({'phase': 'load_model', 'cell': args.cell, 'snapshot': snapshot.name, 'run_id': run_id}), flush=True)
    verified_tokenizer = None
    tokenizer_audit = None
    if args.cell.startswith('mistral_'):
        verified_tokenizer, tokenizer_audit = load_mistral_tokenizer(snapshot, args.cell)
        atomic_json(args.output / 'tokenizer_audit.json', tokenizer_audit)
    loaded = load_model_and_tokenizer(cfg['model_id'], revision=snapshot.name, device=args.device,
                                      dtype='bf16', attn_implementation='eager', trust_remote_code=False)
    if verified_tokenizer is not None:
        if len(verified_tokenizer) != len(loaded.tokenizer):
            raise RuntimeError('Public tokenizer reload changed vocabulary size')
        loaded.tokenizer = verified_tokenizer
    if loaded.model.config._attn_implementation != 'eager':
        raise RuntimeError('Eager attention was not applied by the loader')
    if loaded.dtype != torch.bfloat16:
        raise RuntimeError('This protocol requires BF16 execution on a GPU')
    label_ids = resolve_label_token_ids(loaded.tokenizer)
    rendered_meta = []
    retained_ids = set()
    for row in fit_rows + eval_rows:
        # Render without length filtering first, so excluded lengths are known.
        rendered = _rerender_prompt_safe(row, loaded.tokenizer, 10**9)
        assert rendered is not None
        tokens = len(loaded.tokenizer.encode(rendered, add_special_tokens=False))
        retained = tokens <= args.max_prompt_tokens
        rendered_meta.append({'prompt_id': row['prompt_id'], 'example_id': row['example_id'],
                              'example_rank': row['example_rank'], 'retained': retained,
                              'tokens': tokens})
        if not retained:
            continue
        row['prompt'] = rendered
        retained_ids.add(row['prompt_id'])
    fit_rows = [r for r in fit_rows if r['prompt_id'] in retained_ids]
    eval_rows = [r for r in eval_rows if r['prompt_id'] in retained_ids]
    assert fit_rows and eval_rows
    assert not ({r['example_id'] for r in fit_rows} & {r['example_id'] for r in eval_rows})
    atomic_json(args.output / 'rendered_prompts.json', rendered_meta)
    retention = {}
    for name, original_ids, kept in [('fit', identity['fit_prompt_ids'], fit_rows), ('eval', identity['eval_prompt_ids'], eval_rows)]:
        original_set = set(original_ids)
        original_rows = [r for r in rendered_meta if r['prompt_id'] in original_set]
        retained_by_source = dict(Counter(r['example_id'] for r in kept))
        retention[name] = {'before_prompts': len(original_ids), 'after_prompts': len(kept),
                           'before_sources': len({r['example_id'] for r in original_rows}),
                           'after_sources': len(retained_by_source), 'retained_prompts_by_source': retained_by_source,
                           'excluded': [r for r in original_rows if not r['retained']]}
    atomic_json(args.output / 'prompt_retention.json', retention)
    atomic_json(args.output / 'execution_environment.json', {'device': str(loaded.device), 'gpu': torch.cuda.get_device_name(loaded.device),
                'config_commit_hash': getattr(loaded.config, '_commit_hash', None), 'snapshot_commit': snapshot.name,
                'mistral_regex_fix': tokenizer_audit['raw_backend_fix_flag'] if tokenizer_audit else None,
                'tokenizer_class': type(loaded.tokenizer).__name__,
                'max_memory_allocated': torch.cuda.max_memory_allocated(loaded.device), 'label_token_ids': label_ids})

    fit_dir = args.output / 'fit_vectors'
    fit_dir.mkdir(exist_ok=True)
    extractor = DocumentFeatureExtractor(loaded.model, cfg['window'])
    try:
        for index, row in enumerate(fit_rows):
            path = fit_dir / (hashlib.blake2b(row['prompt_id'].encode(), digest_size=16).hexdigest() + '.npz')
            if path.exists():
                with np.load(path) as cached:
                    assert str(cached['run_id']) == run_id
                    assert str(cached['prompt_id']) == row['prompt_id']
                continue
            spans = {}
            for fact in row['facts']:
                start = row['prompt'].index(fact['text'], row['prompt'].index(fact_line(fact)))
                spans[fact['label']] = (start, start + len(fact['text']))
            positions = get_span_positions(loaded.tokenizer, row['prompt'], spans)
            assert all(positions[label] for label in LABELS)
            tok = loaded.tokenizer(row['prompt'], return_tensors='pt', add_special_tokens=False)
            ids = tok['input_ids'].to(loaded.device)
            mask = tok.get('attention_mask', torch.ones_like(ids)).to(loaded.device)
            synchronize(loaded.device)
            started = time.perf_counter()
            vectors, attention_mass, lengths = extractor.collect(ids, mask, [positions[label] for label in LABELS])
            synchronize(loaded.device)
            seconds = time.perf_counter() - started
            assert vectors.shape == (6, loaded.config.hidden_size) and np.isfinite(vectors).all()
            if index == 0:
                audit = {'prompt_id': row['prompt_id'], 'status': 'unavailable'}
                try:
                    from src.hook_registry import resolve_adapter
                    from run_phase8_experiments import collect_fact_write_vectors
                    adapter = resolve_adapter(loaded.model)
                    legacy = collect_fact_write_vectors(loaded.model, loaded.tokenizer, adapter,
                                adapter.get_base_model(loaded.model), row, loaded.device, cfg['window'])
                    reference = np.stack([legacy[label] for label in LABELS])
                    relative = float(np.linalg.norm(vectors - reference) / max(np.linalg.norm(reference), 1e-30))
                    audit.update(status='passed' if relative < 1e-4 else 'failed', relative_l2=relative,
                                 max_abs=float(np.abs(vectors - reference).max()), tolerance_relative_l2=1e-4)
                except (ImportError, AttributeError, TypeError) as error:
                    audit.update(reason=repr(error))
                atomic_json(args.output / 'legacy_extractor_audit.json', audit)
                if audit['status'] == 'failed':
                    raise RuntimeError('Legacy extractor comparison failed: ' + str(audit))
            atomic_npz(path, run_id=run_id, prompt_id=row['prompt_id'], example_id=row['example_id'],
                       vectors=vectors, attention_mass=attention_mass, token_lengths=lengths, seconds=seconds)
            if (index + 1) % 10 == 0 or index + 1 == len(fit_rows):
                status = {'phase': 'extract_fit', 'done': index + 1, 'total': len(fit_rows), 'last_seconds': seconds}
                print(json.dumps(status), flush=True)
                atomic_json(args.output / 'progress.json', status)
    finally:
        extractor.close()
    direction_path = args.output / 'frozen_pc1.npz'
    if direction_path.exists():
        with np.load(direction_path) as data:
            assert str(data['run_id']) == run_id
            component = data['component'].copy()
        stats = json.loads((args.output / 'fit_statistics.json').read_text())
    else:
        fit_vectors = []
        for row in fit_rows:
            path = fit_dir / (hashlib.blake2b(row['prompt_id'].encode(), digest_size=16).hexdigest() + '.npz')
            with np.load(path) as data:
                fit_vectors.append(data['vectors'])
        component, stats = fit_pc1(np.stack(fit_vectors))
        del fit_vectors
        atomic_json(args.output / 'fit_statistics.json', stats)
        atomic_npz(direction_path, component=component, run_id=run_id)
    unit = torch.tensor(component, dtype=torch.float32, device=loaded.device)
    unit /= unit.norm()
    print(json.dumps({'phase': 'fitted', **stats}), flush=True)

    results_path = args.output / 'per_prompt.jsonl'
    results = read_results(results_path)
    done = {r['prompt_id'] for r in results}
    assert len(done) == len(results)
    assert done <= {r['prompt_id'] for r in eval_rows}
    assert all(r['run_id'] == run_id for r in results)
    for index, row in enumerate(eval_rows):
        if row['prompt_id'] in done:
            continue
        baseline = decode(loaded, row, label_ids, cfg['window'])
        if index == 0:
            legacy_pred, legacy_correct = greedy_decode_baseline(loaded.model, loaded.tokenizer, row, loaded.device)
            audit = {'prompt_id': row['prompt_id'], 'legacy_prediction': legacy_pred,
                     'new_prediction': baseline['prediction'], 'agreement': legacy_pred == baseline['prediction']}
            atomic_json(args.output / 'legacy_decoder_audit.json', audit)
            assert audit['agreement'] and legacy_correct == (baseline['category'] == 'correct')
        result = {k: row[k] for k in ('prompt_id', 'example_id', 'example_rank', 'family_name', 'lexical_pack', 'support_labels')}
        result.update(run_id=run_id, baseline=baseline, pc1=None, random=[], smoke=bool(args.smoke))
        if baseline['category'] == 'correct':
            pc1 = decode(loaded, row, label_ids, cfg['window'], direction=unit)
            result['pc1'] = pc1
            schedule = {(r['step'], r['layer']): r['achieved_norm'] for r in pc1['perturbations']}
            for replicate in range(REPLICATES):
                seed = random_seed(args.cell, row['prompt_id'], replicate)
                random_direction = random_unit(len(component), seed, loaded.device)
                control = decode(loaded, row, label_ids, cfg['window'], direction=random_direction, norm_schedule=schedule)
                requested_energy = sum(p['requested_norm'] ** 2 for p in control['perturbations'])
                achieved_energy = sum(p['achieved_norm'] ** 2 for p in control['perturbations'])
                control.update(replicate=replicate, seed=seed,
                               original_tolerance_passed=all(p['original_tolerance_passed'] for p in control['perturbations']),
                               certified_fallback_sites=sum(p['certified_fallback_used'] for p in control['perturbations']),
                               quantization_limited_sites=sum(p['quantization_limited'] for p in control['perturbations']),
                               requested_displacement_energy=requested_energy,
                               achieved_displacement_energy=achieved_energy,
                               displacement_energy_ratio=achieved_energy / requested_energy if requested_energy > 0 else 1.0)
                result['random'].append(control)
        append_result(results_path, result)
        results.append(result)
        if (index + 1) % 10 == 0 or index + 1 == len(eval_rows):
            status = {'phase': 'evaluate', 'done': len(results), 'total': len(eval_rows),
                      'baseline_correct': sum(r['baseline']['category'] == 'correct' for r in results)}
            print(json.dumps(status), flush=True)
            atomic_json(args.output / 'progress.json', status)
    assert len(results) == len(eval_rows)
    eligible = [r for r in results if r['pc1'] is not None]
    summary = {'status': 'complete', 'smoke': bool(args.smoke), 'cell': args.cell, 'run_id': run_id,
               'fit_sources': len({r['example_id'] for r in fit_rows}), 'eval_sources': len({r['example_id'] for r in eval_rows}),
               'planned_fit_sources': len(identity['fit_example_ids']), 'planned_eval_sources': len(identity['eval_example_ids']),
               'prompt_retention': retention,
               'fit_prompts': len(fit_rows), 'eval_prompts': len(results), 'baseline_correct': len(eligible),
               'baseline_categories': dict(Counter(r['baseline']['category'] for r in results)),
               'pc1_categories': dict(Counter(r['pc1']['category'] for r in eligible)),
               'random_categories_by_replicate': [dict(Counter(r['random'][j]['category'] for r in eligible)) for j in range(REPLICATES)],
               'maximum_random_norm_relative_error': max((p['relative_norm_error'] for r in eligible for c in r['random'] for p in c['perturbations']), default=0),
               'maximum_random_norm_absolute_error': max((p['absolute_norm_error'] for r in eligible for c in r['random'] for p in c['perturbations']), default=0),
               'maximum_random_norm_output_relative_error': max((p['output_relative_norm_error'] for r in eligible for c in r['random'] for p in c['perturbations']), default=0),
               'random_sites_using_absolute_tolerance': sum(p['used_absolute_tolerance'] for r in eligible for c in r['random'] for p in c['perturbations']),
               'random_intervention_sites': sum(len(c['perturbations']) for r in eligible for c in r['random']),
               'random_sites_original_tolerance_failed': sum(not p['original_tolerance_passed'] for r in eligible for c in r['random'] for p in c['perturbations']),
               'random_sites_certified_fallback': sum(p['certified_fallback_used'] for r in eligible for c in r['random'] for p in c['perturbations']),
               'random_sites_quantization_limited': sum(p['quantization_limited'] for r in eligible for c in r['random'] for p in c['perturbations']),
               'random_rollouts_with_original_tolerance_failure': sum(any(not p['original_tolerance_passed'] for p in c['perturbations']) for r in eligible for c in r['random']),
               'maximum_certified_norm_gap': max((p['certified_bracket']['norm_gap'] for r in eligible for c in r['random'] for p in c['perturbations'] if p['certified_fallback_used']), default=0),
               'minimum_random_displacement_energy_ratio': min((c['displacement_energy_ratio'] for r in eligible for c in r['random']), default=1.0),
               'maximum_random_displacement_energy_ratio': max((c['displacement_energy_ratio'] for r in eligible for c in r['random']), default=1.0),
               'all_fit_source_ids_disjoint_from_eval': True}
    atomic_json(args.output / 'summary.json', summary)
    atomic_json(args.output / 'progress.json', {'phase': 'complete', 'done': len(results), 'total': len(eval_rows)})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
