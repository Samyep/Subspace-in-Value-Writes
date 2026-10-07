#!/usr/bin/env python3
"""Capture one unchanged frozen-run norm failure; replay its scalar search on CPU.

This script never refits a direction, appends scientific results, or changes the
frozen runner. Capture rethrows the original exception after saving the site.
Replay changes only search resolution, preserving the captured FP32 unit vector
and BF16 candidate arithmetic. No tolerance is relaxed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

import frozen_direction_intervention as frozen

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_ROOT = ROOT / 'results/frozen_direction/norm_diagnostic'


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def assert_diagnostic_destination(path: Path) -> None:
    if not path.resolve().is_relative_to(DIAGNOSTIC_ROOT.resolve()):
        raise ValueError(f'New outputs must be under {DIAGNOSTIC_ROOT}')


def load_completed_ids_readonly(path: Path) -> tuple[set[str], int]:
    """Read complete lines only; preserve even an interrupted final line."""
    data = path.read_bytes() if path.exists() else b''
    complete_end = len(data) if data.endswith(b'\n') else data.rfind(b'\n') + 1
    rows = [json.loads(line) for line in data[:complete_end].splitlines()]
    ids = {row['prompt_id'] for row in rows}
    if len(ids) != len(rows):
        raise RuntimeError('Existing completed prompt IDs are not unique')
    return ids, len(data) - complete_end


def capture(args) -> None:
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    assert_diagnostic_destination(args.output)
    if args.output.exists():
        raise RuntimeError('Capture output already exists; choose a new directory')
    manifest_path = args.run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    identity = manifest['identity']
    if identity['cell'] != 'qwen_14b':
        raise ValueError('This bounded diagnostic is restricted to the failed Qwen-14B execution')
    data_path = ROOT / 'data/hotpot_400_prompts.jsonl'
    retention_path = args.run_dir / 'rendered_prompts.json'
    retention = {r['prompt_id']: r for r in json.loads(retention_path.read_text())}
    result_path = args.run_dir / 'per_prompt.jsonl'
    completed, incomplete_tail_bytes = load_completed_ids_readonly(result_path)
    retained_order = [pid for pid in identity['eval_prompt_ids'] if retention[pid]['retained']]
    remaining = [pid for pid in retained_order if pid not in completed]
    if not remaining:
        raise RuntimeError('No unevaluated retained prompt remains')
    selected_id = remaining[0]
    row = next(r for r in (json.loads(line) for line in data_path.read_text().splitlines()) if r['prompt_id'] == selected_id)
    args.output.mkdir(parents=True)
    protected = [manifest_path, retention_path, result_path, args.run_dir / 'frozen_pc1.npz',
                 args.run_dir / 'fit_statistics.json', Path(frozen.__file__)]
    before_contents = {str(path): path.read_bytes() for path in protected if path.exists()}
    setup = {'purpose': 'single-site numerical diagnosis only', 'utc': datetime.now(timezone.utc).isoformat(),
             'run_dir': args.run_dir.name,
             'prompt_id': selected_id, 'example_id': row['example_id'],
             'completed_prompt_count': len(completed), 'retained_eval_count': len(retained_order),
             'incomplete_tail_bytes_preserved': incomplete_tail_bytes,
             'cell': identity['cell'], 'model_id': identity['model_id'],
             'revision': identity['resolved_snapshot'], 'writer_window': identity['writer_window']}
    write_new_json(args.output / 'setup.json', setup)
    print(json.dumps({'phase': 'capture_setup', 'prompt_id': selected_id,
                      'completed_prompts': len(completed), 'output': str(args.output)}), flush=True)

    for root in (frozen.DEFAULT_RAG, frozen.DEFAULT_EXP):
        sys.path.insert(0, str(root))
    from run_reviewer_controls import _rerender_prompt_safe, resolve_label_token_ids
    from vw_rag.model_loading import load_model_and_tokenizer
    import transformers
    for key, current in [('torch', torch.__version__), ('transformers', transformers.__version__), ('numpy', np.__version__)]:
        if identity['versions'][key] != current:
            raise RuntimeError(f'{key} version mismatch: {current} versus {identity["versions"][key]}')
    torch.manual_seed(identity['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    original_matcher = frozen.equal_norm_displacement
    active_replicate = -1
    captured = False

    def capturing_matcher(before: torch.Tensor, unit: torch.Tensor, target: float):
        nonlocal captured
        try:
            return original_matcher(before, unit, target)
        except RuntimeError as error:
            if 'BF16 displacement norm cannot meet declared tolerance' not in str(error):
                raise
            caller = inspect.currentframe().f_back
            context = caller.f_locals
            layer = int(context['layer_index'])
            step = int(context['current_step'])
            failed_frame = error.__traceback__
            while failed_frame is not None and failed_frame.tb_frame.f_code.co_name != 'equal_norm_displacement':
                failed_frame = failed_frame.tb_next
            if failed_frame is None:
                raise RuntimeError('Original matching exception has no matcher frame') from error
            matching_locals = failed_frame.tb_frame.f_locals
            # Read the exact vector and endpoints from the original failed
            # function frame; do not recompute its internal normalization.
            used_unit = matching_locals['unit'].detach()
            legacy_best = matching_locals['best']
            legacy_low, legacy_high = matching_locals['low'], matching_locals['high']
            original_fp32 = before.float()
            low_after = (original_fp32 + legacy_low * used_unit).to(before.dtype)
            high_after = (original_fp32 + legacy_high * used_unit).to(before.dtype)
            def endpoint_summary(scale, after):
                delta = after.float() - original_fp32
                return {'scale': float(scale), 'effective_float32_scale': float(np.float32(scale)),
                        'norm_float32': float(delta.norm()), 'norm_float64': float(delta.double().norm())}
            output_norm_gpu = float(before.float().norm())
            metadata = {'prompt_id': selected_id, 'example_id': row['example_id'],
                        'replicate': active_replicate, 'step': step, 'layer': layer,
                        'target': float(target), 'original_output_norm_gpu': output_norm_gpu,
                        'tolerance_gpu': max(frozen.MATCH_RTOL * target,
                                             frozen.MATCH_OUTPUT_NORM_FLOOR * output_norm_gpu),
                        'relative_tolerance': frozen.MATCH_RTOL,
                        'absolute_output_fraction': frozen.MATCH_OUTPUT_NORM_FLOOR,
                        'original_error': str(error), 'before_dtype': str(before.dtype),
                        'unit_dtype': str(unit.dtype), 'device': str(before.device),
                        'gpu_name': torch.cuda.get_device_name(before.device),
                        'seed': frozen.random_seed(identity['cell'], selected_id, active_replicate),
                        'legacy_best': endpoint_summary(legacy_best[3], legacy_best[1]),
                        'legacy_low_endpoint': endpoint_summary(legacy_low, low_after),
                        'legacy_high_endpoint': endpoint_summary(legacy_high, high_after)}
            gpu_endpoint_tensors = {}
            try:
                # Certify the CPU bracket, then test those exact adjacent FP32
                # amplitudes on the still-live GPU. No model forward is added.
                cpu_search = diagnose_scalar_search(before.detach().cpu(), used_unit.cpu(),
                                                     float(target), metadata['tolerance_gpu'])
                write_new_json(args.output / 'capture_cpu_search.json', cpu_search)
                endpoint_records = {}
                for side in ('lower', 'upper'):
                    point = cpu_search['adjacent_float32_bracket'][side]
                    scalar = float(np.float32(point['scale']))
                    gpu_after = (original_fp32 + scalar * used_unit).to(before.dtype)
                    gpu_delta = gpu_after.float() - original_fp32
                    gpu_norm32 = float(gpu_delta.norm())
                    gpu_norm64 = float(gpu_delta.double().norm())
                    saved_gpu_after = gpu_after.detach().cpu().clone()
                    cpu_after = (before.detach().cpu().float() + scalar * used_unit.cpu()).to(before.dtype)
                    gpu_endpoint_tensors['certified_gpu_' + side + '_after_bf16'] = saved_gpu_after
                    gpu_endpoint_tensors['certified_cpu_' + side + '_after_bf16'] = cpu_after
                    endpoint_records[side] = {
                        'scale': scalar, 'scale_float32_bits': point['scale_float32_bits'],
                        'gpu_norm_float32': gpu_norm32, 'gpu_norm_float64': gpu_norm64,
                        'cpu_norm_float32': point['norm_float32'], 'cpu_norm_float64': point['norm_float64'],
                        'gpu_absolute_error_float32': abs(gpu_norm32 - target),
                        'gpu_absolute_error_float64': abs(gpu_norm64 - target),
                        'satisfies_frozen_tolerance_on_gpu': abs(gpu_norm32 - target) <= metadata['tolerance_gpu'],
                        'gpu_and_cpu_bf16_outputs_equal': bool(torch.equal(saved_gpu_after, cpu_after))}
                lower = endpoint_records['lower']
                upper = endpoint_records['upper']
                adjacent = upper['scale_float32_bits'] - lower['scale_float32_bits'] == 1
                straddles = lower['gpu_norm_float32'] < target <= upper['gpu_norm_float32']
                nearest_side = min(('lower', 'upper'), key=lambda side: endpoint_records[side]['gpu_absolute_error_float32'])
                nearest = endpoint_records[nearest_side]
                gpu_certificate = {
                    'endpoint_evaluations_on_gpu': 2, 'endpoints': endpoint_records,
                    'adjacent_float32_amplitudes': adjacent,
                    'gpu_norms_straddle_target': straddles,
                    'gpu_adjacent_bracket_certified': adjacent and straddles,
                    'nearest_endpoint_side': nearest_side,
                    'frozen_tolerance_attainable_on_gpu': nearest['satisfies_frozen_tolerance_on_gpu'] if adjacent and straddles else None,
                    'diagnosis': ('certified_gpu_bracket_has_an_endpoint_within_frozen_tolerance'
                                  if adjacent and straddles and nearest['satisfies_frozen_tolerance_on_gpu']
                                  else 'certified_gpu_scalar_staircase_has_no_value_within_frozen_tolerance'
                                  if adjacent and straddles
                                  else 'cpu_bracket_does_not_certify_gpu_crossing'),
                    'qualification': 'For the captured fixed vector, the original FP32 multiply/add and BF16 cast are coordinatewise monotone in a nonnegative FP32 amplitude; adjacent amplitudes straddling the GPU norm crossing bracket all reachable norms nearest the target.'}
                metadata['gpu_endpoint_certificate'] = gpu_certificate
                write_new_json(args.output / 'gpu_endpoint_certificate.json', gpu_certificate)
            except Exception as diagnosis_error:
                # Preserve the original failed site even if the extra numerical
                # diagnostic fails. The original matching error is rethrown.
                metadata['endpoint_diagnostic_error'] = repr(diagnosis_error)
            tensors = {'before_bf16': before.detach().cpu().clone(),
                       'unit_input_fp32': unit.detach().cpu().clone(),
                       'unit_used_fp32': used_unit.cpu().clone(),
                       'legacy_best_after_bf16': legacy_best[1].detach().cpu().clone(),
                       'legacy_low_after_bf16': low_after.detach().cpu().clone(),
                       'legacy_high_after_bf16': high_after.detach().cpu().clone(),
                       'target': float(target), 'metadata': metadata}
            tensors.update(gpu_endpoint_tensors)
            torch.save(tensors, args.output / 'failure_site.pt')
            write_new_json(args.output / 'failure_site.json', metadata)
            captured = True
            print(json.dumps({'phase': 'failure_captured', 'prompt_id': selected_id,
                              'replicate': active_replicate, 'step': step, 'layer': layer,
                              'target': target, 'original_error': str(error)}), flush=True)
            raise  # Preserve the original failure; never continue this condition.
        finally:
            if 'caller' in locals():
                del caller

    try:
        loaded = load_model_and_tokenizer(identity['model_id'], revision=identity['resolved_snapshot'],
                                          device=args.device, dtype='bf16', attn_implementation='eager',
                                          trust_remote_code=False)
        assert loaded.dtype == torch.bfloat16
        assert loaded.model.config._attn_implementation == 'eager'
        rendered = _rerender_prompt_safe(row, loaded.tokenizer, 10**9)
        assert rendered is not None
        rendered_tokens = len(loaded.tokenizer.encode(rendered, add_special_tokens=False))
        if rendered_tokens != retention[selected_id]['tokens']:
            raise RuntimeError('Reconstructed prompt token count differs from the frozen run')
        row['prompt'] = rendered
        with np.load(args.run_dir / 'frozen_pc1.npz') as saved:
            component = saved['component'].copy()
        pc1_unit = torch.tensor(component, dtype=torch.float32, device=loaded.device)
        pc1_unit /= pc1_unit.norm()
        labels = resolve_label_token_ids(loaded.tokenizer)
        baseline = frozen.decode(loaded, row, labels, identity['writer_window'])
        if baseline['category'] != 'correct':
            raise RuntimeError('Reconstruction baseline is not eligible; trajectory differs from the failed run')
        pc1 = frozen.decode(loaded, row, labels, identity['writer_window'], direction=pc1_unit)
        schedule = {(p['step'], p['layer']): p['achieved_norm'] for p in pc1['perturbations']}
        write_new_json(args.output / 'pc1_norm_schedule.json',
                       [{'step': step, 'layer': layer, 'target': target} for (step, layer), target in schedule.items()])
        frozen.equal_norm_displacement = capturing_matcher
        for active_replicate in range(identity['random_replicates']):
            seed = frozen.random_seed(identity['cell'], selected_id, active_replicate)
            direction = frozen.random_unit(len(component), seed, loaded.device)
            # Return values deliberately not persisted or aggregated: this is a numerical capture.
            frozen.decode(loaded, row, labels, identity['writer_window'], direction=direction, norm_schedule=schedule)
        write_new_json(args.output / 'not_reproduced.json', {'status': 'no_matching_norm_failure_reproduced',
                                                           'prompt_id': selected_id})
    finally:
        frozen.equal_norm_displacement = original_matcher
        unchanged = all(Path(name).read_bytes() == contents for name, contents in before_contents.items())
        write_new_json(args.output / 'readonly_validation.json',
                       {'all_protected_files_unchanged': unchanged, 'failure_captured': captured})
        if not unchanged:
            raise RuntimeError('A protected source artifact changed during diagnostic capture')


def float32_bits(value: float) -> int:
    return int(np.asarray(np.float32(value)).view(np.uint32))


def bits_float32(bits: int) -> float:
    return float(np.asarray(np.uint32(bits)).view(np.float32))


def diagnose_scalar_search(before: torch.Tensor, unit_used: torch.Tensor, target: float,
                           tolerance: float) -> dict:
    """CPU replay using the exact captured normalized vector (no renormalization)."""
    assert before.device.type == unit_used.device.type == 'cpu'
    assert before.dtype == torch.bfloat16 and unit_used.dtype == torch.float32
    original = before.float()
    trace = []
    def candidate(scale: float, phase: str) -> dict:
        effective_scale = float(np.float32(scale))
        # Separate FP32 multiply/add followed by BF16 assignment, as in runner.
        after = (original + effective_scale * unit_used).to(before.dtype)
        delta = after.float() - original
        norm32 = float(delta.norm())
        norm64 = float(delta.double().norm())
        record = {'scale': effective_scale, 'scale_float32_bits': float32_bits(effective_scale),
                  'norm_float32': norm32, 'norm_float64': norm64,
                  'absolute_error_float32': abs(norm32 - target),
                  'absolute_error_float64': abs(norm64 - target),
                  'satisfies_frozen_tolerance_float32': abs(norm32 - target) <= tolerance,
                  'changed_coordinates': int(torch.count_nonzero(delta)),
                  'phase': phase}
        trace.append(record)
        return record

    # Replay the old scalar schedule to distinguish insufficient search from a
    # genuinely absent value on the quantized monotone staircase.
    low, high = 0.0, target
    current = candidate(high, 'legacy_initial')
    steps = 1
    best = current
    while current['norm_float32'] < target and steps < 16:
        low, high = high, high * 2
        current = candidate(high, 'legacy_expand')
        steps += 1
        if current['absolute_error_float32'] < best['absolute_error_float32']:
            best = current
    for _ in range(24):
        if best['absolute_error_float32'] <= frozen.MATCH_RTOL * target:
            break
        middle = (low + high) / 2
        current = candidate(middle, 'legacy_bisect')
        if current['absolute_error_float32'] < best['absolute_error_float32']:
            best = current
        if current['norm_float32'] < target:
            low = middle
        else:
            high = middle
    legacy = {'best': best, 'final_low': candidate(low, 'legacy_low_endpoint'),
              'final_high': candidate(high, 'legacy_high_endpoint')}

    # Ordered positive FP32 bit patterns cover every scalar that can affect
    # this tensor operation. Bracket globally from zero, then binary-search
    # until the lower/upper amplitudes are adjacent representable FP32 values.
    lower_bits = 0
    upper_scalar = max(float(np.float32(target)), float(np.finfo(np.float32).tiny))
    upper = candidate(upper_scalar, 'certify_initial')
    expansions = 0
    while upper['norm_float32'] < target:
        lower_bits = upper['scale_float32_bits']
        upper_scalar = float(np.float32(upper_scalar * 2))
        if not np.isfinite(upper_scalar):
            raise RuntimeError('Could not form a finite scalar bracket')
        upper = candidate(upper_scalar, 'certify_expand')
        expansions += 1
        if expansions > 150:
            raise RuntimeError('Excessive scalar bracket expansion')
    upper_bits = upper['scale_float32_bits']
    iterations = 0
    while upper_bits - lower_bits > 1:
        middle_bits = (lower_bits + upper_bits) // 2
        point = candidate(bits_float32(middle_bits), 'certify_bisect_bits')
        if point['norm_float32'] < target:
            lower_bits = middle_bits
        else:
            upper_bits = middle_bits
        iterations += 1
        assert iterations <= 32
    lower = candidate(bits_float32(lower_bits), 'certified_lower_endpoint')
    upper = candidate(bits_float32(upper_bits), 'certified_upper_endpoint')
    assert upper_bits - lower_bits == 1
    assert lower['norm_float32'] < target <= upper['norm_float32']
    nearest = min([lower, upper], key=lambda p: p['absolute_error_float32'])
    attainable = nearest['satisfies_frozen_tolerance_float32']
    return {'target': target, 'frozen_tolerance': tolerance, 'unit_renormalized_on_cpu': False,
            'original_output_norm_cpu_float32': float(original.norm()),
            'original_output_norm_cpu_float64': float(original.double().norm()),
            'legacy_replay': legacy,
            'adjacent_float32_bracket': {'lower': lower, 'upper': upper, 'bits_difference': 1,
                                       'binary_search_iterations': iterations, 'expansions': expansions},
            'nearest_endpoint': nearest, 'frozen_tolerance_attainable_in_cpu_candidate_arithmetic': attainable,
            'diagnosis': ('search_resolution_or_runtime_arithmetic_issue' if attainable
                          else 'no_reachable_scalar_within_tolerance_in_cpu_candidate_arithmetic'),
            'qualification': 'Certification is for the captured vector and explicit CPU FP32 multiply/add, BF16 cast, and FP32 norm. GPU endpoint replay is required before asserting identical device arithmetic.',
            'candidate_trace': trace}


def replay(args) -> None:
    assert_diagnostic_destination(args.output)
    endpoints_path = args.output.with_suffix('.endpoints.pt')
    if args.output.exists() or endpoints_path.exists():
        raise RuntimeError('Replay outputs already exist; choose a new output filename')
    capture_data = torch.load(args.capture, map_location='cpu', weights_only=True)
    metadata = capture_data['metadata']
    result = diagnose_scalar_search(capture_data['before_bf16'], capture_data['unit_used_fp32'],
                                    float(capture_data['target']), float(metadata['tolerance_gpu']))
    result.update(capture_path=str(args.capture.resolve()), capture_metadata=metadata)
    original = capture_data['before_bf16'].float()
    unit = capture_data['unit_used_fp32']
    endpoints = {}
    for side in ('lower', 'upper'):
        point = result['adjacent_float32_bracket'][side]
        after = (original + point['scale'] * unit).to(torch.bfloat16)
        endpoints[side + '_after_bf16'] = after
        endpoints[side + '_scale'] = point['scale']
    # Compare the CPU replay at the original GPU best amplitude directly.
    gpu_best_scale = metadata['legacy_best']['effective_float32_scale']
    cpu_at_gpu_best = (original + gpu_best_scale * unit).to(torch.bfloat16)
    result['cpu_output_at_legacy_gpu_best_matches_saved_gpu_output'] = bool(
        torch.equal(cpu_at_gpu_best, capture_data['legacy_best_after_bf16']))
    endpoints_path.parent.mkdir(parents=True, exist_ok=True)
    with endpoints_path.open('xb') as handle:
        torch.save(endpoints, handle)
    result['endpoint_tensor_path'] = str(endpoints_path.resolve())
    write_new_json(args.output, result)
    print(json.dumps({key: result[key] for key in ['target', 'frozen_tolerance', 'diagnosis',
                                                  'nearest_endpoint', 'adjacent_float32_bracket']}), flush=True)


def self_test() -> None:
    before = torch.ones(128, dtype=torch.bfloat16)
    before[0] = 0.03125
    unit = torch.zeros(128)
    unit[0] = 1
    target = 5.03 * 2**-12
    strict = diagnose_scalar_search(before, unit, target, 0.005 * target)
    assert not strict['frozen_tolerance_attainable_in_cpu_candidate_arithmetic']
    assert strict['adjacent_float32_bracket']['bits_difference'] == 1
    relaxed_for_test = diagnose_scalar_search(before, unit, target, 1e-6 * float(before.float().norm()))
    assert relaxed_for_test['frozen_tolerance_attainable_in_cpu_candidate_arithmetic']
    assert strict['nearest_endpoint']['norm_float32'] == 5 * 2**-12
    print('PASS: adjacent-FP32 endpoint certification, attainable and unattainable synthetic cases')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    capture_parser = sub.add_parser('capture', help='GPU reconstruction; archives first norm failure and rethrows it')
    capture_parser.add_argument('--run-dir', type=Path, required=True)
    capture_parser.add_argument('--output', type=Path, required=True)
    capture_parser.add_argument('--device', default='cuda')
    replay_parser = sub.add_parser('replay', help='CPU-only scalar search certification for captured tensors')
    replay_parser.add_argument('--capture', type=Path, required=True)
    replay_parser.add_argument('--output', type=Path, required=True)
    sub.add_parser('self-test', help='CPU-only synthetic scalar-search checks; writes no artifacts')
    args = parser.parse_args()
    if args.mode == 'capture':
        capture(args)
    elif args.mode == 'replay':
        replay(args)
    else:
        self_test()


if __name__ == '__main__':
    main()
