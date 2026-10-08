#!/usr/bin/env python3
"""Validate the v4 numerical fallback on a saved GPU site, without any model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

import frozen_direction_intervention_v4 as v4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / 'results/frozen_direction/norm_diagnostic/capture_qwen14b_norm_site/failure_site.pt'
PREFLIGHT_ROOT = ROOT / 'results/frozen_direction/main_v4/preflight'
EXPECTED_LOWER_BITS = 1009352440
EXPECTED_UPPER_BITS = 1009352441
EXPECTED_LOWER_NORM = 0.00472657335922122
EXPECTED_UPPER_NORM = 0.004826403688639402
def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@torch.inference_mode()
def validate(capture_path: Path, device: torch.device) -> dict:
    require(device.type == 'cuda' and torch.cuda.is_available(), 'This regression must run on an allocated GPU')
    original_site = torch.load(capture_path, map_location='cpu', weights_only=True)
    metadata = original_site['metadata']
    before = original_site['before_bf16'].to(device)
    unit = original_site['unit_used_fp32'].to(device)
    target = float(original_site['target'])
    original_norm = float(metadata['original_output_norm_gpu'])
    require(before.dtype == torch.bfloat16 and unit.dtype == torch.float32, 'Captured tensor dtypes changed')
    before_copy = before.clone()
    unit_copy = unit.clone()
    cpu_rng_before = torch.get_rng_state().clone()
    gpu_rng_before = torch.cuda.get_rng_state(device).clone()
    after, record = v4._certify_nearest_quantized_edit(before, unit, target, original_norm)
    torch.cuda.synchronize(device)
    bracket = record['certified_bracket']
    lower, upper = bracket['lower'], bracket['upper']
    expected_after = original_site['certified_gpu_lower_after_bf16']
    require(torch.equal(after.cpu(), expected_after), 'V4 GPU edit differs from the previously captured GPU lower endpoint')
    require(lower['scalar_float32_bits'] == EXPECTED_LOWER_BITS, 'Lower adjacent FP32 amplitude differs')
    require(upper['scalar_float32_bits'] == EXPECTED_UPPER_BITS, 'Upper adjacent FP32 amplitude differs')
    require(lower['achieved_norm'] == EXPECTED_LOWER_NORM, 'Lower actual FP32 norm differs')
    require(upper['achieved_norm'] == EXPECTED_UPPER_NORM, 'Upper actual FP32 norm differs')
    require(bracket['adjacent_scalar_bits_difference'] == 1 and bracket['straddles_target'], 'Adjacent GPU bracket not certified')
    require(bracket['selected_endpoint'] == 'lower', 'Nearest-endpoint choice changed')
    require(not record['original_tolerance_passed'], 'The original tolerance violation was hidden')
    require(record['certified_fallback_used'] and record['quantization_limited'], 'Required fallback quality flags missing')
    require(not lower['original_tolerance_passed'] and not upper['original_tolerance_passed'], 'An endpoint unexpectedly claims to meet the original criterion')
    require(torch.equal(before, before_copy) and torch.equal(unit, unit_copy), 'Certification mutated its activation or direction')
    require(torch.equal(torch.get_rng_state(), cpu_rng_before), 'Certification changed CPU RNG state')
    require(torch.equal(torch.cuda.get_rng_state(device), gpu_rng_before), 'Certification changed GPU RNG state')

    # These checks exercise the unchanged original-success path as well.
    simple_before = torch.ones(4, dtype=torch.bfloat16, device=device)
    simple_unit = torch.tensor([1., 0., 0., 0.], dtype=torch.float32, device=device)
    zero_after, zero_record = v4.equal_norm_displacement(simple_before, simple_unit, 0.0)
    require(torch.equal(zero_after, simple_before), 'Zero edit changed the activation')
    require(zero_record['original_tolerance_passed'] and not zero_record['certified_fallback_used'], 'Zero edit did not retain the original-success path')
    exact_target = 2 * 2**-7
    exact_after, exact_record = v4.equal_norm_displacement(simple_before, simple_unit, exact_target)
    require(float((exact_after.float() - simple_before.float()).norm()) == exact_target, 'Exactly representable edit norm changed')
    require(exact_record['original_tolerance_passed'] and not exact_record['certified_fallback_used'], 'Representable edit did not retain the original-success path')
    return {'captured_site': {'prompt_id': metadata['prompt_id'], 'layer': metadata['layer'],
                              'step': metadata['step'], 'replicate': metadata['replicate']},
            'target': target, 'captured_gpu_tensor_byte_identical': True,
            'adjacent_amplitudes_and_actual_norms_identical': True,
            'activation_and_direction_unchanged': True, 'rng_states_unchanged': True,
            'zero_edit_original_path_passed': True, 'representable_edit_original_path_passed': True,
            'certifier_record': record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument('--output', type=Path, help='New JSON file under main_v4/preflight; defaults to a job-specific name')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_id = os.environ.get('SLURM_JOB_ID') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or PREFLIGHT_ROOT / f'gpu_norm_regression_{run_id}.json'
    require(output.resolve().is_relative_to(PREFLIGHT_ROOT.resolve()), 'Output must be inside main_v4/preflight')
    require(not output.exists(), 'Output already exists; select a new preflight filename')
    report = {'purpose': 'numerical GPU regression only; no model loading or scientific evaluation',
              'utc': datetime.now(timezone.utc).isoformat(),
              'torch_version': str(torch.__version__)}
    try:
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        result = validate(args.capture, device)
        report.update(status='passed', device=str(device), gpu_name=torch.cuda.get_device_name(device), **result)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'passed', 'output': str(output), 'gpu': report['gpu_name'],
                      'captured_gpu_tensor_byte_identical': True,
                      'original_tolerance_passed': result['certifier_record']['original_tolerance_passed']}), flush=True)


if __name__ == '__main__':
    main()
