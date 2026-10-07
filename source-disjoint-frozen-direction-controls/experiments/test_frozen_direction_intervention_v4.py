"""CPU verification of the separate, explicitly approximate v4 norm control."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import frozen_direction_intervention as legacy
import frozen_direction_intervention_v4 as v4


CAPTURE = Path(__file__).resolve().parents[1] / 'results/frozen_direction/norm_diagnostic/capture_3279792/failure_site.pt'


class V4NormControlTests(unittest.TestCase):
    def test_frozen_source_and_original_matcher_are_preserved(self):
        old_ast = ast.parse(Path(legacy.__file__).read_text())
        new_ast = ast.parse(Path(v4.__file__).read_text())
        old_functions = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
        new_functions = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(ast.dump(ast.Module(body=old_functions['equal_norm_displacement'].body, type_ignores=[])),
                         ast.dump(ast.Module(body=new_functions['_legacy_equal_norm_displacement'].body, type_ignores=[])))
        for name in ['source_split', 'random_seed', 'random_unit', 'project_remove', 'decode', 'fit_pc1', 'load_mistral_tokenizer']:
            self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]), name)
        self.assertEqual(v4.CELLS, legacy.CELLS)
        self.assertEqual((v4.MATCH_RTOL, v4.MATCH_OUTPUT_NORM_FLOOR),
                         (legacy.MATCH_RTOL, legacy.MATCH_OUTPUT_NORM_FLOOR))

    def test_captured_site_certifies_same_gap_and_lower_endpoint(self):
        data = torch.load(CAPTURE, map_location='cpu', weights_only=True)
        metadata = data['metadata']
        after, record = v4._certify_nearest_quantized_edit(
            data['before_bf16'], data['unit_used_fp32'], data['target'], metadata['original_output_norm_gpu'])
        bracket = record['certified_bracket']
        self.assertEqual(bracket['lower']['scalar_float32_bits'], 1009352440)
        self.assertEqual(bracket['upper']['scalar_float32_bits'], 1009352441)
        self.assertEqual(bracket['selected_endpoint'], 'lower')
        self.assertEqual(bracket['adjacent_scalar_bits_difference'], 1)
        self.assertTrue(bracket['straddles_target'])
        self.assertFalse(record['original_tolerance_passed'])
        self.assertTrue(record['quantization_limited'])
        self.assertTrue(record['certified_fallback_used'])
        self.assertFalse(bracket['lower']['original_tolerance_passed'])
        self.assertFalse(bracket['upper']['original_tolerance_passed'])
        self.assertFalse(record['used_absolute_tolerance'])
        self.assertTrue(torch.equal(after, data['certified_gpu_lower_after_bf16']))

    def test_captured_failure_wrapper_does_not_reseed_or_change_vector(self):
        data = torch.load(CAPTURE, map_location='cpu', weights_only=True)
        unit = data['unit_input_fp32'].clone()
        before_unit = unit.clone()
        state_before = torch.get_rng_state().clone()
        with patch.object(v4, 'random_unit', side_effect=AssertionError('No vector resampling allowed')), \
             patch('numpy.random.default_rng', side_effect=AssertionError('No RNG allowed in fallback')):
            after, record = v4.equal_norm_displacement(data['before_bf16'], unit, data['target'])
        self.assertTrue(torch.equal(unit, before_unit))
        self.assertTrue(torch.equal(torch.get_rng_state(), state_before))
        self.assertTrue(record['certified_fallback_used'])
        self.assertFalse(record['original_tolerance_passed'])
        self.assertEqual(record['certified_bracket']['selected_endpoint'], 'lower')
        self.assertTrue(torch.equal(after, data['certified_gpu_lower_after_bf16']))

    def test_successful_original_edits_are_bit_identical_with_unchanged_fields(self):
        generator = torch.Generator().manual_seed(917)
        checked = 0
        for dimension in [128, 3072, 5120]:
            before = torch.zeros(dimension, dtype=torch.bfloat16)
            direction = torch.randn(dimension, generator=generator)
            direction /= direction.norm()
            for target in [0.0, 0.01, 0.5, 3.0]:
                expected_after, expected_record = legacy.equal_norm_displacement(before, direction, target)
                actual_after, actual_record = v4.equal_norm_displacement(before, direction, target)
                self.assertTrue(torch.equal(actual_after, expected_after))
                for key, value in expected_record.items():
                    self.assertEqual(actual_record[key], value)
                self.assertTrue(actual_record['original_tolerance_passed'])
                self.assertFalse(actual_record['certified_fallback_used'])
                self.assertFalse(actual_record['quantization_limited'])
                checked += 1
        self.assertEqual(checked, 12)

    def test_exactly_representable_norm_uses_original_success_path(self):
        before = torch.ones(4, dtype=torch.bfloat16)
        unit = torch.tensor([1., 0., 0., 0.])
        target = 2 * 2**-7
        after, record = v4.equal_norm_displacement(before, unit, target)
        self.assertEqual(float((after.float() - before.float()).norm()), target)
        self.assertTrue(record['original_tolerance_passed'])
        self.assertFalse(record['certified_fallback_used'])

    def test_large_quantization_mismatch_remains_explicitly_flagged(self):
        before = torch.tensor([1.0], dtype=torch.bfloat16)
        unit = torch.tensor([1.0])
        target = 0.01
        with self.assertRaises(RuntimeError):
            legacy.equal_norm_displacement(before, unit, target)
        after, record = v4.equal_norm_displacement(before, unit, target)
        self.assertEqual(record['achieved_norm'], 2**-7)
        self.assertGreater(record['relative_norm_error'], 0.20)
        self.assertFalse(record['original_tolerance_passed'])
        self.assertTrue(record['quantization_limited'])
        self.assertTrue(record['certified_fallback_used'])
        self.assertFalse(record['used_absolute_tolerance'])
        self.assertGreater(record['absolute_norm_error'], record['certified_bracket']['original_combined_tolerance'])
        self.assertEqual(float((after.float() - before.float()).norm()), record['achieved_norm'])

    def test_exact_tie_uses_lower_endpoint(self):
        before = torch.tensor([1.0], dtype=torch.bfloat16)
        unit = torch.tensor([1.0])
        after, record = v4._certify_nearest_quantized_edit(before, unit, 1.5 * 2**-7, 1.0)
        self.assertEqual(record['certified_bracket']['selected_endpoint'], 'lower')
        self.assertEqual(record['achieved_norm'], 2**-7)


if __name__ == '__main__':
    unittest.main()
