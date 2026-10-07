import unittest
import numpy as np

from aggregate_frozen_controls_v4 import KINDS, category, clustered_intervals, outcomes, reassignment_intervals, site_matches_original_rule


class AggregationTests(unittest.TestCase):
    def test_error_partition_from_predictions(self):
        for pair, expected in [(['A', 'D'], [0, 0, 0, 0, 0]),
                               (['A', 'A'], [1, 0, 1, 0, 0]),
                               (['A', 'B'], [1, 1, 0, 1, 0]),
                               (['B', 'C'], [1, 1, 0, 0, 1])]:
            result = {'prediction': pair, 'category': category(pair, ['A', 'D'])}
            np.testing.assert_array_equal(outcomes(result, ['A', 'D']), expected)

    def test_conditional_rate_resamples_sources_and_recomputes_denominator(self):
        # Every source has the same rate, with unequal numbers of clean-correct
        # variants. Their conditional rates must remain constant in every draw.
        rows = np.zeros((3, 1 + 2 * len(KINDS)))
        for i, n in enumerate((2, 4, 6)):
            rows[i, 0] = n
            rows[i, 1:6] = n * np.array([0.5, 0.25, 0.25, 0.25, 0])
            rows[i, 6:] = n * np.array([0.25, 0.25, 0, 0.25, 0])
        result = clustered_intervals(rows, draws=1000)
        self.assertEqual(result['destroyed']['pc1'], 0.5)
        self.assertEqual(result['destroyed']['random_mean'], 0.25)
        np.testing.assert_allclose(result['destroyed']['difference_ci95'], [0.25, 0.25])
        np.testing.assert_allclose(result['wrong_distinct_pair']['difference_ci95'], [0, 0])
        np.testing.assert_allclose(result['repeated_label']['difference_ci95'], [0.25, 0.25])

    def test_paired_equal_outcomes_have_zero_difference_interval(self):
        rows = np.zeros((4, 1 + 2 * len(KINDS)))
        rows[:, 0] = [1, 2, 3, 4]
        rows[:, 1] = [0, 1, 3, 2]
        rows[:, 6] = rows[:, 1]
        result = clustered_intervals(rows, draws=1000)
        self.assertEqual(result['destroyed']['difference'], 0)
        np.testing.assert_array_equal(result['destroyed']['difference_ci95'], [0, 0])


class ReassignmentTests(unittest.TestCase):
    def test_all_unflagged_bounds_collapse(self):
        rows = np.array([[1, 1/3, 1/3, 0], [2, 2/3, 2/3, 0], [4, 4/3, 4/3, 0]])
        result = reassignment_intervals(rows, draws=1000)
        self.assertAlmostEqual(result['lower'], 1/3)
        self.assertEqual(result['width'], 0)
        np.testing.assert_allclose(result['lower_ci95'], [1/3, 1/3])
        np.testing.assert_allclose(result['upper_ci95'], result['lower_ci95'])

    def test_flagged_bounds_against_exact_source_resampling(self):
        import itertools
        rows = np.array([[2, 0.5, 5/6, 1], [1, -2/3, -1/3, 1], [3, 1/3, 1, 2]])
        exact = []
        for indices in itertools.product(range(3), repeat=3):
            sample = rows[list(indices)].sum(axis=0)
            exact.append(sample[1:3]/sample[0])
        # The 27 resamples have equal probability; use inverse-CDF quantiles
        # of their discrete distribution, not interpolation between 27 ranks.
        expected = np.quantile(exact, [0.025, 0.975], axis=0, method='inverted_cdf')
        result = reassignment_intervals(rows, draws=20000, seed=29911)
        self.assertEqual(result['flagged_random_rollouts'], 4)
        self.assertAlmostEqual(result['width'], 4/18)
        self.assertAlmostEqual(result['upper'] - result['lower'], result['width'])
        np.testing.assert_allclose(result['lower_ci95'], expected[:,0])
        np.testing.assert_allclose(result['upper_ci95'], expected[:,1])

    def test_flag_uses_original_combined_rule(self):
        identity = {'norm_relative_tolerance': 0.005, 'norm_absolute_output_fraction': 1e-6}
        site = {'requested_norm': 0.0001, 'achieved_norm': 0.000101,
                'original_output_norm': 10, 'original_tolerance_passed': True,
                'quantization_limited': False, 'certified_fallback_used': False}
        self.assertTrue(site_matches_original_rule(site, identity))
        site.update(achieved_norm=0.00012, original_tolerance_passed=False,
                    quantization_limited=True, certified_fallback_used=True,
                    certified_bracket={})
        with self.assertRaises(KeyError):
            site_matches_original_rule(site, identity)


if __name__ == '__main__':
    unittest.main()
