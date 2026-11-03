import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conditional_audit import overlap_weights, permutation_test

class AuditChecks(unittest.TestCase):
    def test_balance_within_each_training_stratum(self):
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])
        w = overlap_weights(y, z)
        for group in [0, 1]:
            self.assertAlmostEqual(w[(z == group) & (y == 1)].sum(), w[(z == group) & (y == 0)].sum())
        self.assertEqual(w[-1], 0)
        self.assertAlmostEqual(w.mean(), 1)
    def test_no_overlap_stops(self):
        with self.assertRaisesRegex(ValueError, 'No training overlap'):
            overlap_weights([0, 0, 1, 1], [0, 0, 1, 1])
    def test_constant_score_cannot_reject(self):
        result = permutation_test(np.ones(8), [0, 1] * 4, [0] * 4 + [1] * 4)
        self.assertEqual(result['p_value'], 1)
    def test_no_test_information(self):
        result = permutation_test([1, 2, 3, 4], [0, 0, 1, 1], [0, 0, 1, 1])
        self.assertEqual(result['p_value'], 1)
        self.assertEqual(result['test_information'], 0)
