"""Step 5: evaluation metrics."""
import unittest

from shipment_risk._metrics import average_precision, brier_score


class AveragePrecisionTests(unittest.TestCase):
    def test_textbook_example(self):
        # Same numbers as scikit-learn's documented example: AP = 0.8333...
        self.assertAlmostEqual(average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 5 / 6)

    def test_perfect_and_worst_ranking(self):
        self.assertEqual(average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertAlmostEqual(average_precision([0, 0, 1], [0.9, 0.8, 0.1]), 1 / 3)

    def test_all_tied_scores_give_the_positive_rate_whatever_the_order(self):
        self.assertEqual(average_precision([1, 0, 0, 0], [0.5] * 4), 0.25)
        self.assertEqual(average_precision([0, 0, 0, 1], [0.5] * 4), 0.25)

    def test_no_positives_means_undefined(self):
        self.assertIsNone(average_precision([0, 0], [0.3, 0.7]))


class BrierTests(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(brier_score([1, 0], [0.8, 0.3]), (0.04 + 0.09) / 2)
        self.assertIsNone(brier_score([], []))

    def test_constant_base_rate_scores_b_times_one_minus_b(self):
        labels = [1] * 10 + [0] * 90
        self.assertAlmostEqual(brier_score(labels, [0.1] * 100), 0.1 * 0.9)


if __name__ == "__main__":
    unittest.main()
