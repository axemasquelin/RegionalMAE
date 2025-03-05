import unittest
from RegionalMAE.data.preprocess import normalization

class TestNormalization(unittest.TestCase):
    def test_normalization_positive_values(self):
        data = [10, 20, 30, 40, 50]
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = normalization(data)
        self.assertEqual(result, expected)

    def test_normalization_negative_values(self):
        data = [-10, -20, -30, -40, -50]
        expected = [1.0, 0.75, 0.5, 0.25, 0.0]
        result = normalization(data)
        self.assertEqual(result, expected)

    def test_normalization_mixed_values(self):
        data = [-10, 0, 10, 20, 30]
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = normalization(data)
        self.assertEqual(result, expected)

    def test_normalization_single_value(self):
        data = [10]
        expected = [0.0]
        result = normalization(data)
        self.assertEqual(result, expected)

    def test_normalization_identical_values(self):
        data = [10, 10, 10, 10, 10]
        expected = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = normalization(data)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()