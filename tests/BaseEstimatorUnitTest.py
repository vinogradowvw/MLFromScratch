from ml_from_scratch.base import Estimator
import unittest


class CoefEstimator(Estimator):
    def __init__(self):
        self.coef = []
        super().__init__()

    def fit(self, X, y):
        return []

    def predict(self, X):
        return []


class WeightsEstimator(Estimator):
    def __init__(self):
        self.weights = []
        super().__init__()

    def fit(self, X, y):
        return []

    def predict(self, X):
        return []


class ImportanceEstimator(Estimator):
    def __init__(self):
        self.feature_importances = []
        super().__init__()

    def fit(self, X, y):
        return []

    def predict(self, X):
        return []


class BaseEstimatorUnitTest(unittest.TestCase):
    def test_coef_estimator_initialization(self):
        coef_estimator = CoefEstimator()
        self.assertIsInstance(coef_estimator, CoefEstimator)
        self.assertEqual(coef_estimator.coef, [])

    def test_weights_estimator_initialization(self):
        weights_estimator = WeightsEstimator()
        self.assertIsInstance(weights_estimator, WeightsEstimator)
        self.assertEqual(weights_estimator.weights, [])

    def test_importance_estimator_initialization(self):
        importance_estimator = ImportanceEstimator()
        self.assertIsInstance(importance_estimator, ImportanceEstimator)
        self.assertEqual(importance_estimator.feature_importances, [])


if __name__ == "__main__":
    unittest.main()
