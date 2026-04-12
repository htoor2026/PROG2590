import unittest
import numpy as np
import joblib

class TestLogisticRegressionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model and test data once for all tests."""
        cls.model = joblib.load('../notebooks/data/lr_model.pkl')
        X_train, cls.X_test, y_train, cls.y_test = joblib.load('../notebooks/data/train_test_split.pkl')
        cls.predictions = cls.model.predict(cls.X_test)
        cls.probabilities = cls.model.predict_proba(cls.X_test)

    def test_output_shape(self):
        """Predictions count must match test set size."""
        self.assertEqual(self.predictions.shape[0], self.X_test.shape[0])

    def test_prediction_labels(self):
        """Predictions must only contain valid class labels 0 or 1."""
        self.assertTrue(set(self.predictions).issubset({0, 1}))

    def test_no_nan_predictions(self):
        """Predictions must not contain any NaN values."""
        self.assertFalse(np.isnan(self.predictions).any())

    def test_probabilities_sum_to_one(self):
        """Each row's predicted probabilities must sum to 1."""
        row_sums = self.probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))

    def test_model_accuracy_above_threshold(self):
        """Model accuracy must be above 80%."""
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(self.y_test, self.predictions)
        self.assertGreater(acc, 0.80)

if __name__ == '__main__':
    unittest.main()