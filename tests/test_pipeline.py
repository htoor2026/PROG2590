import unittest
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

class TestLogisticRegressionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load('notebooks/data/lr_model.pkl')
        X_train, cls.X_test, y_train, cls.y_test = joblib.load('notebooks/data/train_test_split.pkl')
        cls.predictions = cls.model.predict(cls.X_test)
        cls.probabilities = cls.model.predict_proba(cls.X_test)

    def test_output_shape(self):
        self.assertEqual(self.predictions.shape[0], self.X_test.shape[0])

    def test_prediction_labels(self):
        self.assertTrue(set(self.predictions).issubset({0, 1}))

    def test_no_nan_predictions(self):
        self.assertFalse(np.isnan(self.predictions).any())

    def test_probabilities_sum_to_one(self):
        row_sums = self.probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))

    def test_model_accuracy_above_threshold(self):
        acc = accuracy_score(self.y_test, self.predictions)
        self.assertGreater(acc, 0.80)

class TestRandomForestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load('notebooks/data/rf_model.pkl')
        cls.preprocessor = joblib.load('notebooks/data/preprocessor.pkl')

        X_train, cls.X_test, y_train, cls.y_test = joblib.load('notebooks/data/train_test_split.pkl')

        X_test_processed = cls.preprocessor.transform(cls.X_test)

        cls.predictions = cls.model.predict(X_test_processed)

    def test_rf_output_shape(self):
        self.assertEqual(len(self.predictions), len(self.X_test))

    def test_rf_prediction_labels(self):
        self.assertTrue(set(self.predictions).issubset({0, 1}))

    def test_rf_no_nan_predictions(self):
        self.assertFalse(np.isnan(self.predictions).any())

if __name__ == '__main__':
    unittest.main()