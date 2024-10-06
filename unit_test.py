import unittest
import joblib
import numpy as np
import pandas as pd
import app as model


class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the model and scaler before running the tests
        cls.model = joblib.load("model.pkl")  # Load your machine learning model
        cls.scaler = joblib.load("a3_scaler.pkl")  # Load your scaler model
        cls.feature_names = ['engine', 'max_power', 'mileage', 'km_driven']

    def test_input_shape(self):
        # Test normal input values using a DataFrame
        normal_input = pd.DataFrame([[1396.0, 103.52, 23.00, 120000]], columns=self.feature_names)
        self.assertEqual(normal_input.shape, (1, 4), "Input shape should be (1, 4)")

        # Scale the input data before prediction
        scaled_normal_input = self.scaler.transform(normal_input)
        self.assertEqual(scaled_normal_input.shape, (1, 4), "Scaled input shape should be (1, 4)")

        # Make prediction using scaled normal input
        prediction = self.model.predict(scaled_normal_input)
        self.assertIsInstance(prediction, np.ndarray, "Prediction should be a numpy array")
        self.assertEqual(prediction.shape, (1,), "Prediction output shape should be (1,)")

    def test_output_shape(self):
        # Test boundary values using a DataFrame
        boundary_input = pd.DataFrame([[1000, 50, 10, 10000]], columns=self.feature_names)
        scaled_boundary_input = self.scaler.transform(boundary_input)
        prediction = self.model.predict(scaled_boundary_input)
        self.assertEqual(prediction.shape, (1,), "Output shape should be (1,)")

        # Test extreme values using a DataFrame
        extreme_input = pd.DataFrame([[1298.0, 90.00, 17.70 , 120000]], columns=self.feature_names)
        scaled_extreme_input = self.scaler.transform(extreme_input)
        prediction = self.model.predict(scaled_extreme_input)
        self.assertEqual(prediction.shape, (1,), "Output shape should be (1,)")

if __name__ == '__main__':
    unittest.main()


