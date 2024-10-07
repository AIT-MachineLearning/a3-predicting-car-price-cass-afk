import unittest
import joblib
import numpy as np
import pandas as pd
from app import out_check
from app import input_check




class TestModel(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(cls):
    #     # Load the model and scaler before running the tests
    #     cls.model = joblib.load("model.pkl")  # Load your machine learning model
    #     cls.scaler = joblib.load("a3_scaler.pkl")  # Load your scaler model
    #     cls.feature_names = ['engine', 'max_power', 'mileage', 'km_driven']

    def test_input_shape(self):
        # Test normal input values using a DataFrame
        normal_input = input_check(1000.0, 50.0, 10.0, 100000.0)
        # print(normal_input.shape)
        self.assertEqual(normal_input.shape, (1, 4), "Input shape should be (1, 4)")

        normal_input = input_check(2000.0, 100.0, 20.0, 150000)
        # print(normal_input.shape)
        self.assertEqual(normal_input.shape, (1, 4), "Input shape should be (1, 4)")


    def test_output_shape(self):
        # Test boundary values using a DataFrame
        self.pred = out_check(value1=1298.0, value2=90.00, value3=17.70 , value4=120000)
        # print(self.pred)
        # print("ulalalal")
        # print(self.pred)
        self.assertEqual(self.pred, True, "Output should be Integer")

        self.pred = out_check(value1=1000.0, value2=100.00, value3=20.70 ,value4= 150000)
        self.assertEqual(self.pred, True, "Output should be Integer")



if __name__ == '__main__':
    unittest.main()


