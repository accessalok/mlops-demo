import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import dvc.api
import os

# Import the functions from the script
from scripts.train import load_data

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load data and initialize model for testing.
        """
        cls.X, cls.y = load_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)
        cls.model = LogisticRegression(max_iter=200)
        cls.model.fit(cls.X_train, cls.y_train)

    def test_load_data(self):
        """
        Test that data is loaded and processed correctly.
        """
        self.assertEqual(self.X.shape[0], 1143)  # Adjust expected value based on the dataset
        self.assertEqual(self.y.shape[0], 1143)  # Adjust expected value based on the dataset
        self.assertIn('fixed acidity', self.X.columns)  # Check for expected feature

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup after tests.
        """
        # Any necessary cleanup can be performed here.
        pass

if __name__ == '__main__':
    unittest.main()
