import unittest
from model.ml_model import train_model

class TestModel(unittest.TestCase):
    def test_train_model(self):
        model = train_model()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
