import unittest
from src.core.model import HybridConceptBottleneckModel

class TestNeuroGuard(unittest.TestCase):
    def test_model(self):
        model = HybridConceptBottleneckModel()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()