import unittest

class TestHook(unittest.TestCase):
    def test_failed(self):
        assert 1 == 1

    def test_train(self):
        assert 1 == 1

    def test_evaluate(self):
        assert 1 == 1
