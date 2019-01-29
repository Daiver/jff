import unittest


class TestCase(unittest.TestCase):
    def test_useless(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
