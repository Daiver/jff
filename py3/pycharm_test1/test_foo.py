from unittest import TestCase

import module1


class TestFoo(TestCase):
    def test_foo01(self):
        x = 10
        self.assertEqual(100, module1.foo(x))

    def test_foo02(self):
        x = 12
        self.assertEqual(144, module1.foo(x))
