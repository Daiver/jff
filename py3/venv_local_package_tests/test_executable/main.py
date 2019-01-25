#!/usr/bin/env python
import test_package
import test_package.awesome_module
import test_package.awesome_module2

import yet_another_package
import file_mod

print('This is low level file!')
test_package.awesome_module.foo()
yet_another_package.foo()
file_mod.foo()
