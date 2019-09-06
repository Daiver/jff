from distutils.core import setup, Extension


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        res = pybind11.get_include(self.user)
        return res


ext_modules = [
    Extension(
        'demo',
        ['demo.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        libraries=["casadi"],
        language='c++'
    ),
]

setup(
    name='PackageName',
    version='1.0',
    description='This is a demo package',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.3'],
    setup_requires=['pybind11>=2.3'],
)
