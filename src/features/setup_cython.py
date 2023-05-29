from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("/Users/oscarbalcells/Desktop/AI/task4/src/features/algos.pyx"),
    include_dirs=[numpy.get_include()]
)