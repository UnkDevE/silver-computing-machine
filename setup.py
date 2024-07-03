#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="silver-compute-machine",
    ext_modules=cythonize('ver2.py')
)
