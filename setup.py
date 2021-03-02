#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='ImageClassification',
      version='1.0',
      description='First iteration of an image classifier',
      author='Ahmed Alsabag',
      author_email='ahmed.alsabag@gmail.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
     )