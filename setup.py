#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(name='21cmDELFI',
      version='v0.1',
      description='Application in 21 cm with LFI in TensorFlow',
      author='Justin Alsing, Xiaosheng-Zhao',
      url='https://github.com/Xiaosheng-Zhao/21cmDELFI',
      packages=find_packages(),
      install_requires=[
          "tensorflow >=v1.1.0, <v2.0",
          "getdist",
          "emcee>=v3.0.2",
          "mpi4py",
          "scipy<v1.8.0",
          "tqdm"
      ])

