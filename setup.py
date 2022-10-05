#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from setuptools import setup
import sys
import warnings

NAME = 'cubic_strat'
DESCRIPTION = 'Higher-order stochastic integration based on cubic stratification'

with open('README.md') as f:
    long_description = f.read()

METADATA = dict(
    name=NAME,
    version='0.1',
    url='http://github.com/nchopin/cubic_strat/',
    license='MIT',
    author='Nicolas Chopin',
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'findiff',
                      'particles'
                      ],
    author_email='nicolas.chopin@ensae.fr',
    description=DESCRIPTION,
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages=[NAME],
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
    ]
)

setup(**METADATA)
