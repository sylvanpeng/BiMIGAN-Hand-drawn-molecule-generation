#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'bimigan',
    version          = '0.0.1',
    author           = 'sylvanpeng',
    author_email     = 'sylvanpeng@gmail.com',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "BiMIGAN paper code",
    packages         = setuptools.find_packages(
        include = [ 'bimigan', 'bimigan.*' ]
    ),
    install_requires = [ 'numpy', 'pandas', 'tqdm', 'Pillow' ],
)

