#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2019 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of bmtools
#
# bmtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# bmtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with bmtools. If not, see <http://www.gnu.org/licenses/>.
#
#
# Creation Date : 2019-10-10 - 9:10:17
"""
-----------
DOCSTRING

@author: Cyril Desjouy
"""

from setuptools import setup, find_packages

setup(

    name='bmtools',
    description="System informations",
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    version='0.1.6',
    license="GPL",
    url='http://github.com/ipselium/bmtools',
    author="Cyril Desjouy",
    author_email="cyril.desjouy@univ-lemans.fr",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'matplotlib'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ]
)
