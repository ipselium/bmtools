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
# Creation Date : 2019-10-11 - 17:01:15
"""
-----------
DOCSTRING

-----------
"""

import numpy as np
from bmtools import Compare

def star_op(x):
    """ Double star operator. """
    return x**0.5

def pow_op(x):
    """ pow function. """
    return pow(x, 0.5)

def sqrt_op(x):
    """ numpy.sqrt function. """
    return np.sqrt(x)

if __name__ == "__main__":

    # Single comparison
    bm1 = Compare(pow_op, star_op, sqrt_op, unit='ms')
    bm1.run(fargs=(np.random.rand(1000000), ))
    bm1.display()

    # Parametric comparison
    bm2 = Compare(pow_op, star_op, sqrt_op, unit='ms')
    for n in [2**n for n in range(16, 23)]:
        bm2.run(fargs=(np.random.rand(n), ), desc=n)

    bm2.display()
    bm2.bars()
