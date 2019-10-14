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
# Creation Date : 2019-10-11 - 11:37:28
"""
-----------
Multi probes example
-----------
"""

import time
from bmtools import TimeProbes
import test_probes_away


def sleep_func():
    """ Example function. """
    for _ in range(10):
        time.sleep(0.1)
        bm1('Loop in sleep_func')

class MyClass:
    """ Example in class. """

    def __init__(self):
        self.a = 1
        self.b = 1

    def my_method(self):
        """ Method. """
        self.b = self.a*2
        bm1('method() in MyClass')

bm1 = TimeProbes()
bm2 = TimeProbes(name='Global', unit='ms')

bm1('Start main')
time.sleep(0.1)
bm1('Sleep 0.1 in main')

sleep_func()
bm1('End sleep_func in main')

MyClass().my_method()
bm1('End MyClass().my_method() in main')

bm1.display()
bm2.display()
