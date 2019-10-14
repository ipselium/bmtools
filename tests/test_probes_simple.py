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
# Creation Date : 2019-10-14 - 09:50:57
"""
-----------
TimeProbes simple example
-----------
"""

import time
from bmtools import TimeProbes

bm = TimeProbes()        # Create our probes
time.sleep(0.1)
bm('example')            # Create a probe named 'example'
time.sleep(0.2)
bm()                     # Create a probe without name

with bm as my_context:  # Use probe as context manager.
    time.sleep(0.8)     # my_context will be the name of the probe

bm.display()            # Display times measured at probe locations
