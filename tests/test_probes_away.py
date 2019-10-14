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
# Creation Date : 2019-10-11 - 11:38:51
"""
-----------
Multi probes example
-----------
"""
from bmtools import TimeProbes

bmx1 = TimeProbes()
bmx2 = TimeProbes('Global')

bmx1('Probe bmx1 in module')
bmx2('Probe bmx2 in module')
