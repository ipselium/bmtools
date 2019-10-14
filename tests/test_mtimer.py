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
# Creation Date : 2019-10-14 - 09:31:14
"""
-----------
mtimer example
-----------
"""

import time
from bmtools import mtimer, format_mtimer


class MtimeExample:
    """ mtimer examples. """

    def __init__(self):
        self.string = 'mtimer example'

    @mtimer(name='with arg')
    def method1(self, string):
        """ Example with argument. """
        time.sleep(0.2)
        print(self.string, string)
        time.sleep(0.2)

    @mtimer
    def method2(self, string):
        """ Example without argument. """
        time.sleep(0.1)
        print(self.string, string)
        time.sleep(0.1)


if __name__ == "__main__":

    mt = MtimeExample()

    for _ in range(2):
        mt.method1('with argument')

    mt.method2('without argument')

    format_mtimer(mt)
