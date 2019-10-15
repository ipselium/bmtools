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
# Creation Date : 2019-10-15 - 16:11:09
"""
-----------
objects in use in mbtools
-----------
"""

import gc
import dataclasses


class Singleton(type):
    """ Singleton metaclass. """

    _instances = {}

    def __call__(cls, name="TimeProbes", **kwargs):
        if name not in cls._instances:
            cls._instances[name] = super(Singleton, cls).__call__(name, **kwargs)
        return cls._instances[name]


@dataclasses.dataclass
class Timing:
    """ Timing dataclass. """
    name: str
    time: list
    best: list
    worst: list
    std: list
    output: bool
    desc: str


@dataclasses.dataclass
class Probe:
    """ Create a time probe. """

    name: str
    time: list
    line: int
    file: str
    function: str

    @property
    def mtime(self):
        """ Return mean time. """
        return sum(self.time)/len(self.time)

    @property
    def ttime(self):
        """ Return total time. """
        return sum(self.time)

    @property
    def location(self):
        """ Return location of timer. """
        return f'{self.file}:{self.line}'


class GarbageCollector:
    """ Context manager for garbage collection. """

    def __init__(self):
        self.initial_state = gc.isenabled()
        self.current_state = self.initial_state

    def __enter__(self):
        self.initial_state = gc.isenabled()
        gc.disable()
        self.current_state = gc.isenabled()

    def __exit__(self, *args):
        if self.initial_state:
            gc.enable()
        self.current_state = gc.isenabled()
