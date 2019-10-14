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
# Creation Date : 2019-10-10 - 12:07:48
"""
-----------
Benchmarking tools
-----------
"""

import os
import time
import inspect
import functools
import itertools
import dataclasses
import numpy as np
import matplotlib.pyplot as plt


def kb_to_mb(kb):
    return f'{kb/2**20:.2f} Mo'


class Singleton(type):
    """ Singleton metaclass. """

    _instances = {}

    def __call__(cls, name="TimeProbes", **kwargs):
        if name not in cls._instances:
            cls._instances[name] = super(Singleton, cls).__call__(name, **kwargs)
        return cls._instances[name]


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


class TimeProbes(metaclass=Singleton):
    """ Measure time between probes. Largely inspired by Bench-it.

    Parameters
    ----------
    unit: str
        Define whether results are displayed in seconds ('s') or milliseconds
        ('ms'). Default to 'ms'

    Reference
    ---------
    https://pypi.org/project/bench-it/
    """

    def __init__(self, name='TimeProbes', unit='s'):

        self.name = name
        self.unit = unit
        self._unit = 1e3 if unit == 'ms' else 1
        self.reset()

    def reset(self):
        """ Reset results, but not timer ! """

        self._probes = dict()
        self._pidx = 0
        self._info = None

        self._cminfo = None
        self._cmtime = None
        self._cmname = None
        self._cmidx = 0

        self._itime = time.perf_counter()
        self._ltime = 0

    @property
    def _etime(self):
        """ Return current time. """
        return (time.perf_counter() - self._itime)*self._unit

    @property
    def _filename(self):
        fn = self._info.filename if 'ipython' not in self._info.filename else 'IPython'
        return os.path.basename(fn)

    @property
    def _function(self):
        return self._info.function if '<module>' not in self._info.function else '--'

    @property
    def _cmfilename(self):
        fn = self._cminfo.filename if 'ipython' not in self._cminfo.filename else 'IPython'
        return os.path.basename(fn)

    @property
    def _cmfunction(self):
        return self._cminfo.function if '<module>' not in self._cminfo.function else '--'

    @property
    def _probe_no(self):
        self._pidx += 1
        return self._pidx

    @property
    def _context_no(self):
        self._cmidx += 1
        return self._cmidx

    def _ispresent(self, line):
        """ Test if a proble has already been declared in line. """
        for probe in self._probes.values():
            if probe.line == line:
                return probe.name
        return None

    def __call__(self, probe_name=None):

        self._info = inspect.getframeinfo(inspect.currentframe().f_back)
        name = self._ispresent(self._info.lineno)
        if name:
            self._probes[name].time.append(self._etime - self._ltime)

        else:
            if probe_name is None:
                probe_name = f'Probe {self._probe_no}'

            self._probes[probe_name] = Probe(name=probe_name,
                                             time=[self._etime - self._ltime],
                                             file=self._filename,
                                             line=self._info.lineno,
                                             function=self._function)

        self._ltime = self._etime

    def __enter__(self):

        self._cminfo = inspect.getframeinfo(inspect.currentframe().f_back)
        self._cmtime = self._etime
        ctxt = self._cminfo.code_context[0]
        if 'as' in ctxt:
            self._cmname = ctxt.split('#')[0].split('as')[-1].split(':')[0].strip()
        else:
            self._cmname = "Context {self._context_no}"

    def __exit__(self, *args):

        self._probes[self._cmname] = Probe(name=self._cmname,
                                           time=[self._etime - self._cmtime],
                                           file=self._cmfilename,
                                           line=self._cminfo.lineno,
                                           function=self._cmfunction)
        self._ltime = self._etime

    def display(self):
        """ Display results in a formated table. """
        print(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):

        ttime = time.perf_counter() - self._itime
        maxmk = max([len(p.name) for p in self._probes.values()] + [len(' Makers ')])
        maxlc = max([len(p.location) for p in self._probes.values()] + [len(' File:line ')])
        maxfc = max([len(p.function) for p in self._probes.values()] + [len(' Function ')])

        rows = '| {mk:<{maxmk}} | {lc:^{maxlc}} | {fc:^{maxfc}} |'
        rows += ' {at:^13} | {rt:^13} | {pc:^10} |\n'

        line = f"+ {'':-<{maxmk}} + {'':-^{maxlc}} + {'':-^{maxfc}} +"
        line += f" {'':-^13} + {'':-^13} + {'':-^10} +\n"

        table = '+' + (len(line)-3)*'-' + '+\n'
        table += f"|{self.name:^{len(line)-3}}|\n"
        table += line
        table += rows.format(mk='Makers', maxmk=maxmk,
                             lc='File:line', maxlc=maxlc,
                             fc='Function', maxfc=maxfc,
                             at=f'Avg time [{self.unit}]',
                             rt=f'Runtime [{self.unit}]',
                             pc='Percent')
        table += line
        for p in self._probes.values():
            table += rows.format(mk=p.name, maxmk=maxmk,
                                 lc=p.location, maxlc=maxlc,
                                 fc=p.function, maxfc=maxfc,
                                 at=round(p.mtime, 5),
                                 rt=round(p.ttime, 5),
                                 pc=round(p.ttime/ttime*100, 1))
        table += line

        return table


class Compare:
    """ Compares the time it takes to execute functions

    Parameters
    ----------
    *funcs: functions
        Functions to compare.
    unit: str = 's' or 'ms'
        Time measurement unit. Can be second (s) or millisecond (ms). Millisecond by default.
    """

    def __init__(self, *funcs, unit='s'):

        self.funcs = funcs
        self.fnames = [f.__name__ for f in self.funcs]
        self.results = dict()
        self.description = []
        self.unit = unit
        self._unit = 1e3 if unit == 'ms' else 1

    def run(self, fargs, desc='-', N=10000, max_time=1):
        """ Run the benchmark.

        Parameters
        ----------
        N: int. Default to 10000.
            Number of repetition of each func.
        max_time: float. Default to 1.
            Maximum runtime for each func.
        """

        output = []
        self.description.append(str(desc))

        for f in self.funcs:

            ti = time.perf_counter()

            for i in range(1, N+1):
                tmp = f(*fargs)
                if self._get_time(ti) > max_time*self._unit:
                    break

            output.append(tmp)
            self._update(f.__name__, self._get_time(ti)/i, self.is_equal(output)[-1])

    @staticmethod
    def is_equal(seq):
        """ Check if all elements of seq are equal. """

        types = [type(obj) for obj in seq]

        if len(set(types)) > 1:
            raise ValueError('To be compared, all elements must be of the same type')

        if all([isinstance(o, np.ndarray) for o in seq]):
            return [np.all(x == seq[0]) for x in seq]

        return [x == seq[0] for x in seq]

    def _get_time(self, ti):
        return (time.perf_counter() - ti)*self._unit

    def _update(self, name, time, output):

        if self.results.get(name):
            self.results[name]['time'].append(time)
            self.results[name]['output'].append(output)
        else:
            self.results[name] = dict()
            self.results[name]['time'] = [time]
            self.results[name]['output'] = [output]

    def _set_figure(self, fig, ax, xlabel, ylabel):

        ax.set_xticklabels(self.description, rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        plt.show()

    def plot(self, xlabel='Function arguments', log=False, relative=False):
        """ Display results as plots.

        Parameters
        ----------
        xlabel: str. Default to 'Function arguments'
            Define xlabel.
        log: bool. Default to False.
            Plot in log scale.
        relative: bool.
            If True, displays relative results. Reference is the 1st function passed to instance.
        """

        plot = plt.semilogy if log else plt.plot
        xlabel = str(xlabel)
        ylabel = r'$t_f / t_{ref}$' if relative else r'$t_f$ [{}]'.format(self.unit)
        if relative:
            reference = np.array(self.results[self.funcs[0].__name__]['time'])
        else:
            reference = 1

        fig, ax = plt.subplots(figsize=(9, 4))
        for f in self.funcs:
            plot(self.description,
                 np.array(self.results[f.__name__]['time'])/reference,
                 marker='o', markersize=3, label=f.__name__)

        ax.grid()
        self._set_figure(fig, ax, xlabel, ylabel)

    def bars(self, xlabel='Function arguments', log=False, relative=False):
        """ Displays results as bar chart.

        Parameters
        ----------
        xlabel: str. Default to 'Function arguments'
            Define xlabel.
        log: bool. Default to False.
            Plot in log scale.
        relative: bool.
            If True, displays relative results. Reference is the 1st function passed to instance.
        """

        xlabel = str(xlabel)
        ylabel = r'$t_f / t_{ref}$' if relative else r'$t_f$ [{}]'.format(self.unit)
        if relative:
            reference = np.array(self.results[self.funcs[0].__name__]['time'])
        else:
            reference = 1

        width = 0.80/len(self.funcs)
        locs = np.arange(len(self.description))  # the label locations
        offset = np.linspace(-1, 1, len(self.funcs))*width*(len(self.funcs)-1)/2
        rects = []

        fig, ax = plt.subplots(figsize=(9, 4))
        for i, f in enumerate(self.funcs):
            rects.append(ax.bar(locs + offset[i],
                                np.array(self.results[f.__name__]['time'])/reference,
                                width, label=f.__name__, log=log))

        ax.set_xticks(locs)

        for rect in rects:
            self._autolabel(ax, rect)

        self._set_figure(fig, ax, xlabel, ylabel)

    @staticmethod
    def _autolabel(ax, rects):
        """
        From matplotlib.org : Attach a text label above each bar in *rects*, displaying its height.
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    def reset(self):
        """ Reset all results. """

        self.results = dict()
        self.description = []

    def display(self):
        """ Display results as table. """

        print(self)

    def __str__(self):

        maxfc = max([len(key) for key in self.results] + [len(' Function ')])
        maxdc = max([len(desc) for desc in self.description] + [len(' Description ')])

        rows = '| {fc:<{maxfc}} | {dc:^{maxdc}} | {rt:^13} | {out:^5} |\n'
        line = f"+{'':-<{maxfc+2}}+{'':-^{maxdc+2}}+{'':-^15}+{'':-^7}+\n"


        table = line
        table += rows.format(fc='Function', maxfc=maxfc,
                             dc='Description', maxdc=maxdc,
                             rt=f'Runtime [{self.unit}]',
                             out='Equal')
        table += line
        for idx, dc in enumerate(self.description):
            for fc, results in self.results.items():
                table += rows.format(fc=fc, maxfc=maxfc,
                                     dc=dc, maxdc=maxdc,
                                     rt=round(results['time'][idx], 5),
                                     out='True' if results['output'][idx] else 'False')
            table += line

        return table

    def __repr__(self):
        return self.__str__()


def mtimer(func=None, *, name=None):
    """ Measure execution time of instance methods.

    Create a __bm__ instance attribute that refers to a dictionary containing
    execution times of instance methods.

    Note that static or class method cannot be timed with this decorator.

    Parameters
    ----------
    name: str
        If provided, name is the key of the __bm__ dictionary where times are saved.

    Example
    -------
        @mtimer                      # Can be @mtimer(name='key')
        def instance_method(self):
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal name
            if not name:
                name = func.__name__
            if not hasattr(args[0], '__bm__'):
                args[0].__bm__ = dict()
            start = time.perf_counter()
            output = func(*args, **kwargs)
            if args[0].__bm__.get(name):
                args[0].__bm__[name].append(time.perf_counter() - start)
            else:
                args[0].__bm__[name] = [time.perf_counter() - start]
            return output
        return wrapper

    if func:
        return decorator(func)

    return decorator


def format_mtimer(instance, table=True, unit='s', precision=5, display=True):
    """ Create table from mtimer results.

    Parameters
    ----------
    instance:
        Class instance from which to create table. Must have __bm__ attribute.
    table: bool.
        If True, create table, else basically formats the results. Default to True.
    unit: str
        Define whether results are displayed in seconds ('s') or milliseconds
        ('ms'). Default to 'ms'.
    precision: int
        Precision of the results. Default to 5
    display: bool
        Define whether or not table is displayed
    """

    if not hasattr(instance, '__bm__'):
        raise ValueError('Instance does not have __mb__ attribute')

    _unit = 1e3 if unit == 'ms' else 1
    kmax = max([len(k) for k in instance.__bm__])
    pmax = len(str(int(max(itertools.chain.from_iterable(instance.__bm__.values()))*_unit)))+1
    tmax = max(len(f' Avg. time ({unit})'), pmax + precision) if table else pmax + precision

    output = ''
    if table:
        tmp = '| {key:<{kmax}} | {calls:^5} | {atime:^{tmax}} | {rtime:^{tmax}} |\n'
        line = f"+{'':-^{kmax+2}}+{'':-^7}+{'':-^{tmax+2}}+{'':-^{tmax+2}}+\n"
        output += line
        output += tmp.format(key='Method', calls='Calls',
                             atime=f'Avg. time ({unit})',
                             rtime=f'Runtime ({unit})',
                             kmax=kmax, tmax=tmax)
        output += line
    else:
        tmp = "\t+ {key:<{kmax}} : {rtime:>{tmax}} {unit}. ({calls} calls of {atime} {unit}.)\n"

    for key, values in instance.__bm__.items():
        output += tmp.format(key=key, calls=len(values),
                             atime=round(sum(values)/len(values)*_unit, precision),
                             rtime=round(sum(values)*_unit, precision),
                             tmax=tmax, kmax=kmax, unit=unit)

    if table:
        output += line

    if display:
        print(output)
        return

    return output
