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
import numpy as np
import matplotlib.pyplot as plt
from bmtools.objects import Timing, Probe, GarbageCollector, Singleton

units = {"nsec": 1e-9, "usec": 1e-6, "msec": 1e-3, "sec": 1.0}


class TimeProbes(metaclass=Singleton):
    """ Measure time between probes. Largely inspired by Bench-it.

    Parameters
    ----------
    unit: str = 'sec', 'msec', 'usec', 'nsec'
        Time measurement unit. Millisecond by default.

    Reference
    ---------
    https://pypi.org/project/bench-it/
    """

    def __init__(self, name='TimeProbes', unit='msec'):

        self.name = name
        self.unit = unit if unit in units.keys() else 'msec'
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
    def digits(self):
        """ Returns number of digits to display. """
        return 2 if self.unit == 'nsec' else 5

    @property
    def _etime(self):
        """ Return current time. """
        return (time.perf_counter() - self._itime)/units[self.unit]

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
        rows += ' {at:^16} | {rt:^16} | {pc:^10} |\n'

        line = f"+ {'':-<{maxmk}} + {'':-^{maxlc}} + {'':-^{maxfc}} +"
        line += f" {'':-^16} + {'':-^16} + {'':-^10} +\n"

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
                                 at=round(p.mtime, self.digits),
                                 rt=round(p.ttime, self.digits),
                                 pc=round(p.ttime/ttime*100, 1))
        table += line

        return table


class Compare:
    """ Compares the time it takes to execute functions

    Parameters
    ----------
    *funcs: functions
        Functions to compare.
    unit: str = 'sec', 'msec', 'usec', 'nsec'
        Time measurement unit. Millisecond by default.
    """

    def __init__(self, *funcs, unit='msec'):

        self.funcs = tuple(set(funcs))
        self.fnames = [f.__name__ for f in self.funcs]
        self.results = dict()
        self.unit = unit if unit in units.keys() else 'msec'
        self.gc = GarbageCollector()
        self.order = 'by_description'
        self.description = []

    def run(self, fargs=(), desc='--', N=None, runs=7):
        """ Run the benchmark.

        Parameters
        ----------
        fargs: tuple.
            Arguments of the function. Default to ().
        desc: str:
            Description of the run.
        N: int.
            Execute N times the functions. Default to None.
        runs: int.
            Number of repeat. Default to 7. Must be at least 5.
        """

        runs = 5 if runs < 5 else runs
        outputs = []
        times = []
        self.description.append(str(desc))

        for func in self.funcs:

            with self.gc:
                tmp = func(*fargs)
            outputs.append(tmp)

            if N is None:
                for N in [10**i for i in range(0, 10)]:
                    runtime = self.timeit(func=func, fargs=fargs, N=N)
                    if runtime >= 0.2:
                        times.append(runtime/N)
                        break

            # Correction : empty loop
            te = self._passit(N=N)/N

            for _ in itertools.repeat(None, runs-1):
                with self.gc:
                    times.append(self.timeit(func=func, fargs=fargs, N=N)/N - te)

            self._update(func.__name__, times, self.is_equal(outputs)[-1], desc)

    @staticmethod
    def timeit(func, fargs=(), N=1_000_000):
        """ Timing method. """

        ti = time.perf_counter()
        for _ in itertools.repeat(None, N):
            func(*fargs)
        te = time.perf_counter()
        return te - ti

    @staticmethod
    def _passit(N=1_000_000):

        ti = time.perf_counter()
        for _ in itertools.repeat(None, N):
            pass
        te = time.perf_counter()
        return te - ti


    @staticmethod
    def is_equal(seq):
        """ Check if all elements of seq are equal. """

        types = [type(obj) for obj in seq]

        if len(set(types)) > 1:
            raise ValueError('To be compared, all elements must be of the same type')

        if all([isinstance(o, np.ndarray) for o in seq]):
            return [np.all(x == seq[0]) for x in seq]

        return [x == seq[0] for x in seq]

    def reset(self):
        """ Reset all results. """
        self.results = dict()
        self.description = []

    def display(self, order='by_description'):
        """ Display results as table.

        Parameter
        ---------
        order: str.
            Order by function or description. Can be 'by_description' or 'by_function'.
            Default to 'by_description'.
        """
        self.order = order
        print(self)

    def _update(self, name, times, output, desc):

        if self.results.get(name):
            self.results[name].time.append(sum(times)/len(times))
            self.results[name].best.append(min(times))
            self.results[name].worst.append(max(times))
            self.results[name].std.append(np.std(times))
            self.results[name].output.append(output)
            self.results[name].desc.append(desc)

        else:
            self.results[name] = Timing(name=name,
                                        time=[sum(times)/len(times)],
                                        best=[min(times)], worst=[max(times)], std=[np.std(times)],
                                        output=[output], desc=[str(desc)])

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
            reference = np.array(self.results[self.funcs[0].__name__].time)
        else:
            reference = units[self.unit]

        fig, ax = plt.subplots(figsize=(9, 4))
        for f in self.funcs:
            plot(self.results[f.__name__].desc,
                 np.array(self.results[f.__name__].time)/reference,
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
            reference = np.array(self.results[self.funcs[0].__name__].time)
        else:
            reference = units[self.unit]

        width = 0.80/len(self.funcs)
        locs = np.arange(len(self.description))  # the label locations
        offset = np.linspace(-1, 1, len(self.funcs))*width*(len(self.funcs)-1)/2
        rects = []

        fig, ax = plt.subplots(figsize=(9, 4))
        for i, f in enumerate(self.funcs):
            rects.append(ax.bar(locs + offset[i],
                                np.array(self.results[f.__name__].time)/reference,
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

    @property
    def digits(self):
        """ Returns number of digits to display. """
        return 2 if self.unit == 'nsec' else 5

    def __str__(self):

        nd = len(self.description)
        nf = len(self.funcs)
        maxfc = max([len(key) for key in self.results] + [len(' Function ')])
        maxdc = max([len(desc) for desc in self.description] + [len(' Description ')])

        rows = '| {fc:<{maxfc}} | {dc:^{maxdc}} | {rt:^14} | {std:^14} | {out:^5} |\n'
        line = f"+{'':-<{maxfc+2}}+{'':-^{maxdc+2}}+{'':-^16}+{'':-^16}+{'':-^7}+\n"


        table = line
        table += rows.format(fc='Function', maxfc=maxfc,
                             dc='Description', maxdc=maxdc,
                             rt=f'Runtime [{self.unit}]',
                             std=f'Std [{self.unit}]', out='Equal')
        table += line

        results = []

        for idx in range(nd):
            for fc, tg in self.results.items():
                results.append(rows.format(fc=fc, maxfc=maxfc,
                                           dc=tg.desc[idx], maxdc=maxdc,
                                           rt=round(tg.time[idx]/units[self.unit], self.digits),
                                           std=round(tg.std[idx]/units[self.unit], self.digits),
                                           out='True' if tg.output[idx] else False))

        if self.order == 'by_function':
            results = list(itertools.chain.from_iterable([results[i::nf] for i in range(nf)]))
            results = itertools.chain(*zip(*[results[i::nd] for i in range(nd)], [line]*nf))
        elif self.order == 'by_description':
            results = itertools.chain(*zip(*[results[i::nf] for i in range(nf)], [line]*nd))
        else:
            raise ValueError("order must be 'by_function' or 'by_description'")

        return table + ''.join(results)

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


def format_mtimer(instance, table=True, unit='msec', precision=5, display=True):
    """ Create table from mtimer results.

    Parameters
    ----------
    instance:
        Class instance from which to create table. Must have __bm__ attribute.
    table: bool.
        If True, create table, else basically formats the results. Default to True.
    unit: str = 'sec', 'msec', 'usec', 'nsec'
        Time measurement unit. Millisecond by default.
    precision: int
        Precision of the results. Default to 5
    display: bool
        Define whether or not table is displayed
    """

    if not hasattr(instance, '__bm__'):
        raise ValueError('Instance does not have __mb__ attribute')

    _unit = units[unit]
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
