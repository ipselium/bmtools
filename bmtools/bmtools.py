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
# pylint: disable=line-too-long
# pylint: disable=attribute-defined-outside-init
# pylint: disable=unused-argument
"""
------------------
Benchmarking tools
------------------


Classes:

    TimeProbes
    Compare

Functions:

    mtimer(method, name)
    format_mtimer(instance)

"""

import os
import time
import timeit
import inspect
import functools
import itertools
import numpy as np
import matplotlib.pyplot as plt
from bmtools.objects import Timing, Probe, Singleton


__all__ = ["Compare", "TimeProbes", "mtimer", "format_mtimer"]


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
        """Format the results in a table. """

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
    """
    Compares the time it takes to execute functions.

    Parameters
    ----------
    *funcs : Sequence of functions
        Sequence of functions to compare.
    unit : {'sec', 'msec', 'usec', 'nsec'}, optional
        Time measurement unit. 'msec' by default.
    """

    def __init__(self, *funcs, unit='msec'):
        """Constructor. See class doc string."""

        self.funcs = list(dict.fromkeys(funcs))
        self.results = dict()
        self.unit = unit if unit in units.keys() else 'msec'
        self.sort = 'by_description'
        self.hide_nc = True
        self.description = []
        self._outputs = []
        self._rcounter = 0

    @staticmethod
    def parameters(*fargs, **fkwargs):
        """
        Decorator giving a function the __fargs__ and __fkwargs__ attributes.

        This decorator prepares functions for future execution with
        `run_parametric` method passing a combination of fargs/fkwargs to the
        decorated function

        Be aware that decorating a function with Compare.parameters can adds a
        small overhead (several nsec). This is generally not critical, except
        when evaluating the execution time of very fast functions (about ten
        nsec), in particular when this function does not take input arguments.

        Example
        -------

            @Compare.parameters((1,), (2,), b=(3, 4))
            def multiplication(a, b=1):
                return a*b

        This decorator prepares `multiplication` to be executed with the following
        combinations of args/kwargs:
            - args=(1, ) and kwargs={'b':3}
            - args=(1, ) and kwargs={'b':4}
            - args=(2, ) and kwargs={'b':3}
            - args=(2, ) and kwargs={'b':4}
        """
        def decorator(func):

            func.__fargs__ = fargs
            func.__fkwargs__ = fkwargs

            @functools.wraps(func)
            def both(*args, **kwargs):
                return func(*args, **kwargs)

            @functools.wraps(func)
            def none():
                return func()

            if fargs or fkwargs:
                return both

            return none

        return decorator

    @staticmethod
    def compile_args(funcs):
        """
        Return a list of possible combinations of args and kwargs from the __fargs__
        and __fkwargs__ attributes of a sequence of functions `funcs`. All the
        functions in this sequence must have been decorated with
        `Compare.parameters`.

        Parameters
        ----------
        funcs : list or tuple
            Sequence of functions

        Raises
        ------
        Raises ValueError if at least one of the functions in `funcs` does not
        have __fargs__/__fkwargs__ attributes.
        """

        tmp = []
        arg_set = []

        for func in funcs:
            if not hasattr(func, '__fargs__') or not hasattr(func, '__fkwargs__'):
                raise ValueError(f"function '{func.__name__}' must be decorated with 'parameter()'")
            kw = (dict(zip(func.__fkwargs__, x)) for x in itertools.product(*func.__fkwargs__.values()))
            if func.__fargs__:
                tmp.append(list(itertools.product(func.__fargs__, kw)))
            else:
                tmp.append(list(itertools.product((tuple(), ), kw)))

        for item in itertools.chain.from_iterable(tmp):
            if item not in arg_set:
                arg_set.append(item)

        # Handle no args/kwargs case
        if tmp.count([]) > 0:
            arg_set.insert(0, [tuple(), dict()])

        return arg_set

    @staticmethod
    def is_configured(func, args, kwargs):
        """Check if func has been configured (with `parameters`) to run with args/kwargs."""

        flag_arg = True
        flag_kwarg = True

        # Check args
        if args not in func.__fargs__:
            flag_arg = False

        # Check kwargs
        if kwargs:
            for key, value in kwargs.items():
                if value not in func.__fkwargs__.get(key, ['__none__']):
                    flag_kwarg = False
        elif not kwargs and  func.__fkwargs__:
            flag_kwarg = False

        # Particular cases: no args
        if args == func.__fargs__:
            flag_arg = True

        return True if flag_arg and flag_kwarg else False

    @staticmethod
    def _make_description(args, kwargs, sep=', '):
        """Format run description. """
        if not args:
            desc = '()' + sep
        else:
            args = str(args)[1:-1]
            desc = args + sep if not args.endswith(',') else args + ' '
        desc += sep.join([f'{key}={value}' for key, value in kwargs.items()]) if kwargs else '{}'
        return desc

    def run_parametric(self, N=None, runs=7, display=False):
        """
        Use __fargs__/__fkwargs__ defined with `parameter` decorator to make a parametric study.

        Parameters
        ----------
        N: int, optional
            Execute N times the functions. Default to None, that means N will be automatically set.
        runs: int, optional
            Number of repeat. Default to 7. Must be at least 5.
        """

        arglist = self.compile_args(self.funcs)

        for args, kwargs in arglist:
            desc = self._make_description(args, kwargs)
            self.description.append(desc)
            self._outputs = []
            for func in self.funcs:
                if self.is_configured(func, args, kwargs):
                    self._run(func, fargs=args, fkwargs=kwargs, desc=desc, N=N, runs=runs)
                else:
                    self._update(func.__name__, [0], None, desc)

        if display:
            self.display()

    def run_single(self, fargs=None, fkwargs=None, desc='--', N=None, runs=7, display=False):
        """
        Compare execution time and outputs of the functions in `funcs`.

        They are all executed with the same args and kwargs defined as by the
        `fargs` and `fkwargs` arguments passed to this method.

        The execution time is measured using `Timer` class from the standard `timeit` module.
        See `timeit` doc for more informations on the timing procedure.

        Parameters
        ----------
        fargs: tuple, optional
            Arguments to pass to function. Default to None.
        kwargs: dict, optional
            Keyword arguments to pass to func. Default to None.
        desc: str, optional
            Description of the run. Default to '--'.
        N: int, optional
            Execute N times the functions. Default to None, that means N will be automatically set.
        runs: int, optional
            Number of repeat. Default to 7. Must be at least 5.
        """

        if not fargs:
            fargs = tuple()

        if not fkwargs:
            fkwargs = dict()

        self.description.append(str(desc))
        self._outputs = []

        for func in self.funcs:
            self._run(func, fargs=fargs, fkwargs=fkwargs, desc=desc, N=N, runs=runs)

        if display:
            self.display()

    def _run(self, func, fargs, fkwargs, desc='--', N=None, runs=7):
        """Run the comparison. Wrap timeit to work with functions."""

        self._outputs.append(func(*fargs, **fkwargs))

        partial = functools.partial(func, *fargs, **fkwargs)

        timer = timeit.Timer(partial)
        runs = 5 if runs < 5 else runs
        if not N:
            N, _ = timer.autorange()

        times = timer.repeat(repeat=runs, number=N)
        times = [time/N for time in times]
        self._update(func.__name__, times, self._output, desc)

    def _update(self, name, times, output, desc):
        """Update (or create) Timing objects."""

        _rtime = sum(times)/len(times)
        _best = min(times)
        _worst = max(times)
        _std = np.std(times)
        _output = output if output else 'NC.'

        if self.results.get(name):
            self.results[name].time.append(_rtime)
            self.results[name].best.append(_best)
            self.results[name].worst.append(_worst)
            self.results[name].std.append(_std)
            self.results[name].output.append(_output)
            self.results[name].desc.append(desc)

        else:
            self.results[name] = Timing(name=name, time=[_rtime],
                                        best=[_best], worst=[_worst], std=[_std],
                                        output=[_output], desc=[str(desc)])

    @property
    def rcounter(self):
        """Reference counter for output checking."""
        self._rcounter += 1
        return self._rcounter

    @property
    def _output(self):
        """Format information about outputs."""
        if len(self._outputs) == 1:
            return f'R{self.rcounter}'

        if self.is_equal(self._outputs)[-1]:
            return f'==R{self._rcounter}'

        return f'!=R{self._rcounter}'

    @staticmethod
    def is_equal(seq):
        """Check if all elements of seq are equal."""
        types = [type(obj) for obj in seq]

        if len(set(types)) > 1:
            raise ValueError('To be compared, all elements must be of the same type')

        if all([isinstance(o, np.ndarray) for o in seq]):
            return [np.all(x == seq[0]) for x in seq]

        return [x == seq[0] for x in seq]

    def reset(self):
        """Reset all results."""
        self.results = dict()
        self.description = []

    @staticmethod
    def _autolabel(ax, rects):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        Reference
        ---------
        Function from matplotlib documentation.
        See https://matplotlib.org/gallery/api/barchart.html
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    def _set_figure(self, fig, ax, xlabel, ylabel):
        """Set some figure parameters."""

        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_xticklabels(self.description, rotation=45)
        ax.legend()
        fig.tight_layout()
        plt.show()

    def plot(self, xlabel='Function arguments', log=False, relative=False):
        """
        Display results as plots.

        Parameters
        ----------
        xlabel: str, optional
            Define xlabel. Default to 'Function arguments'.
        log: bool. optional
            Plot in log scale. Default to False.
        relative: bool, optional
            If True, displays relative results. Then, reference is the 1st function passed to instance.
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
        """
        Display results as bar chart.

        See help(Compare.plot) for more informations.
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

    def display(self, sort='by_description', hide_nc=True):
        """
        Display results as table.

        Parameters
        ----------
        sort: {'by_description', 'by_function}, optional
            Sort by function or description. Default to 'by_description'.
        """
        self.sort = sort
        self.hide_nc = hide_nc
        print(self)

    @property
    def digits(self):
        """Returns number of digits to display. """
        return 1 if self.unit == 'nsec' else 2 if self.unit in ['usec', 'msec'] else 4

    def __str__(self):
        """Display results as table."""

        if self.sort not in ['by_function', 'by_description']:
            raise ValueError("sort must be 'by_function' or 'by_description'")

        # Define max width for some columns
        nd = len(self.description)
        nf = len(self.funcs)
        maxfc = max([len(key) for key in self.results] + [len(' Function ')])
        maxdc = max([len(desc) for desc in self.description] + [len(' Description ')])

        # Define the templates
        rows = '| {fc:<{maxfc}} | {dc:^{maxdc}} | {rt:^14} | {std:^14} | {out:^5} |\n'
        hline = f"+{'':-<{maxfc+2}}+{'':-^{maxdc+2}}+{'':-^16}+{'':-^16}+{'':-^7}+\n"

        # Setup the results
        results = []
        for idx in range(nd):
            for fc, tg in self.results.items():
                results.append(rows.format(fc=fc, maxfc=maxfc,
                                           dc=tg.desc[idx], maxdc=maxdc,
                                           rt=round(tg.best[idx]/units[self.unit], self.digits) if tg.best[idx] != 0 else 'NC.',
                                           std=round(tg.std[idx]/units[self.unit], self.digits) if tg.std[idx] != 0 else 'NC.',
                                           out=tg.output[idx]))

        if self.sort == 'by_function':
            results = list(itertools.chain.from_iterable([results[i::nf] for i in range(nf)]))
            results = itertools.chain(*zip(*[results[i::nd] for i in range(nd)], [hline]*nf))
        elif self.sort == 'by_description':
            results = itertools.chain(*zip(*[results[i::nf] for i in range(nf)], [hline]*nd))

        if self.hide_nc:
            results = [line for line in results if 'NC.' not in line]

        # Start creating the table
        table = hline
        table += rows.format(fc='Function', maxfc=maxfc,
                             dc='Description', maxdc=maxdc,
                             rt=f'Runtime [{self.unit}]',
                             std=f'Std [{self.unit}]', out='Equal')
        table += hline

        return table + ''.join(results)

    def __repr__(self):
        """Display results in a table"""
        return self.__str__()


def mtimer(method=None, *, name=None):
    """ Measure execution time of instance methods of a class.

    Create a __bm__ instance attribute that refers to a dictionary containing
    execution times of the decorated instance methods.

    Note that static or class method cannot be timed with this decorator.

    Parameters
    ----------
    name: str, optional
        If provided, name is the key of the __bm__ dictionary where times are
        saved for the current method. If not provided, name is method.__name__.

    Example
    -------

        class Example:

            @mtimer                      # Can be @mtimer(name='key')
            def instance_method(self):
                pass
    """
    def decorator(method):

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            nonlocal name
            if not name:
                name = method.__name__
            if not hasattr(args[0], '__bm__'):
                args[0].__bm__ = dict()
            start = time.perf_counter()
            output = method(*args, **kwargs)
            if args[0].__bm__.get(name):
                args[0].__bm__[name].append(time.perf_counter() - start)
            else:
                args[0].__bm__[name] = [time.perf_counter() - start]
            return output
        return wrapper

    if method:
        return decorator(method)

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
                             atime=f'Avg. time [{unit}]',
                             rtime=f'Runtime [{unit}]',
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
        return None

    return output
