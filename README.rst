Introducing bmtools
===================

|Pypi| |Build| |Licence|


.. image:: https://github.com/ipselium/bmtools/blob/master/docs/compare.png


**bmtools** provides some tools dedicated to benchmarking.


Requirements
------------

:python: >= 3.7
:matplotlib: >= 3.0
:numpy: >= 1.1

Installation
------------

Clone the github repo and

.. code:: console

    $ python setup.py install

or install via Pypi

.. code:: console

    $ pip install bmtools


Compare execution times
-----------------------

Benchmarking functions execution can be done with **`Compare`** class as follows:

.. code-block:: python

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
       bm1 = Compare(pow_op, star_op, sqrt_op)
       bm1.run_single(fargs=(np.random.rand(1000000), ))
       bm1.display()

       # Parametric comparison
       bm2 = Compare(pow_op, star_op, sqrt_op, unit='ms')
       for n in [2**n for n in range(16, 23)]:
           bm2.run_single(fargs=(np.random.rand(n), ), desc=n)

       bm2.display()
       bm2.bars()

.. code:: console

   +------------+---------------+----------------+----------------+-------+
   | Function   |  Description  | Runtime [msec] |   Std [msec]   | Equal |
   +------------+---------------+----------------+----------------+-------+
   | pow_op     |      --       |    1.56256     |    0.00798     |  R1   |
   | star_op    |      --       |    1.55787     |    0.00752     | ==R1  |
   | sqrt_op    |      --       |    1.58628     |    0.04214     | ==R1  |
   +------------+---------------+----------------+----------------+-------+

   (...)

`Compare` provides three ways to display results:

   * As a simple plot with the `Compare.plot()` method
   * As a bar chart with the `Compare.bar()` method
   * As a text table with the `Compare.display()` method


`Compare` also provides the `parameters` decorator to specify a list of
args/kwarg that have to be passed to a function for parametric study. The the
`Compare.run_parametric` method performs the comparison:

.. code-block:: python

   from bmtools import Compare

   @Compare.parameters((1, 2,), (2, 3, ), x=(1, 10))
   def op1(a, b, x=1):
       return a*x + b

   @Compare.parameters((1, 2,), (2, 3,), x=(1, 10))
   def op2(a, b, x=1):
       return a*x + b

   if __name__ == "__main__":
       bm3 = Compare(op1, op2, unit='nsec')
       bm3.run_parametric()
       bm3.display()

.. code:: console

   +------------+---------------+----------------+----------------+-------+
   | Function   |  Description  | Runtime [nsec] |   Std [nsec]   | Equal |
   +------------+---------------+----------------+----------------+-------+
   | op1        |   1, 2, x=1   |     359.8      |      12.0      |  R1   |
   | op2        |   1, 2, x=1   |     354.5      |      8.4       | ==R1  |
   +------------+---------------+----------------+----------------+-------+
   | op1        |  1, 2, x=10   |     352.5      |      6.1       |  R2   |
   | op2        |  1, 2, x=10   |     351.2      |      8.6       | ==R2  |
   +------------+---------------+----------------+----------------+-------+

Time instance methods
---------------------

The **`mtimer`** decorator can be used to time instance methods as follows:

.. code-block:: python

   import time
   from bmtools import mtimer


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


Add time probes to your code
----------------------------

The **`TimeProbes`** class provide a way to time blocks of code. Note that this
class is largely inspired by Bench-it.

.. code-block:: python

   bm = TimeProbes()        # Create our probes
   time.sleep(0.1)
   bm('example')            # Create a probe named 'example'
   time.sleep(0.2)
   bm()                     # Create a probe without name

   with bm as my_context:  # Use probe as context manager.
       time.sleep(0.8)      # my_context will be the name of the probe

   bm.display()            # Display times measured at probe locations


.. code:: console


   +-------------------------------------------------------------------------------------------------------+
   |                                              TimeProbes                                               |
   + ---------- + ------------------------ + ---------- + ---------------- + ---------------- + ---------- +
   | Makers     |        File:line         |  Function  | Avg time [msec]  |  Runtime [msec]  |  Percent   |
   + ---------- + ------------------------ + ---------- + ---------------- + ---------------- + ---------- +
   | example    | test_probes_simple.py:33 |     --     |    167.75452     |    167.75452     |  14334.3   |
   | Probe 1    | test_probes_simple.py:35 |     --     |    201.12324     |    201.12324     |  17185.6   |
   | my_context | test_probes_simple.py:37 |     --     |    800.91822     |    800.91822     |  68436.9   |
   + ---------- + ------------------------ + ---------- + ---------------- + ---------------- + ---------- +


References
----------

The **`TimeProbes`** class is largely inpired by Bench-it:

https://pypi.org/project/bench-it/



.. |Pypi| image:: https://badge.fury.io/py/bmtools.svg
    :target: https://pypi.org/project/bmtools
    :alt: Pypi Package

.. |Licence| image:: https://img.shields.io/github/license/ipselium/bmtools.svg

.. |Build| image:: https://travis-ci.org/ipselium/bmtools.svg?branch=master
    :target: https://travis-ci.org/ipselium/bmtools
