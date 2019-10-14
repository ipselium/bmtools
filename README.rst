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
       bm1 = Compare(pow_op, star_op, sqrt_op, unit='ms')
       bm1.run(fargs=(np.random.rand(1000000), ))
       bm1.display()

       # Parametric comparison
       bm2 = Compare(pow_op, star_op, sqrt_op, unit='ms')
       for n in [2**n for n in range(16, 23)]:
           bm2.run(fargs=(np.random.rand(n), ), desc=n)

       bm2.display()
       bm2.bars()

.. code:: console

   +------------+---------------+---------------+-------+
   | Function   |  Description  | Runtime [ms]  | Equal |
   +------------+---------------+---------------+-------+
   | pow_op     |       -       |    2.12607    | True  |
   | star_op    |       -       |    2.16648    | True  |
   | sqrt_op    |       -       |    1.95636    | True  |
   +------------+---------------+---------------+-------+


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

       print(mt.__bm__)



Add time probes to your code
----------------------------



.. |Pypi| image:: https://badge.fury.io/py/bmtools.svg
    :target: https://pypi.org/project/bmtools
    :alt: Pypi Package

.. |Licence| image:: https://img.shields.io/github/license/ipselium/bmtools.svg

.. |Build| image:: https://travis-ci.org/ipselium/bmtools.svg?branch=master
    :target: https://travis-ci.org/ipselium/bmtools
