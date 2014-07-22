---------------
Setup
---------------

Installation
=============

Install using pip ::

    $ pip install pydecode

Running Notebooks
=================

All of the documentation examples are written as IPython notebooks. They are available in the notebooks/ directory.

To modify examples locally, install IPython and run ::

    $ cd notebooks
    $ ipython notebook --pylab inline

.. _tunneling: http://wisdomthroughknowledge.blogspot.com/2012/07/accessing-ipython-notebook-remotely.html
.. _emacs: http://tkf.github.io/emacs-ipython-notebook/

Dependencies
=====================

Requires numpy and scipy.


Optional Dependencies
=====================

Networkx and PyGraphviz
-------------

Provides features for displaying hypergraphs.

Pandas and matplotlib
-------------

Provides abilty to display performance reports and charts.

IPython
-------------

Provides features for working with graphs in IPython and IPython notebook.


Scikit Learn and PyStruct
-------------------------

Provides methods for training the parameters of a hypergraph model.


PuLP and an LP solver (such as glpk or gurobi)
-------------

Provides a module for converting dynamic programming problems to linear programs which can be solved or exported through PuLP.



Nose, py.test, and PyZMQ
------------

Needed to run tests. (including tests in IPython notebooks.)

NLTK
----------

Used for constructing NLP examples.
