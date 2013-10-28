---------------
Setup
---------------

Installation
=============

Install using pip

    $ pip install pydecode

Running Notebooks
=================

All of the examples in the documentation consist of IPython notebooks available in the notebooks/ directory. 

To get up and running, install IPython and run ::

    $ cd notebooks
    $ ipython notebook --pylab inline 

( You can also run IPython notebook on a research server using ssh tunneling. 

http://wisdomthroughknowledge.blogspot.com/2012/07/accessing-ipython-notebook-remotely.html

and in emacs 

http://tkf.github.io/emacs-ipython-notebook/
)

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


PyStruct
-------------

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
