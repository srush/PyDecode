---------------
Setup
---------------

Installation
=====================


The easiest way to install PyDecode is through pip.  ::

    $ pip install pydecode


The base functionality of the library requires Numpy and Scipy as well
as Boost. To install Boost on Debian/Ubuntu run::

    $ sudo apt-get install libboost-dev


Optional Dependencies
====================

The core of PyDecode only requires Numpy and Scipy; however the
library includes functions that can integrate with other python libraries.  

* **NetworkX, PyGraphviz, IPython**  (:ref:`display`)
  
  Provides methods for model visualization.

* **PyStruct**  (:ref:`structured`)
  
  Provides methods for training the parameters of a model .

* **PuLP and an LP solver** (:ref:`lp`) 
  
  Provides methods for solving models using general-purpose
  linear-programming solvers.



Running Notebooks
=================

In addition to this documentation, the distribution also include a set
of example tutorials written as IPython notebooks. 

These notebooks can be run locally after installation. Assuming ENV is
the base install directory (for instance using virtualenv), the
notebooks can be run using::

    $ ipython notebook ENV/pydecode/notebooks/
