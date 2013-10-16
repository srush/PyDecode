Parsing
=============


PyDecode is a dynamic programming toolkit being developed to help researchers studying natural language processing.

The interface and visualization code is written Python, the core algorithms are written in C++.
For full description of the project see the documentation_.

.. _documentation: http://pydecode.readthedocs.org/



.. image:: _images/parsing_9_0.png
   :width: 500 px
   :align: center


Motivation
-------------

We built this toolkit because:

* **Dynamic programming is hard** to get right and painful to debug.
* **Run-time efficiency** is crucial for many NLP tasks.
* **Extensions** to dynamic programming often require extensive extra programming.

This presentation_ discusses the background for this work.

.. _presentation: https://github.com/srush/PyDecode/raw/master/writing/slides/slides.pdf


Features
-------------

Currently the toolkit is in early development but should be ready to be used.

Dynamic Programming
======================

* Construction of dynamic programming algorithms.
* Graphical output for debugging.
* Finding the best path, inside scores, outside scores, and oracle scores.
* Pruning based on max-marginals.
* Integration with an LP solver.

Constrained Dynamic Programming
===============================

* Hypergraphs with additional constraints.
* Subgradient-style optimization for constraints.
* ILP optimization with constraints.

Future Additions
===============================

* Hooks for features and training.
* Formal A* and beam search extensions.
* K-best algorithms.

While the interface is in Python, the underlying algorithms and data
structures are written in C++ with Cython bindings. Our aim is to keep
the C++ codebase as small as possible.


Tutorial and Galleries
----------------------

.. hlist::
   :columns: 2

   * documentation_
   * tutorial_
   * gallery_
   * api_

.. _gallery: http://pydecode.readthedocs.org/en/latest/notebooks/tutorial.html
.. _tutorial: http://pydecode.readthedocs.org/en/latest/notebooks/tutorial.html
.. _api: http://pydecode.readthedocs.org/en/latest/notebooks/api.html
