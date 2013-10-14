PyDecode is a dynamic programming toolkit. It is being developed as a
research tool for natural language processing. The interface and
visualization code is written Python, the core algorithms are written in
C++. For full description of the project see the
[documentation](http://pydecode.readthedocs.org/).

Motivation -----

We built this toolkit because:

-   Dynamic programming is hard to get right and painful to debug.
-   Run-time efficiency is crucial for many NLP tasks.
-   Extensions to dynamic programming often require extensive extra
    programming.

This
[presentation](https://github.com/srush/PyDecode/raw/master/writing/slides/slides.pdf)
discusses the background for this work.

Features
========

Currently the toolkit is in early development but should be ready to be
used.

Dynamic programming:

-   Construction of a dynamic programming algorithms.
-   Graphical output for debugging.
-   Finding the best path, inside scores, outside scores, and oracle
    scores.
-   Pruning based on max-marginals.
-   Integration with an LP solver.

Constrained dynamic programming:

-   Hypergraphs with additional constraints.
-   Subgradient-style optimization for constraints.
-   ILP optimization with constraints.

Future additions:

-   Hooks for features and training.
-   Formal A\* and beam search extensions.
-   K-best algorithms.

While the interface is in Python, the underlying algorithms and data
structures are written in C++ with Cython bindings. Our aim is to keep
the C++ codebase as small as possible.

Tutorial and Galleries
======================

To get started check out the
[tutorial](http://pydecode.readthedocs.org/en/latest/notebooks/tutorial.html)
and a
[gallery](http://pydecode.readthedocs.org/en/latest/notebooks/tutorial.html)
of examples.
