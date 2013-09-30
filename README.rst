PyDecode
=========

PyDecode is a Python toolkit for decoding statistical models with a
focus on dynamic programming.  It is being developed with the hope of
(1) simplifying and formalizing the process of writing decoding
algorithms and (2) trying out new experimental algorithms for
constrained decoding problems. The main application for the library is
natural language processing (NLP), however it should be applicable to
other areas that use dynamic programming.

Currently the toolkit is in early development. The code now includes:

Functionality for dynamic programming:

* Construction of a dynamic program (using hypergraphs).
* Graphical output for debugging.
* Finding the best path, inside scores, outside scores, and oracle scores.

Functionality for constrained decoding:

* Hypergraphs with additional side-constraints. 
* Subgradient-style optimization for side-constraints 

Todos:

* Hooks for features and training.
* Formal A* and beam search extensions.
* Integration with an ILP solver.   

While the interface is in Python, the underlying algorithms and data
structures are written in C++ with Cython bindings. Our aim is to keep
the C++ codebase as small as possible.



