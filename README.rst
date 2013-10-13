PyDecode
=========

PyDecode is a pragmatic toolkit for dynamic programming with a focus on natural language processing (NLP).  
The interface is in Python, the core code is in C++. 

We built this toolkit because:

* Dynamic programming is hard to get right and painful to debug.
* Run-time efficiency is crucial for many NLP tasks.
* Extensions to dynamic programming often require extensive extra programming.

Currently the toolkit is in early development. The code now includes:

Dynamic programming:

* Construction of a dynamic program (using hypergraphs).
* Graphical output for debugging.
* Finding the best path, inside scores, outside scores, and oracle scores.
* Pruning.
* Integration with an LP solver.

Constrained dynamic programming:

* Hypergraphs with additional side-constraints. 
* Subgradient-style optimization for side-constraints 

Future additions:

* Hooks for features and training.
* Formal A* and beam search extensions.
* K-best algorithms.

While the interface is in Python, the underlying algorithms and data
structures are written in C++ with Cython bindings. Our aim is to keep
the C++ codebase as small as possible.

It is being developed with the hope of
(1) simplifying and formalizing the process of writing decoding
algorithms and (2) trying out new experimental algorithms for
constrained decoding problems. The main application for the library is
natural language processing (NLP), however it should be applicable to
other areas that use dynamic programming.
