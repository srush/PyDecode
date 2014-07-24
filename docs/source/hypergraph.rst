==========
Hypergraphs
==========

The dynamic 


Structure
==========

Internally, PyDecode uses directed hypergraphs to represent the
structure of a dynamic programming algorithms. It also includes
a low-level interface for working with hypergraphs directly.

.. automodule:: pydecode.potentials
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Hypergraph
   Vertex
   Edge
   Path

.. _algorithms:

Algorithms
==========

The toolkit contains a collection of algorithms for working with hypergraphs.


.. automodule:: pydecode
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   best_path
   inside
   outside
   marginals
   project
   binarize
