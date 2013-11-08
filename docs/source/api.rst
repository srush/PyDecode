.. toctree::
   :maxdepth:

=====
Python API
=====

.. _dp:

Dynamic Programming
===================


PyDecode provides an imperative interface for constructing
dynamic programs. The goal is to make construction as simple
as pseudocode.

See these example algorithms.

.. automodule:: pydecode.chart
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ChartBuilder


Behind the scenes the toolkit will convert the code to a
graph structured representation known as a hypergraph.

.. _hypergraph:

Hypergraph
==========

The main data structure used PyDecode is a weighted directed hypergraph, which
is a graphical representation of a dynamic program.
The algorithms and tools in the rest package make heavy use of this data structure.

.. automodule:: pydecode.hyper
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Hypergraph
   Path

.. _algorithms:


Algorithms
==========

The toolkit contains a collection of algorithms for working with weighted hypergraphs, including finding the best path, inside scores, outside score


.. automodule:: pydecode.hyper
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   best_path
   inside
   outside
   compute_marginals
   Marginals
   Chart

Potentials
============

.. automodule:: pydecode.hyper
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LogViterbiPotentials
   InsidePotentials
   ViterbiPotentials
   BoolPotentials
   SparseVectorPotentials
