.. toctree::
   :maxdepth:

=====
Python API
=====

.. _dp:

.. _hypergraph:

Hypergraph
==========

The main data structure used PyDecode is a weighted directed hypergraph, which
is a graphical representation of a dynamic program.

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

The toolkit contains a collection of algorithms for working with weighted hypergraphs, including finding the best path, inside scores, outside score


.. automodule:: pydecode.potentials
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   best_path
   inside
   outside
   compute_marginals

Potentials
============

.. automodule:: pydecode.potentials
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Potentials
   LogViterbiPotentials
   InsidePotentials
   BoolPotentials

Data Structures
===============

.. automodule:: pydecode.potentials
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Marginals
   Chart


Dynamic Programming
===================


PyDecode provides an imperative interface for constructing
dynamic programs. The goal is to make construction as simple
as pseudocode.

.. automodule:: pydecode.chart
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ChartBuilder


Behind the scenes the toolkit will convert the code to a
graph structured representation known as a hypergraph.
