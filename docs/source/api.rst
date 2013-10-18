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
   SemiRing


Behind the scenes the toolkit will convert the code to a
graph structured representation known as a hypergraph.

.. _hypergraph:

Hypergraph
==========

The main data structure used PyDecode is a weighted directed hypergraph, which
is a graphical representation of a dynamic program.
The algorithms and tools in the rest package make heavy use of this data structure.

The notation used to describe hypergraphs is based on this :ref:`introduction to hypergraphs <theory>`.

.. automodule:: pydecode.hyper
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Hypergraph
   Weights


.. _algorithms:

Algorithms
==========

The toolkit contains a collection of algorithms for working with weighted hypergraphs, including finding the best path, outside scores, pruning, and max-marginals.

Planned algorithms include k-best paths, marginal computations, and general semi-ring algorithms.

.. automodule:: pydecode.hyper
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Path
   Chart
   MaxMarginals

.. autosummary::
   :toctree: generated/

   best_path
   outside_path
   prune_hypergraph
   compute_max_marginals



.. _constraints:

Constraints
=============

Many algorithms in natural language processing, such as translation decoding, can be represented as constrained dynamic programming problems. These can be described as hypergraphs with additional constraints on hyperedges. The notation used to describe constrained hypergraphs is based on this :ref:`introduction to constraints <intro_constraints>`

Constrained hypergraphs can be solved in PyDecode either using a subgradient-based solver (:py:function:`best_constrained`) or by using an :ref:`(integer) linear programming solver<lp>`.

.. automodule:: pydecode.hyper
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Constraints
   Constraint

.. autosummary::
   :toctree: generated/

   best_constrained


.. _lp:

Linear Programming
==================

Standard hypergraph search problems can also be solved by using linear programming,
and constrained hypergraph search problems can be solved using integer linear programming. The notation used to describe constrained hypergraphs is based on this :ref:` conversion to linear program <intro_lp>`.

PyDecode uses PuLP_ to generate these (integer) linear programs.

.. automodule:: pydecode.lp
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HypergraphLP


.. _structured:

Structured Prediction
=======================

Structured prediction is a class of machine learning problem that aims
to train a model to predict the best structure out of an often
exponentially large set, for instance the best dependency parse for a
sentence. For many problems, both the training stage and testing stage
require solving problems involving dynamic programming or constrained
dynamic programming.

PyStruct_ is a general structured prediction framework that implements
many useful training algorithms. The :py:module:`pydecode.model` module
wraps the StructuredModel class from PyStruct. This allows the user to
train the parameters of a model by specifying a :py:class:`HypergraphModelBuilder`.

.. automodule:: pydecode.model
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    HypergraphModel
    HypergraphModelBuilder


.. _display:


Visualization
==========

One major benefit of creating a hypergraph representation is that it allows for
easy visualization. The display package converts a hypergraph to a NetworkX_ graph
and then uses PyGraphViz_ to render an image. The style of the graph can be easily customized by inheriting from :py:class:`HypergraphFormatter`.

.. automodule:: pydecode.display
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   HypergraphFormatter
   to_image


   .. :toctree: generated/
   .. :template: class.rst

.. _PuLP: http://pythonhosted.org/PuLP/
.. _NetworkX: http://networkx.github.io/documentation/latest/
.. _PyGraphViz: http://pygraphviz.github.io/
.. _PyStruct: http://pystruct.github.io/
