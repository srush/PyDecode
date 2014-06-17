.. toctree::
   :maxdepth: 2

==========
API
==========

.. _dp:


Construction
===================

.. automodule:: pydecode.potentials
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ChartBuilder
   IndexSet

.. _hypergraph:

Hypergraph
==========

PyDecode uses directed hypergraphs to represent the
structure of a dynamic programming algorithm.

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


.. automodule:: pydecode.potentials
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

.. Potentials
.. ============

.. There are several types of potentials implemented.

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    LogViterbi
..    Inside
..    Bool


.. Potentials
.. ============

.. Users can specify potential vectors that are associated with
.. each edge of the hypergraph.

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    Potentials

.. There are several types of potentials implemented.

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    LogViterbiPotentials
..    InsidePotentials
..    BoolPotentials


.. Using the corresponding data structures


.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/

..    Chart
..    Marginals
..    BackPointers

.. Graph Builder
.. ------------

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    GraphBuilder
