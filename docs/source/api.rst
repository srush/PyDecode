=====
Python API
=====

Construction
---------------

.. automodule:: pydecode.hyper
.. autosummary::
   :toctree:

   Hypergraph.builder
   GraphBuilder.add_node
   Weights.build
   Constraints.build

Access
---------------

.. automodule:: pydecode.hyper
.. autosummary::
   :toctree:

   Hypergraph.nodes
   Hypergraph.edges
   Hypergraph.label
   Hypergraph.node_label
   Node.edges
   Node.is_terminal
   Edge.head
   Edge.tail

Display
---------------

.. automodule:: pydecode.display
.. autosummary::
   :toctree:

   to_ipython
   to_networkx

Algorithms
---------------

.. automodule:: pydecode.hyper
.. autosummary::
   :toctree:

   best_path
   outside_path
   best_constrained
   Path
   Path.__iter__
   Path.__contains__

Weights
---------------

.. automodule:: pydecode.hyper
.. autosummary::
   :toctree:

   Weights
   Weights.dot


Constraints
---------------

.. automodule:: pydecode.hyper
.. autosummary::
   :toctree:

   Constraints
   Constraints.check
   Constraint.__getitem__
   Constraint.__iter__
   Constraint.constant
   Constraint.__str__

.. automodule:: pydecode.hyper
   :members:
   :special-members:

.. automodule:: pydecode.display
   :members:
