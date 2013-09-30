=====
API
=====


Construction
---------------

.. automodule:: pydecode.hyper
.. autosummary::
   :toctree: 

   Hypergraph.builder
   GraphBuilder.add_node

Access and Display
---------------

.. automodule:: pydecode.hyper
.. autosummary::
   :toctree: 

   Hypergraph.nodes
   Hypergraph.edges
   Hypergraph.label
   Node.edges
   Edge.tail
   
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
   Path.edges
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
   Constraints.add
   Constraints.check


.. automodule:: pydecode.hyper
   :members:

.. automodule:: pydecode.display
   :members:



