
pydecode.outside
================


.. currentmodule:: pydecode                             
.. autofunction:: outside    

Example
-------


.. code:: python

    This examples creates a simple hypergraph with random integer weights, and overlays the outside scores onto the graph for several different weight types.
.. code:: python

    import pydecode
    import pydecode.test.utils
    import numpy as np
    graph = pydecode.test.utils.simple_hypergraph()
    weights = np.random.randint(10, size=(len(graph.edges)))
    pydecode.draw(graph, weights)



.. image:: outside_files/outside_4_0.png



.. code:: python

    def show_outside(weights, weight_type):
        inside = pydecode.inside(graph, weights, weight_type=weight_type)
        outside = pydecode.outside(graph, weights, inside, weight_type=weight_type)
        return pydecode.draw(graph, weights, outside)

.. code:: python

    show_outside(np.array(weights, dtype=np.double), pydecode.LogViterbi)



.. image:: outside_files/outside_6_0.png



.. code:: python

    show_outside(np.array(weights, dtype=np.int32), pydecode.Counting)



.. image:: outside_files/outside_7_0.png



.. code:: python

    show_outside(np.array(weights > 5, dtype=np.int8), pydecode.Boolean)



.. image:: outside_files/outside_8_0.png



Invariants
----------



