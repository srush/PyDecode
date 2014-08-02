
pydecode.inside
===============


.. currentmodule:: pydecode                             
.. autofunction:: inside    

Example
-------


This examples creates a simple hypergraph with random integer weights, and overlays the inside scores onto the graph for several different weight types.

.. code:: python

    import pydecode
    import pydecode.test.utils
    import numpy as np
    graph = pydecode.test.utils.simple_hypergraph()
    weights = np.random.randint(10, size=(len(graph.edges)))
    pydecode.draw(graph, weights)



.. image:: inside_files/inside_4_0.png



.. code:: python

    inside = pydecode.inside(graph, np.array(weights, dtype=np.double), 
                    weight_type=pydecode.LogViterbi)
    pydecode.draw(graph, weights, inside)



.. image:: inside_files/inside_5_0.png



.. code:: python

    inside = pydecode.inside(graph, np.array(weights, dtype=np.int32), 
                             weight_type=pydecode.Counting)
    pydecode.draw(graph, weights, inside)



.. image:: inside_files/inside_6_0.png



.. code:: python

    inside = pydecode.inside(graph, np.array(weights > 5, dtype=np.int8), 
                    weight_type=pydecode.Boolean)
    pydecode.draw(graph, weights, inside)



.. image:: inside_files/inside_7_0.png



Invariants
----------


Check that the Real inside score of a vertex is the sum of its direct children. 

.. code:: python

    import numpy.testing as test
    
    graph = pydecode.test.utils.random_hypergraph()
    weights = np.random.random(len(graph.edges))
    inside = pydecode.inside(graph, weights, weight_type=pydecode.Real)
    inside2 = np.ones(inside.shape, dtype=np.double)
    
    for vertex in graph.vertices:
        if vertex.is_terminal: 
            continue
        score = 0.0    
        for edge in vertex.edges:
            score += np.product([inside[sub.id] for sub in edge.tail]) * weights[edge.id]
        inside2[vertex.id] = score
    
    test.assert_almost_equal(inside, inside2, 5)