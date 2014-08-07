
pydecode.marginals
==================


.. currentmodule:: pydecode                             
.. autofunction:: marginals 

Example
-------


.. code:: python

    import pydecode, pydecode.test
    import numpy as np
Invariants
----------


.. code:: python

    
Marginals represent to the sum of all paths through each edge.

.. code:: python

    @pydecode.test.property()
    def test_all_marginals(graph, weights, weight_type):
        marginals = pydecode.marginals(graph, weights, weight_type=weight_type)
        marginals2 = [weight_type.Value.zero()] * len(graph.edges)
        for path in pydecode.test.all_paths(graph):
            score = pydecode.score(path, weights, weight_type)
            for edge in path:
                marginals2[edge.id] += score
        
        marginals2 = np.array([m.value for m in marginals2])
        pydecode.test.assert_almost_equal(marginals, 
                                          marginals2, 5)
    test_all_marginals()