
pydecode.marginals
==================


.. currentmodule:: pydecode                             
.. autofunction:: marginals 

Example
-------


.. code:: python

    import pydecode
    import pydecode.test.utils
    import numpy as np
Invariants
----------


.. code:: python

    import numpy.testing as test
    import pydecode.test.utils as test_utils
    
    graph, weights, weight_type = test_utils.random_setup()
    marginals = pydecode.marginals(graph, weights, weight_type=weight_type)
Marginals represent to the sum of all paths through each edge.

.. code:: python

    marginals2 = [weight_type.Value.zero()] * len(graph.edges)
    for path in test_utils.all_paths(graph):
        score = test_utils.path_score(path, weights, weight_type)
        for edge in path:
            marginals2[edge.id] += score
    for i in range(len(marginals2)):
        marginals2[i] = marginals2[i].value
    
    test.assert_almost_equal(marginals, np.array(marginals2), 5)