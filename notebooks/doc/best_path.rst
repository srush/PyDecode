
pydecode.best\_path
===================


.. currentmodule:: pydecode                             
.. autofunction:: best_path    

Example
-------


This examples creates a simple hypergraph with random integer weights,
and highlights the best path in the hypergraph.

.. code:: python

    import pydecode
    import pydecode.test.utils
    import numpy as np
    graph = pydecode.test.utils.simple_hypergraph()
    weights = np.random.randint(10, size=(len(graph.edges)))
    pydecode.draw(graph, weights)



.. image:: best_path_files/best_path_4_0.png



.. code:: python

    path = pydecode.best_path(graph, weights * 1.)
    pydecode.draw(graph, weights, paths=[path])



.. image:: best_path_files/best_path_5_0.png



Invariants
----------


Check that the LogViterbi best path scores at least as high as all other
paths.

.. code:: python

    
    graph = pydecode.test.utils.random_hypergraph()
    weights = np.random.random(len(graph.edges))
    path = pydecode.best_path(graph, weights)
    match = False
    for path2 in pydecode.test.utils.all_paths(graph):
        assert weights.T * path.v >= weights.T * path2.v
        if path == path2:
            match = True
            score = pydecode.test.utils.path_score(path2, weights, pydecode.LogViterbi)
            assert math.fabs(score.value - weights.T * path.v) < 1e-4
    assert match

::


    ---------------------------------------------------------------------------
    AssertionError                            Traceback (most recent call last)

    <ipython-input-10-bda59e0186df> in <module>()
          9         match = True
         10         score = pydecode.test.utils.path_score(path2, weights, pydecode.LogViterbi)
    ---> 11         assert math.fabs(score.value - weights.T * path.v) < 1e-4
         12 assert match


    AssertionError: 

