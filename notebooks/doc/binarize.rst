
pydecode.binarize
=================


.. currentmodule:: pydecode                             
.. autofunction:: binarize

Examples
--------


.. code:: python

    import pydecode, pydecode.test
    import numpy as np
.. code:: python

    items = np.arange(10)
    chart = pydecode.ChartBuilder(items)
    chart.init(items[:4])
    chart.set(items[5], [items[:4]], labels=[10])
    graph = chart.finish()
.. code:: python

    pydecode.draw(graph, graph.labeling, vertex_labels=None)



.. image:: binarize_files/binarize_5_0.png



.. code:: python

    new_graph = pydecode.binarize(graph)
.. code:: python

    pydecode.draw(new_graph, new_graph.labeling, vertex_labels=None)



.. image:: binarize_files/binarize_7_0.png



Invariants
----------


.. code:: python

    
Binarizing does not change best path score.

.. code:: python

    @pydecode.test.property()
    def test_binarize(graph, weights, weight_type):
        binary_graph = pydecode.binarize(graph)
        size = np.max(graph.labeling) + 1
        label_weights = pydecode.test.random_weights(weight_type, size)
        
        new_weights = pydecode.transform(graph, label_weights, weight_type=weight_type)
        score1 = pydecode.inside(graph, new_weights, weight_type=weight_type)[graph.root.id]
    
        weights2 = pydecode.transform(binary_graph, label_weights, weight_type=weight_type)
        score2 = pydecode.inside(binary_graph, weights2, weight_type=weight_type)[graph.root.id]
        pydecode.test.assert_almost_equal(score1, score2)
    test_binarize()