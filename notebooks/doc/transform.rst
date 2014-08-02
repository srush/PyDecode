
pydecode.transform
==================


.. currentmodule:: pydecode                             
.. autofunction:: transform    

.. currentmodule:: pydecode                             
.. autofunction:: inverse_transform

Examples
--------


.. code:: python

    import pydecode
.. code:: python

    items = np.arange(11)
    chart = pydecode.ChartBuilder(items)
    chart.init(items[0])
    chart.set(items[5], [[0] for i in range(5)], np.arange(5) % 3)
    chart.set(items[10], [[5] for i in range(5)], np.arange(5) % 3)
    graph = chart.finish()
    pydecode.draw(graph, labels=True)



.. image:: transform_files/transform_5_0.png



.. code:: python

    values = np.array([-100, 0, 100])
    weights = pydecode.transform(graph, values)
    pydecode.draw(graph, weights)



.. image:: transform_files/transform_6_0.png



.. code:: python

    marginals = pydecode.marginals(graph, values * 1.)
    print pydecode.inverse_transform(graph, marginals)

::


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-36-eb6672d0b575> in <module>()
          1 marginals = pydecode.marginals(graph, values * 1.)
    ----> 2 print pydecode.inverse_transform(graph, marginals)
    

    /home/srush/Projects/decoding/python/pydecode/__init__.pyc in inverse_transform(graph, weights, weight_type)
        317       The corresponding label array. Represented as a vector in :math:`\mathbb{S}^{L}`.
        318     """
    --> 319     raise NotImplementedError()
        320     # Slow implementation, make faster.
        321     return _get_type(weight_type).transform_to_labels(weights)


    NameError: global name 'NotImplementedErro' is not defined

