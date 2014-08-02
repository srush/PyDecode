
pydecode.Hypergraph
===================


.. note::
   This section gives a formal overview of the use of ``Hypergraph``. For a series of tutorials and practical examples see :doc:`../index`.

.. currentmodule:: pydecode                                            
.. autoclass:: Hypergraph

.. currentmodule:: pydecode                                            
.. autoclass:: Vertex

.. currentmodule:: pydecode                                            
.. autoclass:: Edge

.. currentmodule:: pydecode                                            
.. autoclass:: Path

Examples
--------


.. code:: python

    import pydecode
    import pydecode.test.utils
    hypergraph = pydecode.test.utils.simple_hypergraph()
.. code:: python

    for vertex in hypergraph.vertices:
        print vertex.id, vertex.is_terminal
        for edge in vertex.edges:
            print "\t", edge.id, edge.label

.. parsed-literal::

    0 True
    1 True
    2 True
    3 True
    4 False
    	0 -1
    	1 -1
    5 False
    	2 -1

