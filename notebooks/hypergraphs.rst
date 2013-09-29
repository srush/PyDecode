
Simple Hypergraph Example
=========================


.. code:: python

    import pydecode.hyper as hyper
    import pydecode.display as display
    import networkx as nx 
    import matplotlib.pyplot as plt 
    from IPython.display import Image
.. code:: python

    hyp = hyper.Hypergraph()
    with hyp.builder() as b:
         n1 = b.add_node("first", terminal=True)
         n2 = b.add_node("second")
         b.add_edge(n2, [n1], label = "Edge")
Draw the graph

.. code:: python

    G = display.to_networkx(hyp)
    d = nx.drawing.to_agraph(G)
    d.layout("dot")
    d.draw("/tmp/tmp.png")
    Image(filename ="/tmp/tmp.png")



.. image:: hypergraphs_files/hypergraphs_4_0.png


