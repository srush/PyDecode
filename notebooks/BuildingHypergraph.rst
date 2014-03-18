
Hypergraph Interface
====================


.. code:: python

    import pydecode.hyper as ph
.. code:: python

    hyper1 = ph.Hypergraph()
The code assumes that the hypergraph is immutable. The python interface
enforces this by using a builder pattern. The important function to
remember is add\_node.

-  If there no arguments, then a terminal node is created. Terminal
   nodes must be created first.
-  If it is given an iterable, it create hyperedges to the new node.
   Each element in the iterable is a pair
-  A list of tail nodes for that edge.
-  A label for that edge.


.. code:: python

    with hyper1.builder() as b:
        node_a = b.add_node(label = "a")
        node_b = b.add_node(label = "b")
        node_c = b.add_node(label = "c")
        node_d = b.add_node(label = "d")
        node_e = b.add_node([([node_b, node_c], "First Edge")], label = "e")
        b.add_node([([node_a, node_e], "Second Edge"),
                    ([node_a, node_d], "Third Edge")], label = "f")
Outside of the ``with`` block the hypergraph is considered finished and
no new nodes can be added.

We can also display the hypergraph to see our work.

.. code:: python

    import pydecode.display as display
    display.HypergraphFormatter(hyper1).to_ipython()



.. image:: BuildingHypergraph_files/BuildingHypergraph_7_0.png



After creating the hypergraph we can assign additional property
information. One useful property is to add potentials. We do this by
defining a function to map labels to potentials.

.. code:: python

    def build_potentials(label):
        if "First" in label: return 1
        if "Second" in label: return 5
        if "Third" in label: return 5
        return 0
    potentials = ph.Potentials(hyper1).from_vector([build_potentials(edge.label) 
                                                    for edge in hyper1.edges])
.. code:: python

    for edge in hyper1.edges:
        print edge.label, potentials[edge]

.. parsed-literal::

    First Edge 1.0
    Second Edge 5.0
    Third Edge 5.0


We use the best path.

.. code:: python

    path = ph.best_path(hyper1, potentials)
.. code:: python

    print potentials.dot(path)

.. parsed-literal::

    6.0


.. code:: python

    display.HypergraphFormatter(hyper1).to_ipython()



.. image:: BuildingHypergraph_files/BuildingHypergraph_14_0.png


