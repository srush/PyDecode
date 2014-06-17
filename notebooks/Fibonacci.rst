
Tutorial 1: Fibonacci
=====================


.. code:: python

    import pydecode.hyper as ph
    import pydecode.display as display
    import pydecode.potentials as potentials
.. code:: python

    n = 10
    c = ph.ChartBuilder(item_set=ph.IndexSet(n))
    c[0] = c.init()
    c[1] = [c.merge(0)]
    for i in range(2, n + 1):
        print i
        c[i] = c.merge(i - 1, i - 2)
    hypergraph = c.finish()

::


    ---------------------------------------------------------------------------
    Exception                                 Traceback (most recent call last)

    <ipython-input-6-4d510667e2ac> in <module>()
          4 c[1] = [c.merge(0)]
          5 for i in range(2, n + 1):
    ----> 6     c[i] = c.merge(i - 1, i - 2)
          7 hypergraph = c.finish()


    /home/srush/Projects/decoding/python/pydecode/potentials.so in pydecode.potentials.ChartBuilder.__setitem__ (python/pydecode/potentials.cpp:14751)()


    Exception: Chart already has label


.. code:: python

    c = chart.ChartBuilder(semiring=potentials.LogViterbiValue)
    fibo_dp(c, 10).finish()



.. parsed-literal::

    55.0



.. code:: python

    c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing, 
                           build_hypergraph=True)
    hypergraph = fibo_dp(c, 10).finish()
.. code:: python

    display.HypergraphFormatter(hypergraph, show_hyperedges=False).to_ipython()



.. image:: Fibonacci_files/Fibonacci_5_0.png


