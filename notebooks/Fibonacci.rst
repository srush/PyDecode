
Tutorial 1: Fibonacci
=====================


.. code:: python

    import pydecode.hyper as ph
    import pydecode.chart as chart
    import pydecode.display as display
    import pydecode.potentials as potentials
.. code:: python

    def fibo_dp(c, n):
        c.init(0)
        c[1] = c[0] * c.sr(1)
        for i in range(2, n + 1):
            c[i] = c[i - 1] * c[i - 2]
        return c
.. code:: python

    c = chart.ChartBuilder(semiring=potentials._LogViterbiW)
    fibo_dp(c, 10).finish()

::


    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)

    <ipython-input-12-f1e511f48692> in <module>()
          1 c = chart.ChartBuilder(semiring=potentials._LogViterbiW)
    ----> 2 fibo_dp(c, 10).finish()
    

    /home/srush/Projects/decoding/python/pydecode/chart.py in finish(self)
         53             return self._hypergraph
         54         else:
    ---> 55             return self._chart[self._last].value()
         56 
         57     def value(self, label):


    AttributeError: 'pydecode.potentials._LogViterbiW' object has no attribute 'value'


.. code:: python

    c = chart.ChartBuilder(semiring=chart.HypergraphSemiRing, 
                           build_hypergraph=True)
    hypergraph = fibo_dp(c, 10).finish()
.. code:: python

    display.HypergraphFormatter(hypergraph, show_hyperedges=False).to_ipython()



.. image:: Fibonacci_files/Fibonacci_5_0.png


