
.. code:: python

    import pydecode.nlp.phrase_based as pb
    import pydecode.chart as chart 
    import pydecode.test.utils as utils
    import pydecode.hyper as ph
.. code:: python

    n = 3
.. code:: python

    c = chart.ChartBuilder(lambda a:a, chart.HypergraphSemiRing, 
                               build_hypergraph = True)
    phrases = [pb.Phrase(*p) for p  in [((0,2), [0]), ((1,2), [0,1]), ((0,2), [1]), ((1,1), [1])]]
    words = range(3)
    phrases = pb.make_phrase_table(phrases)
    
    pb.phrase_lattice(n, phrases, words, c)
    lat = c.finish()
.. code:: python

    import pydecode.display as display
    display.HypergraphFormatter(lat).to_ipython()



.. image:: phrase_based_files/phrase_based_3_0.png



.. code:: python

    w = utils.random_log_viterbi_potentials(lat)
    ins = ph.inside(lat, w)
    out = ph.outside(lat, w, ins)
.. code:: python

    groups = [(node.label.num_source if node.label != "END" else n+1)b for node in lat.nodes ]
    num_groups = max(groups) + 1
    limits = [100] * num_groups
.. code:: python

    def make_constraints(edge):
        b = ph.Bitset()
        if edge.label is None:
            return b    
        for i in range(edge.label.source_span[0], edge.label.source_span[1]):
            b[i] = 1
        return b
    
    constraints = ph.BinaryVectorPotentials(lat)\
        .from_vector([make_constraints(edge) for edge in lat.edges] )
.. code:: python

    print groups
    print num_groups
    print limits

.. parsed-literal::

    [0, 1, 2, 2, 4]
    5
    [100, 100, 100, 100, 100]


.. code:: python

    chart = ph.beam_search_BinaryVector(lat, w, constraints, out, -10000, groups, limits)

::


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-32-7663c832af82> in <module>()
    ----> 1 chart = ph.beam_search_BinaryVector(lat, w, constraints, out, -10000, groups, limits)
    

    /home/srush/Projects/decoding/python/pydecode/potentials.so in pydecode.potentials.beam_search_BinaryVector (python/pydecode/potentials.cpp:15870)()


    TypeError: beam_search_BinaryVector() takes exactly 8 positional arguments (7 given)


.. code:: python

    for n in lat.nodes:
        print chart[n]

.. parsed-literal::

    [(<pydecode.potentials.Bitset object at 0x493b5b0>, 0.0, 2.4306261765868884)]
    [(<pydecode.potentials.Bitset object at 0x493b5b0>, 0.7574388452061992, 1.673187331380689)]
    [(<pydecode.potentials.Bitset object at 0x493b5b0>, 0.3133211265699205, 0.901839078839646)]
    [(<pydecode.potentials.Bitset object at 0x493b5b0>, 0.03291616576722978, 0.7459879956839593)]
    [(<pydecode.potentials.Bitset object at 0x493b5b0>, 1.2151602054095665, 0.0), (<pydecode.potentials.Bitset object at 0x493b978>, 0.7789041614511891, 0.0)]

