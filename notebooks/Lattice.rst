
.. code:: python

    
.. code:: python

    n = 15
.. code:: python

    import pydecode.hyper as ph
    import pydecode.chart as chart
    import pydecode.constraints as constraints
    import numpy as np
.. code:: python

    print [ [j for j in range(3)] for i in range(3) ]

.. parsed-literal::

    [[0, 1, 2], [0, 1, 2], [0, 1, 2]]


.. code:: python

    h = ph.make_lattice(3, 3, [ [j for j in range(3)] for i in range(3) ])
.. code:: python

    class Form(d.HypergraphFormatter):
        def hyperedge_node_attrs(self, n):
            return {}

.. code:: python

    len(h.nodes)
    for n in h.nodes:
        print n.label

.. parsed-literal::

    0 0
    1 0
    1 1
    1 2
    1 0
    1 1
    1 2
    2 0
    2 1
    2 2
    3 0
    3 1
    3 2
    4 0


.. code:: python

    import pydecode.display as d
    a =Form(h).to_image("/tmp/fig.png")
.. code:: python

    a.to_image
.. code:: python

    def make_lattice(c):
        for k in range(n):
            c.init((0,k))
        for i in range(1,n):
            for j in range(n-1):
                c[(i,j)] = c.sum([c[i-1,k] for k in range((n -1) if i > 1 else 1) ])
        return c
.. code:: python

    c = chart.ChartBuilder(lambda a: a, semiring=chart.HypergraphSemiRing, build_hypergraph=True)
    h = make_lattice(c).finish()
.. code:: python

    # import pydecode.display as display
    # display.HypergraphFormatter(h).to_ipython()
.. code:: python

    p = np.random.random(len(h.edges))
    w = ph.LogViterbiPotentials(h).from_array(p)
.. code:: python

    ins = ph.inside(h, w)
    out = ph.outside(h, w, ins)
.. code:: python

    def build_constraints(edge):
        b = ph.Bitset()
        i, j = edge.head.label
        b[j] = 1
        return b
    cons3 = ph.BinaryVectorPotentials(h).from_vector(
               [build_constraints(edge) for edge in h.edges])
.. code:: python

    def build_constraints(edge):
        b = []
        i, j = edge.head.label
        b.append((j, 1))
        return b
    cons1 = constraints.Constraints(h, [(i,-1) for i in range(0,n-1)])\
                .from_vector([build_constraints(edge) for edge in h.edges])
.. code:: python

    # k = [0, 1, 2, 1, 5, 2] 
    # def build_constraints(edge):
    #     b = [-1] * 26
    #     i, j = edge.head.label
    #     b[k[i]] = j
    #     return b
    # cons2 = ph.AlphabetPotentials(h).from_vector([build_constraints(edge) 
    #                                              for edge in h.edges])
.. code:: python

    # for edge in h.edges:
    #     print cons[edge]
.. code:: python

    # groups = [node.label[0] for node in h.nodes]
    # num_groups = max(groups) + 1
    # beam_chart = ph.beam_search_Alphabet(h, w, cons1, out, -10000, [node.label[0] for node in h.nodes], [1000] * num_groups, num_groups)
.. code:: python

    groups = [node.label[0] for node in h.nodes]
    num_groups = max(groups) + 1
    beam_chart = ph.beam_search_BinaryVector(h, w, cons3, out, -10000, [node.label[0] for node in h.nodes], [1000] * num_groups, num_groups)
.. code:: python

    from collections import defaultdict
    d = defaultdict(list)
    for i, g in enumerate(groups):
        d[g].append(i)
.. code:: python

    def show_alphabet(alpha):
        for i, a in enumerate(alpha):
            if a != -1:
                print chr(ord("A")+ i) + "->"+  chr(ord("A")+ a),
        print
.. code:: python

    # for g in d:
    #     print "Group " + str(g)
    #     for node in [h.nodes[i] for i in d[g]] :
    #         print node.label
    #         for (hyp, score, future) in beam_chart[node]:
    #             print "\t", score + future, 
    #             # show_alphabet(hyp)
    #             print " ".join(str(hyp[i]) for i in range(10))
    #         print 
.. code:: python

    import pydecode.lp as lp
    l = lp.HypergraphLP.make_lp(h, w, integral=True)
    l.add_constraints(cons1)
    l.solve()
    print l.objective
    for node in l.path.nodes:
        print node.label

.. parsed-literal::

    12.6431069308
    (0, 0)
    (1, 9)
    (2, 10)
    (3, 4)
    (4, 7)
    (5, 12)
    (6, 2)
    (7, 8)
    (8, 6)
    (9, 3)
    (10, 5)
    (11, 11)
    (12, 1)
    (13, 0)
    (14, 13)

