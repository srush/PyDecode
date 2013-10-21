
.. code:: python

    import pydecode.hyper as ph
    import pydecode.display as display
    from collections import namedtuple
    
    import pydecode.chart as chart
    import pydecode.semiring as semi
A HMM Tagger Example
--------------------

In this example.

Construction

We begin by constructing the HMM probabilities.

.. code:: python

    # The emission probabilities.
    emission = {'ROOT' : {'ROOT' : 1.0},
                'the' :  {'D': 0.8, 'N': 0.1, 'V': 0.1},
                'dog' :  {'D': 0.1, 'N': 0.8, 'V': 0.1},
                'walked':{'V': 1},
                'in' :   {'D': 1},
                'park' : {'N': 0.1, 'V': 0.9},
                'END' :  {'END' : 1.0}}
    
    # The transition probabilities.
    transition = {'D' :    {'D' : 0.1, 'N' : 0.8, 'V' : 0.1, 'END' : 0},
                  'N' :    {'D' : 0.1, 'N' : 0.1, 'V' : 0.8, 'END' : 0},
                  'V' :    {'D' : 0.4, 'N' : 0.3, 'V' : 0.3, 'END' : 0},
                  'ROOT' : {'D' : 0.4, 'N' : 0.3, 'V' : 0.3}}
    
    # The sentence to be tagged.
    sentence = 'the dog walked in the park'
Next we specify the the index set using namedtuples.

.. code:: python

    class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
        def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
    
    class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
        def __str__(self): return "%s %s"%(self.word, self.tag)
    
    def bigram_weight(bigram):
        return transition[bigram.prevtag][bigram.tag] + emission[bigram.word][bigram.tag] 
Now we write out dynamic program.

.. code:: python

    def viterbi(chart):
        words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
        c.init(Tagged(0, "ROOT", "ROOT"))    
        for i, word in enumerate(words[1:], 1):
            prev_tags = emission[words[i-1]].keys()
            for tag in emission[word].iterkeys():
                c[Tagged(i, word, tag)] = \
                    c.sum([c[key] * c.sr(Bigram(word, tag, prev)) 
                           for prev in prev_tags 
                           for key in [Tagged(i - 1, words[i - 1], prev)] if key in c])
        return c
Now we are ready to build the hypergraph topology itself.

.. code:: python

    # Create a chart using to compute the probability of the sentence.
    c = chart.ChartBuilder(bigram_weight)
    viterbi(c).finish()

.. parsed-literal::

    ROOT -> V
    the V 1.4
    ROOT -> D
    the D 2.2
    ROOT -> N
    the N 1.4
    V -> V
    D -> V
    N -> V
    dog V 2.4000000000000004
    V -> D
    D -> D
    N -> D
    dog D 2.4000000000000004
    V -> N
    D -> N
    N -> N
    dog N 3.8000000000000003
    V -> V
    D -> V
    N -> V
    walked V 5.6000000000000005
    V -> D
    in D 7.0
    D -> V
    the V 7.2
    D -> D
    the D 7.9
    D -> N
    the N 7.9
    V -> V
    D -> V
    N -> V
    park V 9.600000000000001
    V -> N
    D -> N
    N -> N
    park N 8.8
    V -> END
    N -> END
    END END 10.600000000000001




.. parsed-literal::

    10.600000000000001



.. code:: python

    # Create a chart to compute the max paths.
    c = chart.ChartBuilder(bigram_weight, 
                           chart.ViterbiSemiRing)
    viterbi(c).finish()

.. parsed-literal::

    ROOT -> V
    the V 0.4
    ROOT -> D
    the D 1.2000000000000002
    ROOT -> N
    the N 0.4
    V -> V
    D -> V
    N -> V
    dog V 1.4000000000000001
    V -> D
    D -> D
    N -> D
    dog D 1.4000000000000001
    V -> N
    D -> N
    N -> N
    dog N 2.8000000000000003
    V -> V
    D -> V
    N -> V
    walked V 4.6000000000000005
    V -> D
    in D 6.0
    D -> V
    the V 6.2
    D -> D
    the D 6.9
    D -> N
    the N 6.9
    V -> V
    D -> V
    N -> V
    park V 8.600000000000001
    V -> N
    D -> N
    N -> N
    park N 7.800000000000001
    V -> END
    N -> END
    END END 9.600000000000001




.. parsed-literal::

    9.600000000000001



.. code:: python

    c = chart.ChartBuilder(lambda a:a, semi.HypergraphSemiRing, 
                           build_hypergraph = True)
    hypergraph = viterbi(c).finish()

.. parsed-literal::

    ROOT -> V
    make ROOT -> V
    the V <pydecode.semiring.HypergraphSemiRing object at 0x36f4750>
    [([<pydecode.hyper.Node object at 0x35a0a08>], Bigram(word='the', tag='V', prevtag='ROOT'))]
    ROOT -> D
    make ROOT -> D
    the D <pydecode.semiring.HypergraphSemiRing object at 0x36f4810>
    [([<pydecode.hyper.Node object at 0x35a0a08>], Bigram(word='the', tag='D', prevtag='ROOT'))]
    ROOT -> N
    make ROOT -> N
    the N <pydecode.semiring.HypergraphSemiRing object at 0x36f4f50>
    [([<pydecode.hyper.Node object at 0x35a0a08>], Bigram(word='the', tag='N', prevtag='ROOT'))]
    V -> V
    make V -> V
    D -> V
    make D -> V
    N -> V
    make N -> V
    dog V <pydecode.semiring.HypergraphSemiRing object at 0x36f4750>
    [([<pydecode.hyper.Node object at 0x35a0c38>], Bigram(word='dog', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0f30>], Bigram(word='dog', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0ee0>], Bigram(word='dog', tag='V', prevtag='N'))]
    V -> D
    make V -> D
    D -> D
    make D -> D
    N -> D
    make N -> D
    dog D <pydecode.semiring.HypergraphSemiRing object at 0x36f4c90>
    [([<pydecode.hyper.Node object at 0x35a0c38>], Bigram(word='dog', tag='D', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0f30>], Bigram(word='dog', tag='D', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0ee0>], Bigram(word='dog', tag='D', prevtag='N'))]
    V -> N
    make V -> N
    D -> N
    make D -> N
    N -> N
    make N -> N
    dog N <pydecode.semiring.HypergraphSemiRing object at 0x36f4f10>
    [([<pydecode.hyper.Node object at 0x35a0c38>], Bigram(word='dog', tag='N', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0f30>], Bigram(word='dog', tag='N', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0ee0>], Bigram(word='dog', tag='N', prevtag='N'))]
    V -> V
    make V -> V
    D -> V
    make D -> V
    N -> V
    make N -> V
    walked V <pydecode.semiring.HypergraphSemiRing object at 0x36f4c90>
    [([<pydecode.hyper.Node object at 0x35a0fd0>], Bigram(word='walked', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0be8>], Bigram(word='walked', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0a30>], Bigram(word='walked', tag='V', prevtag='N'))]
    V -> D
    make V -> D
    in D <pydecode.semiring.HypergraphSemiRing object at 0x36f45d0>
    [([<pydecode.hyper.Node object at 0x37008f0>], Bigram(word='in', tag='D', prevtag='V'))]
    D -> V
    make D -> V
    the V <pydecode.semiring.HypergraphSemiRing object at 0x36ecb10>
    [([<pydecode.hyper.Node object at 0x3700a08>], Bigram(word='the', tag='V', prevtag='D'))]
    D -> D
    make D -> D
    the D <pydecode.semiring.HypergraphSemiRing object at 0x36ec250>
    [([<pydecode.hyper.Node object at 0x3700a08>], Bigram(word='the', tag='D', prevtag='D'))]
    D -> N
    make D -> N
    the N <pydecode.semiring.HypergraphSemiRing object at 0x36ecb10>
    [([<pydecode.hyper.Node object at 0x3700a08>], Bigram(word='the', tag='N', prevtag='D'))]
    V -> V
    make V -> V
    D -> V
    make D -> V
    N -> V
    make N -> V
    park V <pydecode.semiring.HypergraphSemiRing object at 0x36ec4d0>
    [([<pydecode.hyper.Node object at 0x3700cb0>], Bigram(word='park', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x3700fd0>], Bigram(word='park', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x3700120>], Bigram(word='park', tag='V', prevtag='N'))]
    V -> N
    make V -> N
    D -> N
    make D -> N
    N -> N
    make N -> N
    park N <pydecode.semiring.HypergraphSemiRing object at 0x36ec810>
    [([<pydecode.hyper.Node object at 0x3700cb0>], Bigram(word='park', tag='N', prevtag='V')), ([<pydecode.hyper.Node object at 0x3700fd0>], Bigram(word='park', tag='N', prevtag='D')), ([<pydecode.hyper.Node object at 0x3700120>], Bigram(word='park', tag='N', prevtag='N'))]
    V -> END
    make V -> END
    N -> END
    make N -> END
    END END <pydecode.semiring.HypergraphSemiRing object at 0x36ec490>
    [([<pydecode.hyper.Node object at 0x37004b8>], Bigram(word='END', tag='END', prevtag='V')), ([<pydecode.hyper.Node object at 0x3700620>], Bigram(word='END', tag='END', prevtag='N'))]


Step 3: Construct the weights.

.. code:: python

    weights = ph.Weights(hypergraph).build(bigram_weight)
    
    # Find the best path.
    path = ph.best_path(hypergraph, weights)
    print weights.dot(path)
    
    # Output the path.
    #[hypergraph.label(edge) for edge in path.edges]

.. parsed-literal::

    9.6


.. code:: python

    display.HypergraphPathFormatter(hypergraph, [path]).to_ipython()



.. image:: hmm_files/hmm_14_0.png



We can also use a custom fancier formatter. These attributes are from
graphviz (http://www.graphviz.org/content/attrs)

.. code:: python

    class HMMFormat(display.HypergraphPathFormatter):
        def hypernode_attrs(self, node):
            label = self.hypergraph.node_label(node)
            return {"label": label.tag, "shape": ""}
        def hyperedge_node_attrs(self, edge):
            return {"color": "pink", "shape": "point"}
        def hypernode_subgraph(self, node):
            label = self.hypergraph.node_label(node)
            return [("cluster_" + str(label.position), None)]
        def subgraph_format(self, subgraph):
            return {"label": (["ROOT"] + sentence.split() + ["END"])[int(subgraph.split("_")[1])],
                    "rank" : "same"}
        def graph_attrs(self): return {"rankdir":"RL"}
    
    HMMFormat(hypergraph, [path]).to_ipython()



.. image:: hmm_files/hmm_16_0.png



PyDecode also allows you to add extra constraints to the problem. As an
example we can add constraints to enfore that the tag of "dog" is the
same tag as "park".

.. code:: python

    def cons(tag): return "tag_%s"%tag
    
    def build_constraints(bigram):
        if bigram.word == "dog":
            return [(cons(bigram.tag), 1)]
        elif bigram.word == "park":
            return [(cons(bigram.tag), -1)]
        return []
    
    constraints = \
        ph.Constraints(hypergraph).build( 
                       [(cons(tag), 0) for tag in ["D", "V", "N"]], 
                       build_constraints)
This check fails because the tags do not agree.

.. code:: python

    print "check", constraints.check(path)

.. parsed-literal::

    check [<pydecode.hyper.Constraint object at 0x261dd90>, <pydecode.hyper.Constraint object at 0x36e9190>]


Solve instead using subgradient.

.. code:: python

    gpath, duals = ph.best_constrained(hypergraph, weights, constraints)
.. code:: python

    for d in duals:
        print d.dual, d.constraints

.. parsed-literal::

    9.6 [<pydecode.hyper.Constraint object at 0x261dd90>, <pydecode.hyper.Constraint object at 0x36e9190>]
    8.8 []


.. code:: python

    display.report(duals)


.. image:: hmm_files/hmm_24_0.png


.. code:: python

    import pydecode.lp as lp
    hypergraph_lp = lp.HypergraphLP.make_lp(hypergraph, weights)
    path = hypergraph_lp.solve()
.. code:: python

    # Output the path.
    for edge in gpath.edges:
        print hypergraph.label(edge)

.. parsed-literal::

    ROOT -> D
    D -> N
    N -> V
    V -> D
    D -> D
    D -> N
    N -> END


.. code:: python

    print "check", constraints.check(gpath)
    print "score", weights.dot(gpath)

.. parsed-literal::

    check []
    score 8.8


.. code:: python

    HMMFormat(hypergraph, [path, gpath]).to_ipython()




.. image:: hmm_files/hmm_28_0.png



.. code:: python

    for constraint in constraints:
        print constraint.label

.. parsed-literal::

    tag_D
    tag_V
    tag_N


.. code:: python

    class HMMConstraintFormat(display.HypergraphConstraintFormatter):
        def hypernode_attrs(self, node):
            label = self.hypergraph.node_label(node)
            return {"label": label.tag, "shape": ""}
        def hyperedge_node_attrs(self, edge):
            return {"color": "pink", "shape": "point"}
        def hypernode_subgraph(self, node):
            label = self.hypergraph.node_label(node)
            return [("cluster_" + str(label.position), None)]
        def subgraph_format(self, subgraph):
            return {"label": (["ROOT"] + sentence.split() + ["END"])[int(subgraph.split("_")[1])]}
    
    HMMConstraintFormat(hypergraph, constraints).to_ipython()



.. image:: hmm_files/hmm_30_0.png



Pruning

.. code:: python

    pruned_hypergraph, pruned_weights = ph.prune_hypergraph(hypergraph, weights, 0.8)
.. code:: python

    HMMFormat(pruned_hypergraph, []).to_ipython()



.. image:: hmm_files/hmm_33_0.png



.. code:: python

    very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, weights, 0.9)