
A Constrained HMM Example
-------------------------


.. code:: python

    import pydecode.hyper as ph
    import pydecode.display as display
    from collections import namedtuple
    import pydecode.chart as chart
We begin by constructing the HMM probabilities.

.. code:: python

    # The emission probabilities.
    emission = {'ROOT' :  {'ROOT' : 1.0},
                'the' :  {'D': 0.8, 'N': 0.1, 'V': 0.1},
                'dog' :  {'D': 0.1, 'N': 0.8, 'V': 0.1},
                'walked' : {'V': 1},
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
.. code:: python

    def build_weights(bigram):
        return transition[bigram.prevtag][bigram.tag] + emission[bigram.word][bigram.tag] 
.. code:: python

    def viterbi(chart):
        words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
        c.init(Tagged(0, "ROOT", "ROOT"))    
        for i, word in enumerate(words[1:], 1):
            prev_tags = emission[words[i - 1]].keys()
            for tag in emission[word].iterkeys():
                c[Tagged(i, word, tag)] = \
                    c.sum([c[key] * c.sr(Bigram(word, tag, prev)) 
                           for prev in prev_tags 
                           for key in [Tagged(i - 1, words[i - 1], prev)] if key in c])
        return c
Now we are ready to build the hypergraph topology itself.

.. code:: python

    c = chart.ChartBuilder(lambda a: build_weights(Bigram(*a)))
    the_chart = viterbi(c)
    the_chart.finish()

.. parsed-literal::

    the V 1.4
    the D 2.2
    the N 1.4
    dog V 2.4000000000000004
    dog D 2.4000000000000004
    dog N 3.8000000000000003
    walked V 5.6000000000000005
    in D 7.0
    the V 7.2
    the D 7.9
    the N 7.9
    park V 9.600000000000001
    park N 8.8
    END END 10.600000000000001




.. parsed-literal::

    10.600000000000001



.. code:: python

    c = chart.ChartBuilder(lambda a: build_weights(Bigram(*a)), 
                           chart.ViterbiSemiRing)
    the_chart = viterbi(c)
    the_chart.finish()

.. parsed-literal::

    the V 0.4
    the D 1.2000000000000002
    the N 0.4
    dog V 1.4000000000000001
    dog D 1.4000000000000001
    dog N 2.8000000000000003
    walked V 4.6000000000000005
    in D 6.0
    the V 6.2
    the D 6.9
    the N 6.9
    park V 8.600000000000001
    park N 7.800000000000001
    END END 9.600000000000001




.. parsed-literal::

    9.600000000000001



.. code:: python

    c = chart.ChartBuilder(lambda a: Bigram(*a), 
                           chart.HypergraphSemiRing, 
                           build_hypergraph = True)
    the_chart = viterbi(c)
    hypergraph = the_chart.finish()

.. parsed-literal::

    start
    the V <pydecode.semiring.HypergraphSemiRing instance at 0x5379d40>
    [([<pydecode.hyper.Node object at 0x539a288>], Bigram(word='the', tag='V', prevtag='ROOT'))]
    the D <pydecode.semiring.HypergraphSemiRing instance at 0x53812d8>
    [([<pydecode.hyper.Node object at 0x539a288>], Bigram(word='the', tag='D', prevtag='ROOT'))]
    the N <pydecode.semiring.HypergraphSemiRing instance at 0x53813b0>
    [([<pydecode.hyper.Node object at 0x539a288>], Bigram(word='the', tag='N', prevtag='ROOT'))]
    dog V <pydecode.semiring.HypergraphSemiRing instance at 0x5381128>
    [([<pydecode.hyper.Node object at 0x539a350>], Bigram(word='dog', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x539acd8>], Bigram(word='dog', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x539a2b0>], Bigram(word='dog', tag='V', prevtag='N'))]
    dog D <pydecode.semiring.HypergraphSemiRing instance at 0x5381b00>
    [([<pydecode.hyper.Node object at 0x539a350>], Bigram(word='dog', tag='D', prevtag='V')), ([<pydecode.hyper.Node object at 0x539acd8>], Bigram(word='dog', tag='D', prevtag='D')), ([<pydecode.hyper.Node object at 0x539a2b0>], Bigram(word='dog', tag='D', prevtag='N'))]
    dog N <pydecode.semiring.HypergraphSemiRing instance at 0x53814d0>
    [([<pydecode.hyper.Node object at 0x539a350>], Bigram(word='dog', tag='N', prevtag='V')), ([<pydecode.hyper.Node object at 0x539acd8>], Bigram(word='dog', tag='N', prevtag='D')), ([<pydecode.hyper.Node object at 0x539a2b0>], Bigram(word='dog', tag='N', prevtag='N'))]
    walked V <pydecode.semiring.HypergraphSemiRing instance at 0x5381b48>
    [([<pydecode.hyper.Node object at 0x539aeb8>], Bigram(word='walked', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x539ab20>], Bigram(word='walked', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x539aaa8>], Bigram(word='walked', tag='V', prevtag='N'))]
    in D <pydecode.semiring.HypergraphSemiRing instance at 0x5381b00>
    [([<pydecode.hyper.Node object at 0x539a620>], Bigram(word='in', tag='D', prevtag='V'))]
    the V <pydecode.semiring.HypergraphSemiRing instance at 0x53814d0>
    [([<pydecode.hyper.Node object at 0x539a508>], Bigram(word='the', tag='V', prevtag='D'))]
    the D <pydecode.semiring.HypergraphSemiRing instance at 0x53816c8>
    [([<pydecode.hyper.Node object at 0x539a508>], Bigram(word='the', tag='D', prevtag='D'))]
    the N <pydecode.semiring.HypergraphSemiRing instance at 0x5381ab8>
    [([<pydecode.hyper.Node object at 0x539a508>], Bigram(word='the', tag='N', prevtag='D'))]
    park V <pydecode.semiring.HypergraphSemiRing instance at 0x5381998>
    [([<pydecode.hyper.Node object at 0x539a850>], Bigram(word='park', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x539abe8>], Bigram(word='park', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x539a3f0>], Bigram(word='park', tag='V', prevtag='N'))]
    park N <pydecode.semiring.HypergraphSemiRing instance at 0x5381320>
    [([<pydecode.hyper.Node object at 0x539a850>], Bigram(word='park', tag='N', prevtag='V')), ([<pydecode.hyper.Node object at 0x539abe8>], Bigram(word='park', tag='N', prevtag='D')), ([<pydecode.hyper.Node object at 0x539a3f0>], Bigram(word='park', tag='N', prevtag='N'))]
    END END <pydecode.semiring.HypergraphSemiRing instance at 0x5381248>
    [([<pydecode.hyper.Node object at 0x539a3a0>], Bigram(word='END', tag='END', prevtag='V')), ([<pydecode.hyper.Node object at 0x539a328>], Bigram(word='END', tag='END', prevtag='N'))]


.. code:: python

    
Step 3: Construct the weights.

.. code:: python

    weights = ph.Weights(hypergraph).build(build_weights)
.. code:: python

    # Find the viterbi path.
    path, vchart = ph.best_path(hypergraph, weights)
    print weights.dot(path)
    
    # Output the path.
    [hypergraph.label(edge) for edge in path.edges]

.. parsed-literal::

    9.6




.. parsed-literal::

    [Bigram(word='the', tag='D', prevtag='ROOT'),
     Bigram(word='dog', tag='N', prevtag='D'),
     Bigram(word='walked', tag='V', prevtag='N'),
     Bigram(word='in', tag='D', prevtag='V'),
     Bigram(word='the', tag='N', prevtag='D'),
     Bigram(word='park', tag='V', prevtag='N'),
     Bigram(word='END', tag='END', prevtag='V')]



.. code:: python

    format = display.HypergraphPathFormatter(hypergraph, [path])
    display.to_ipython(hypergraph, format)



.. image:: hmm_files/hmm_16_0.png



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
    format = HMMFormat(hypergraph, [path])
    display.to_ipython(hypergraph, format)
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
Solve instead using subgradient.

.. code:: python

    gpath, duals = ph.best_constrained(hypergraph, weights, constraints)
.. code:: python

    for d in duals:
        print d.dual, d.constraints
.. code:: python

    display.report(duals)
.. code:: python

    import pydecode.lp as lp
    hypergraph_lp = lp.HypergraphLP.make_lp(hypergraph, weights)
    path = hypergraph_lp.solve()
.. code:: python

    # Output the path.
    for edge in gpath.edges:
        print hypergraph.label(edge)
.. code:: python

    print "check", constraints.check(gpath)
    print "score", weights.dot(gpath)
.. code:: python

    format = HMMFormat(hypergraph, [path, gpath])
    display.to_ipython(hypergraph, format)
.. code:: python

    for constraint in constraints:
        print constraint.label
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
            return {"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])]}
    
    format = HMMConstraintFormat(hypergraph, constraints)
    display.to_ipython(hypergraph, format)
Pruning

.. code:: python

    pruned_hypergraph, pruned_weights = ph.prune_hypergraph(hypergraph, weights, 0.8)
.. code:: python

    
.. code:: python

    display.to_ipython(pruned_hypergraph, HMMFormat(pruned_hypergraph, []))
.. code:: python

    very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, weights, 0.9)