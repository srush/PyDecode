
Dependency Parsing
==================


.. code:: python

    sentence = "the man walked to the park"
.. code:: python

    import pydecode.hyper as ph
    import pydecode.display as display
    from collections import namedtuple, defaultdict
    import random
    random.seed(0)
.. code:: python

    Tri = "tri"
    Trap = "trap"
    Right = "right"
    Left = "left"
    class NodeType(namedtuple("NodeType", ["type", "dir", "span"])):
        def __str__(self):
            return "%s %s %d-%d"%(self.type, self.dir, self.span[0], self.span[1])
    
    class Arc(namedtuple("Arc", ["head_index", "modifier_index"])):
        pass
.. code:: python

    def first_order(sentence, c):
        tokens = ["*"] + sentence.split()
        n = len(tokens)
    
        # Add terminal nodes.
        [c.init(NodeType(sh, d, (s, s)))
         for s in range(n) 
         for d in [Right, Left]
         for sh in [Trap, Tri]]
        
        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)
                
                # First create incomplete items.            
                c[NodeType(Trap, Left, span)] = \
                    c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))]
                           for r in range(s, t)])
    
                c[NodeType(Trap, Right, span)] = \
                    c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))]
                           for r in range(s, t)])
                
                # Second create complete items.
                c[NodeType(Tri, Left, span)] = \
                    c.sum([c[NodeType(Tri, Left, (s, r))] * c[NodeType(Trap, Left, (r, t))]
                           for r in range(s, t)])
    
                c[NodeType(Tri, Right, span)] = \
                    c.sum([c[NodeType(Trap, Right, (s, r))] * c[NodeType(Tri, Right, (r, t))]
                           for r in range(s + 1, t + 1)])
        return c
    import pydecode.chart as chart
    sentence = "fans went wild"
    c = chart.ChartBuilder(lambda a: None, 
                           chart.HypergraphSemiRing, 
                           build_hypergraph = True)
    the_chart = first_order(sentence, c)
    hypergraph = the_chart.finish()
.. code:: python

    def build_weights(_):
        return random.random()
    weights = ph.Weights(hypergraph).build(build_weights)
    
    # phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.5)
.. code:: python

    path = ph.best_path(hypergraph, weights)
    best = weights.dot(path)
    maxmarginals = ph.compute_max_marginals(hypergraph, weights)
    avg = 0.0
    for edge in hypergraph.edges:
        avg += maxmarginals[edge]
    avg = avg / float(len(hypergraph.edges))
    thres = ((0.9) * best + (0.1) * avg)
    
    kept = set()
    for edge in hypergraph.edges:
        score = maxmarginals[edge]
        if score >= thres:
            kept.add(edge.id)
.. code:: python

    phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.9)
.. code:: python

    import pydecode.lp as lp
    hyperlp = lp.HypergraphLP.make_lp(phyper, pweights)
    hyperlp.lp.writeLP("parse.lp")
.. code:: python

    class ParseFormat(display.HypergraphPathFormatter):
        def __init__(self, hypergraph, sentence, path):
            self.path = path
            self.hypergraph = hypergraph
            self.sentence = sentence
        def graph_attrs(self):
            return {"rankdir": "TB", "clusterrank": "local"}
        def hypernode_attrs(self, node):
            label = self.hypergraph.node_label(node)
            return {"image": 
                    ("triangle" if label.type == Tri else "trap") + "-" + 
                    ("right" if label.dir == Right else "left") + ".png",
                    "labelloc": "t",
                    "shape": "rect",
                    "style" : "dashed",
                    "label": "%d-%d"%(label.span[0], label.span[1]) 
                    if label.span[0] != label.span[1] else 
                    (["*"] + sentence.split())[label.span[0]],
    
                    }
        def hypernode_subgraph(self, node):
            label = self.hypergraph.node_label(node)
            if label.span[0] == label.span[1]:
                return [("clust_terminals", label.span[0] + (0.5 if label.dir == Right else 0))]
            return []
        def subgraph_format(self, subgraph):
            return {"rank": "same"}
        def hyperedge_node_attrs(self, edge):
            return {"shape": "point"}
        def hyperedge_attrs(self, edge):
            return {"arrowhead": "none", 
                    "color": "orange" if edge in self.path else "black",
                    "penwidth": 5 if edge in self.path else 1}
    
    ParseFormat(hypergraph, sentence, path).to_ipython()



.. image:: parsing_files/parsing_9_0.png


