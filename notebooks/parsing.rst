
Tutorial 4: Dependency Parsing
==============================


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
                    c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))] * c.sr(Arc(r, s))
                           for r in range(s, t)])
    
                c[NodeType(Trap, Right, span)] = \
                    c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))] * c.sr(Arc(head_index=s, modifier_index=r))
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
    c = chart.ChartBuilder(lambda a: a, 
                           chart.HypergraphSemiRing, 
                           build_hypergraph = True, strict=False)
    the_chart = first_order(sentence, c)
    hypergraph = the_chart.finish()

::


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-4-78d652f2fde9> in <module>()
         34                        chart.HypergraphSemiRing,
         35                        build_hypergraph = True)
    ---> 36 the_chart = first_order(sentence, c)
         37 hypergraph = the_chart.finish()


    <ipython-input-4-78d652f2fde9> in first_order(sentence, c)
         17             # First create incomplete items.
         18             c[NodeType(Trap, Left, span)] =                 c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))] * c.sr(Arc(r, s))
    ---> 19                        for r in range(s, t)])
         20 
         21             c[NodeType(Trap, Right, span)] =                 c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))] * c.sr(Arc(head_index=s, modifier_index=r))


    /home/srush/Projects/decoding/python/pydecode/chart.pyc in __getitem__(self, label)
        130     def __getitem__(self, label):
        131         if self._strict:
    --> 132             raise Exception("Label not in chart: %s"%label)
        133         if self._debug:
        134             print >>sys.stderr, "Getting", label, label in self._chart


    TypeError: not all arguments converted during string formatting


.. code:: python

    def build_potentials(arc):
        print arc
        return random.random()
    potentials = ph.Potentials(hypergraph).build(build_potentials)
    
    # phyper, ppotentials = ph.prune_hypergraph(hypergraph, potentials, 0.5)
.. code:: python

    path = ph.best_path(hypergraph, potentials)
    best = potentials.dot(path)
    maxmarginals = ph.compute_marginals(hypergraph, potentials)
    avg = 0.0
    for edge in hypergraph.edges:
        avg += float(maxmarginals[edge])
    avg = avg / float(len(hypergraph.edges))
    thres = ((0.9) * best + (0.1) * avg)
    
    kept = set()
    for edge in hypergraph.edges:
        score = float(maxmarginals[edge])
        if score >= thres:
            kept.add(edge.id)
.. code:: python

    potentials = ph.InsidePotentials(hypergraph).build(build_potentials)
    marginals = ph.compute_marginals(hypergraph, potentials)
    base = marginals[hypergraph.root]
    for edge in hypergraph.edges:
        print marginals[edge].value / base.value
.. code:: python

    phyper, ppotentials = ph.prune_hypergraph(hypergraph, potentials, 0.1)
.. code:: python

    import pydecode.lp as lp
    hyperlp = lp.HypergraphLP.make_lp(phyper, ppotentials)
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
.. code:: python

    import networkx as nx
    from networkx.readwrite import json_graph
    import json
    G = ParseFormat(hypergraph, sentence, path).to_graphviz()
    G2 = nx.from_agraph(G)
    d = json_graph.node_link_data(G2) # node-link format to serialize
    # write json 
    json.dump(d, open('force.json','w'))
    #nx.write_gexf(G2, "test_graph.gexf")