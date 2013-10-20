
# In[21]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple, defaultdict
import random
random.seed(0)


# In[22]:

sentence = "the man walked to the park"


# In[23]:

Tri = "tri"
Trap = "trap"
Right = "right"
Left = "left"
class NodeType(namedtuple("NodeType", ["type", "dir", "span"])):
    def __str__(self):
        return "%s %s %d-%d"%(self.type, self.dir, self.span[0], self.span[1])

class Arc(namedtuple("Arc", ["head_index", "modifier_index"])):
    pass


# In[24]:

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
            c[NodeType(Trap, Left, span)] =                 c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))]
                       for r in range(s, t)])

            c[NodeType(Trap, Right, span)] =                 c.sum([c[NodeType(Tri, Right, (s, r))] * c[NodeType(Tri, Left, (r+1, t))]
                       for r in range(s, t)])
            
            # Second create complete items.
            c[NodeType(Tri, Left, span)] =                 c.sum([c[NodeType(Tri, Left, (s, r))] * c[NodeType(Trap, Left, (r, t))]
                       for r in range(s, t)])

            c[NodeType(Tri, Right, span)] =                 c.sum([c[NodeType(Trap, Right, (s, r))] * c[NodeType(Tri, Right, (r, t))]
                       for r in range(s + 1, t + 1)])

    print c[NodeType(Tri, Right, (0, n-1))]
    #c[NodeType(Tri, Right, (0, n))] = c[NodeType(Tri, Right, (0, n-1))] * c.sr(None)  
    return c
import pydecode.chart as chart
sentence = "fans went wild"
c = chart.ChartBuilder(lambda a: None, 
                       chart.HypergraphSemiRing, 
                       build_hypergraph = True)
the_chart = first_order(sentence, c)
hypergraph = the_chart.finish()


# Out[24]:

#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     start
#     trap left 0-1 <pydecode.semiring.HypergraphSemiRing instance at 0x2e22b48>
#     [([<pydecode.hyper.Node object at 0x2bb82b0>, <pydecode.hyper.Node object at 0x2bb81e8>], None)]
#     trap right 0-1 <pydecode.semiring.HypergraphSemiRing instance at 0x2e22c20>
#     [([<pydecode.hyper.Node object at 0x2bb82b0>, <pydecode.hyper.Node object at 0x2bb81e8>], None)]
#     tri left 0-1 <pydecode.semiring.HypergraphSemiRing instance at 0x2e22cf8>
#     [([<pydecode.hyper.Node object at 0x2bb87d8>, <pydecode.hyper.Node object at 0x2bb8cb0>], None)]
#     tri right 0-1 <pydecode.semiring.HypergraphSemiRing instance at 0x2e22dd0>
#     [([<pydecode.hyper.Node object at 0x2bb8ee0>, <pydecode.hyper.Node object at 0x2bb8d28>], None)]
#     trap left 1-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e22ea8>
#     [([<pydecode.hyper.Node object at 0x2bb8d28>, <pydecode.hyper.Node object at 0x2bb85d0>], None)]
#     trap right 1-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e22f80>
#     [([<pydecode.hyper.Node object at 0x2bb8d28>, <pydecode.hyper.Node object at 0x2bb85d0>], None)]
#     tri left 1-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d098>
#     [([<pydecode.hyper.Node object at 0x2bb81e8>, <pydecode.hyper.Node object at 0x2e0e030>], None)]
#     tri right 1-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d170>
#     [([<pydecode.hyper.Node object at 0x2e0e058>, <pydecode.hyper.Node object at 0x2bb8558>], None)]
#     trap left 2-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d248>
#     [([<pydecode.hyper.Node object at 0x2bb8558>, <pydecode.hyper.Node object at 0x2bb8620>], None)]
#     trap right 2-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d320>
#     [([<pydecode.hyper.Node object at 0x2bb8558>, <pydecode.hyper.Node object at 0x2bb8620>], None)]
#     tri left 2-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d3f8>
#     [([<pydecode.hyper.Node object at 0x2bb85d0>, <pydecode.hyper.Node object at 0x2e0e238>], None)]
#     tri right 2-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d4d0>
#     [([<pydecode.hyper.Node object at 0x2e0e350>, <pydecode.hyper.Node object at 0x2bb86c0>], None)]
#     trap left 0-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d758>
#     [([<pydecode.hyper.Node object at 0x2bb82b0>, <pydecode.hyper.Node object at 0x2e0e198>], None), ([<pydecode.hyper.Node object at 0x2bb8580>, <pydecode.hyper.Node object at 0x2bb85d0>], None)]
#     trap right 0-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d830>
#     [([<pydecode.hyper.Node object at 0x2bb82b0>, <pydecode.hyper.Node object at 0x2e0e198>], None), ([<pydecode.hyper.Node object at 0x2bb8580>, <pydecode.hyper.Node object at 0x2bb85d0>], None)]
#     tri left 0-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d908>
#     [([<pydecode.hyper.Node object at 0x2bb87d8>, <pydecode.hyper.Node object at 0x2e0ebe8>], None), ([<pydecode.hyper.Node object at 0x2bb8288>, <pydecode.hyper.Node object at 0x2e0e030>], None)]
#     tri right 0-2 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1d9e0>
#     [([<pydecode.hyper.Node object at 0x2bb8ee0>, <pydecode.hyper.Node object at 0x2e0e2b0>], None), ([<pydecode.hyper.Node object at 0x2e0ea08>, <pydecode.hyper.Node object at 0x2bb8558>], None)]
#     trap left 1-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1dab8>
#     [([<pydecode.hyper.Node object at 0x2bb8d28>, <pydecode.hyper.Node object at 0x2e0e3a0>], None), ([<pydecode.hyper.Node object at 0x2e0e2b0>, <pydecode.hyper.Node object at 0x2bb8620>], None)]
#     trap right 1-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1db90>
#     [([<pydecode.hyper.Node object at 0x2bb8d28>, <pydecode.hyper.Node object at 0x2e0e3a0>], None), ([<pydecode.hyper.Node object at 0x2e0e2b0>, <pydecode.hyper.Node object at 0x2bb8620>], None)]
#     tri left 1-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1dc68>
#     [([<pydecode.hyper.Node object at 0x2bb81e8>, <pydecode.hyper.Node object at 0x2e0e0d0>], None), ([<pydecode.hyper.Node object at 0x2e0e198>, <pydecode.hyper.Node object at 0x2e0e238>], None)]
#     tri right 1-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1dd40>
#     [([<pydecode.hyper.Node object at 0x2e0e058>, <pydecode.hyper.Node object at 0x2e0e3c8>], None), ([<pydecode.hyper.Node object at 0x2e0e738>, <pydecode.hyper.Node object at 0x2bb86c0>], None)]
#     trap left 0-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1de18>
#     [([<pydecode.hyper.Node object at 0x2bb82b0>, <pydecode.hyper.Node object at 0x2e0e2d8>], None), ([<pydecode.hyper.Node object at 0x2bb8580>, <pydecode.hyper.Node object at 0x2e0e3a0>], None), ([<pydecode.hyper.Node object at 0x2e0eb48>, <pydecode.hyper.Node object at 0x2bb8620>], None)]
#     trap right 0-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1def0>
#     [([<pydecode.hyper.Node object at 0x2bb82b0>, <pydecode.hyper.Node object at 0x2e0e2d8>], None), ([<pydecode.hyper.Node object at 0x2bb8580>, <pydecode.hyper.Node object at 0x2e0e3a0>], None), ([<pydecode.hyper.Node object at 0x2e0eb48>, <pydecode.hyper.Node object at 0x2bb8620>], None)]
#     tri left 0-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e1dfc8>
#     [([<pydecode.hyper.Node object at 0x2bb87d8>, <pydecode.hyper.Node object at 0x2e0e120>], None), ([<pydecode.hyper.Node object at 0x2bb8288>, <pydecode.hyper.Node object at 0x2e0e0d0>], None), ([<pydecode.hyper.Node object at 0x2e0ed50>, <pydecode.hyper.Node object at 0x2e0e238>], None)]
#     tri right 0-3 <pydecode.semiring.HypergraphSemiRing instance at 0x2e190e0>
#     [([<pydecode.hyper.Node object at 0x2bb8ee0>, <pydecode.hyper.Node object at 0x2e0eeb8>], None), ([<pydecode.hyper.Node object at 0x2e0ea08>, <pydecode.hyper.Node object at 0x2e0e3c8>], None), ([<pydecode.hyper.Node object at 0x2e0e5f8>, <pydecode.hyper.Node object at 0x2bb86c0>], None)]
#     <pydecode.semiring.HypergraphSemiRing instance at 0x2e1dfc8>
# 

# In[25]:

display.to_ipython(hypergraph, display.HypergraphFormatter(hypergraph))


# Out[25]:

#     <IPython.core.display.Image at 0x2de8f90>

# In[26]:

# def build_first_order(sentence):
#     tokens = ["*"] + sentence.split()
#     hypergraph = ph.Hypergraph()
#     with hypergraph.builder() as b:
#         chart = defaultdict(lambda: None)
#         def add_node(b, edges, key, terminal = False):
#             edges = [e for e in edges if e is not None]
#             if edges or terminal:
#                 chart[key] = b.add_node(edges, label = key)

#         def add_edge(key1, key2):
#             left = chart[key1]
#             right = chart[key2]
#             if left is not None and right is not None:
#                 return ([left, right], None)
#             return None

    

#         # Add terminal nodes.
#         [add_node(b, [], NodeType(c, d, (s, s)), True)
#          for s in range(n)
#          for d in [Right, Left]
#          for c in [Trap, Tri]]

#         for k in range(n):
#             for s in range(n):
#                 t = k + s
#                 if t >= n: break
#                 span = (s, t)

#                 # First create incomplete items.
#                 edges = [add_edge(NodeType(Tri, Right, (s, r)),
#                                   NodeType(Tri, Left, (r+1, t)))
#                          for r in range(s, t)]
#                 add_node(b, edges, NodeType(Trap, Left, span))

#                 edges = [add_edge(NodeType(Tri, Right, (s, r)),
#                                   NodeType(Tri, Left, (r+1, t)))
#                          for r in range(s, t)]
#                 add_node(b, edges, NodeType(Trap, Right, span))

#                 # Second create complete items.
#                 edges = [add_edge(NodeType(Tri, Left, (s, r)),
#                                   NodeType(Trap, Left, (r, t)))
#                          for r in range(s, t)]
#                 add_node(b, edges, NodeType(Tri, Left, span))
            
#                 edges = [add_edge(NodeType(Trap, Right, (s, r)),
#                                   NodeType(Tri, Right, (r, t)))
#                          for r in range(s + 1, t + 1)]
#                 print len(edges), span, n -1, edges
#                 add_node(b, edges, NodeType(Tri, Right, span))
#         b.add_node([([chart[NodeType(Tri, Right, (0, n-1))]], "")], NodeType(Tri, Right, (0, n-1)))
#     return hypergraph
# sentence = "fans went wild"
# hypergraph = build_first_order(sentence)


# In[27]:

def build_weights(_):
    return random.random()
weights = ph.Weights(hypergraph).build(build_weights)

# phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.5)


# In[28]:

path, _ = ph.best_path(hypergraph, weights)
best = weights.dot(path)
maxmarginals = ph.compute_max_marginals(hypergraph, weights)
avg = 0.0
for edge in hypergraph.edges:
    avg += maxmarginals[edge]
avg = avg / float(len(hypergraph.edges))
thres = ((0.9) * best + (0.1) * avg)
print thres
kept = set()
for edge in hypergraph.edges:
    score = maxmarginals[edge]
    print score, score < thres
    if score >= thres:
        kept.add(edge.id)


# Out[28]:

#     4.15689503835
#     4.15764270301 False
#     4.15764270301 False
#     4.12753610309 True
#     3.32159117163 True
#     4.12753610309 True
#     3.32159117163 True
#     4.19870703727 False
#     3.42792971127 True
#     4.19870703727 False
#     3.42792971127 True
#     3.67879308045 True
#     3.94589700334 True
#     2.87314027256 True
#     3.94589700334 True
#     4.19870703727 False
#     3.23429972651 True
#     3.84509660701 True
#     3.32159117163 True
#     4.19870703727 False
#     4.12753610309 True
#     2.40088214474 True
#     3.84509660701 True
#     4.19870703727 False
#     4.15764270301 False
#     3.94589700334 True
#     3.84509660701 True
#     3.42792971127 True
#     4.19870703727 False
# 

# In[29]:

phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.9)


# In[30]:

#path, _ = ph.best_path(phyper, pweights)


# In[32]:

import pydecode.lp as lp
hyperlp = lp.HypergraphLP.make_lp(phyper, pweights)
hyperlp.lp.writeLP("parse.lp")
# with open("parse.lp") as w:
#     print >>w, open("/tmp/tmp.lp").read()


# Out[32]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-32-d7393fea49c5> in <module>()
          1 import pydecode.lp as lp
    ----> 2 hyperlp = lp.HypergraphLP.make_lp(phyper, pweights)
          3 hyperlp.lp.writeLP("parse.lp")
          4 # with open("parse.lp") as w:
          5 #     print >>w, open("/tmp/tmp.lp").read()


    /home/srush/Projects/decoding/python/pydecode/lp.pyc in make_lp(hypergraph, weights, name, var_type)
        124         # x(v) = \sum_{e : h(e) = v} x(e)
        125         for node in hypergraph.nodes:
    --> 126             if node.is_terminal: continue
        127             prob += node_vars[node.id] == sum([edge_vars[edge.id]
        128                                             for edge in node.edges])


    TypeError: 'bool' object is not callable


# In[ ]:

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
        #return {"arrowhead": "none", "style": "" if edge in self.path else "invis" }
# "shape": "polygon",
#                 "skew" : 0.5 if label.dir == Left  else -0.5,
#                 "sides" : 3 if label.type == Tri else 4,
                
#display.to_ipython(phyper, ParseFormat(phyper, sentence, path))

# display.to_image(hypergraph, "parse_hypergraph.png", ParseFormat(hypergraph, sentence, path))
# display.to_image(hypergraph, "parse_hypergraph_no_path.png", ParseFormat(hypergraph, sentence, []))
display.to_ipython(hypergraph, ParseFormat(hypergraph, sentence, path))

