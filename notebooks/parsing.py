
## Dependency Parsing

# In[2]:

sentence = "the man walked to the park"


# In[3]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple, defaultdict
import random
random.seed(0)


# In[4]:

Tri = "tri"
Trap = "trap"
Right = "right"
Left = "left"
class NodeType(namedtuple("NodeType", ["type", "dir", "span"])):
    def __str__(self):
        return "%s %s %d-%d"%(self.type, self.dir, self.span[0], self.span[1])

class Arc(namedtuple("Arc", ["head_index", "modifier_index"])):
    pass


# In[5]:

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


# Out[5]:

#     trap left 0-1 [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4030>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4030>], None)]
#     trap right 0-1 [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4030>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4030>], None)]
#     tri left 0-1 [([<pydecode.hyper.Node object at 0x33c4530>, <pydecode.hyper.Node object at 0x33c4dc8>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4530>, <pydecode.hyper.Node object at 0x33c4dc8>], None)]
#     tri right 0-1 [([<pydecode.hyper.Node object at 0x33c4170>, <pydecode.hyper.Node object at 0x33c4468>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4170>, <pydecode.hyper.Node object at 0x33c4468>], None)]
#     trap left 1-2 [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4ee0>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4ee0>], None)]
#     trap right 1-2 [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4ee0>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4ee0>], None)]
#     tri left 1-2 [([<pydecode.hyper.Node object at 0x33c4030>, <pydecode.hyper.Node object at 0x33c4e40>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4030>, <pydecode.hyper.Node object at 0x33c4e40>], None)]
#     tri right 1-2 [([<pydecode.hyper.Node object at 0x33c4800>, <pydecode.hyper.Node object at 0x33c4878>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4800>, <pydecode.hyper.Node object at 0x33c4878>], None)]
#     trap left 2-3 [([<pydecode.hyper.Node object at 0x33c4878>, <pydecode.hyper.Node object at 0x33c4f80>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4878>, <pydecode.hyper.Node object at 0x33c4f80>], None)]
#     trap right 2-3 [([<pydecode.hyper.Node object at 0x33c4878>, <pydecode.hyper.Node object at 0x33c4f80>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4878>, <pydecode.hyper.Node object at 0x33c4f80>], None)]
#     tri left 2-3 [([<pydecode.hyper.Node object at 0x33c4ee0>, <pydecode.hyper.Node object at 0x33c4e18>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4ee0>, <pydecode.hyper.Node object at 0x33c4e18>], None)]
#     tri right 2-3 [([<pydecode.hyper.Node object at 0x33c4d78>, <pydecode.hyper.Node object at 0x33c4508>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4d78>, <pydecode.hyper.Node object at 0x33c4508>], None)]
#     trap left 0-2 [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4558>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4ee0>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4558>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4ee0>], None)]
#     trap right 0-2 [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4558>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4ee0>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4558>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4ee0>], None)]
#     tri left 0-2 [([<pydecode.hyper.Node object at 0x33c4530>, <pydecode.hyper.Node object at 0x33c4378>], None), ([<pydecode.hyper.Node object at 0x33c4648>, <pydecode.hyper.Node object at 0x33c4e40>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4530>, <pydecode.hyper.Node object at 0x33c4378>], None), ([<pydecode.hyper.Node object at 0x33c4648>, <pydecode.hyper.Node object at 0x33c4e40>], None)]
#     tri right 0-2 [([<pydecode.hyper.Node object at 0x33c4170>, <pydecode.hyper.Node object at 0x33c4c10>], None), ([<pydecode.hyper.Node object at 0x33c4f08>, <pydecode.hyper.Node object at 0x33c4878>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4170>, <pydecode.hyper.Node object at 0x33c4c10>], None), ([<pydecode.hyper.Node object at 0x33c4f08>, <pydecode.hyper.Node object at 0x33c4878>], None)]
#     trap left 1-3 [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4c10>, <pydecode.hyper.Node object at 0x33c4f80>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4c10>, <pydecode.hyper.Node object at 0x33c4f80>], None)]
#     trap right 1-3 [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4c10>, <pydecode.hyper.Node object at 0x33c4f80>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4468>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4c10>, <pydecode.hyper.Node object at 0x33c4f80>], None)]
#     tri left 1-3 [([<pydecode.hyper.Node object at 0x33c4030>, <pydecode.hyper.Node object at 0x33c42b0>], None), ([<pydecode.hyper.Node object at 0x33c4558>, <pydecode.hyper.Node object at 0x33c4e18>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4030>, <pydecode.hyper.Node object at 0x33c42b0>], None), ([<pydecode.hyper.Node object at 0x33c4558>, <pydecode.hyper.Node object at 0x33c4e18>], None)]
#     tri right 1-3 [([<pydecode.hyper.Node object at 0x33c4800>, <pydecode.hyper.Node object at 0x33c4418>], None), ([<pydecode.hyper.Node object at 0x33c4da0>, <pydecode.hyper.Node object at 0x33c4508>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4800>, <pydecode.hyper.Node object at 0x33c4418>], None), ([<pydecode.hyper.Node object at 0x33c4da0>, <pydecode.hyper.Node object at 0x33c4508>], None)]
#     trap left 0-3 [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4210>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4b48>, <pydecode.hyper.Node object at 0x33c4f80>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4210>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4b48>, <pydecode.hyper.Node object at 0x33c4f80>], None)]
#     trap right 0-3 [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4210>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4b48>, <pydecode.hyper.Node object at 0x33c4f80>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c44b8>, <pydecode.hyper.Node object at 0x33c4210>], None), ([<pydecode.hyper.Node object at 0x33c4238>, <pydecode.hyper.Node object at 0x33c4440>], None), ([<pydecode.hyper.Node object at 0x33c4b48>, <pydecode.hyper.Node object at 0x33c4f80>], None)]
#     tri left 0-3 [([<pydecode.hyper.Node object at 0x33c4530>, <pydecode.hyper.Node object at 0x33c4760>], None), ([<pydecode.hyper.Node object at 0x33c4648>, <pydecode.hyper.Node object at 0x33c42b0>], None), ([<pydecode.hyper.Node object at 0x33c43a0>, <pydecode.hyper.Node object at 0x33c4e18>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4530>, <pydecode.hyper.Node object at 0x33c4760>], None), ([<pydecode.hyper.Node object at 0x33c4648>, <pydecode.hyper.Node object at 0x33c42b0>], None), ([<pydecode.hyper.Node object at 0x33c43a0>, <pydecode.hyper.Node object at 0x33c4e18>], None)]
#     tri right 0-3 [([<pydecode.hyper.Node object at 0x33c4170>, <pydecode.hyper.Node object at 0x33c4e68>], None), ([<pydecode.hyper.Node object at 0x33c4f08>, <pydecode.hyper.Node object at 0x33c4418>], None), ([<pydecode.hyper.Node object at 0x33c4d28>, <pydecode.hyper.Node object at 0x33c4508>], None)] [] None False
#     [([<pydecode.hyper.Node object at 0x33c4170>, <pydecode.hyper.Node object at 0x33c4e68>], None), ([<pydecode.hyper.Node object at 0x33c4f08>, <pydecode.hyper.Node object at 0x33c4418>], None), ([<pydecode.hyper.Node object at 0x33c4d28>, <pydecode.hyper.Node object at 0x33c4508>], None)]
#     [] [<pydecode.hyper.Node object at 0x33c4828>] None False
# 

# In[14]:

#display.HypergraphFormatter(hypergraph).to_ipython()


# In[7]:

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


# In[8]:

def build_weights(_):
    return random.random()
weights = ph.Weights(hypergraph).build(build_weights)

# phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.5)


# In[9]:

path = ph.best_path(hypergraph, weights)
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


# Out[9]:

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

# In[10]:

phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.9)


# In[11]:

#path = ph.best_path(phyper, pweights)


# In[12]:

import pydecode.lp as lp
hyperlp = lp.HypergraphLP.make_lp(phyper, pweights)
hyperlp.lp.writeLP("parse.lp")
# with open("parse.lp") as w:
#     print >>w, open("/tmp/tmp.lp").read()


# In[13]:

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

ParseFormat(hypergraph, sentence, path).to_ipython()


# Out[13]:

#     <IPython.core.display.Image at 0x35a0f10>
