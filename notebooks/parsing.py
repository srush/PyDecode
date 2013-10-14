
# In[3]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple, defaultdict
import random
random.seed(0)


# In[4]:

sentence = "the man walked to the park"


# In[5]:

Tri = "tri"
Trap = "trap"
Right = "right"
Left = "left"
class NodeType(namedtuple("NodeType", ["type", "dir", "span"])):
    def __str__(self):
        return "%s %s %d-%d"%(self.type, self.dir, self.span[0], self.span[1])

class Arc(namedtuple("Arc", ["head_index", "modifier_index"])):
    pass


# In[19]:

def build_first_order(sentence):
    tokens = ["*"] + sentence.split()
    hypergraph = ph.Hypergraph()
    with hypergraph.builder() as b:
        chart = defaultdict(lambda: None)
        def add_node(b, edges, key, terminal = False):
            edges = [e for e in edges if e is not None]
            if edges or terminal:
                chart[key] = b.add_node(edges, label = key)

        def add_edge(key1, key2):
            left = chart[key1]
            right = chart[key2]
            if left is not None and right is not None:
                return ([left, right], None)
            return None

        n = len(tokens)

        # Add terminal nodes.
        [add_node(b, [], NodeType(c, d, (s, s)), True)
         for s in range(n)
         for d in [Right, Left]
         for c in [Trap, Tri]]

        for k in range(n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)

                # First create incomplete items.
                edges = [add_edge(NodeType(Tri, Right, (s, r)),
                                  NodeType(Tri, Left, (r+1, t)))
                         for r in range(s, t)]
                add_node(b, edges, NodeType(Trap, Left, span))

                edges = [add_edge(NodeType(Tri, Right, (s, r)),
                                  NodeType(Tri, Left, (r+1, t)))
                         for r in range(s, t)]
                add_node(b, edges, NodeType(Trap, Right, span))

                # Second create complete items.
                edges = [add_edge(NodeType(Tri, Left, (s, r)),
                                  NodeType(Trap, Left, (r, t)))
                         for r in range(s, t)]
                add_node(b, edges, NodeType(Tri, Left, span))
            
                edges = [add_edge(NodeType(Trap, Right, (s, r)),
                                  NodeType(Tri, Right, (r, t)))
                         for r in range(s + 1, t + 1)]
                print len(edges), span, n -1, edges
                add_node(b, edges, NodeType(Tri, Right, span))
        b.add_node([([chart[NodeType(Tri, Right, (0, n-1))]], "")], NodeType(Tri, Right, (0, n-1)))
    return hypergraph
sentence = "fans went wild"
hypergraph = build_first_order(sentence)


# Out[19]:

#     0 (0, 0) 3 []
#     0 (1, 1) 3 []
#     0 (2, 2) 3 []
#     0 (3, 3) 3 []
#     1 (0, 1) 3 [([<pydecode.hyper.Node object at 0x4421f30>, <pydecode.hyper.Node object at 0x44217b0>], None)]
#     1 (1, 2) 3 [([<pydecode.hyper.Node object at 0x4421580>, <pydecode.hyper.Node object at 0x44218c8>], None)]
#     1 (2, 3) 3 [([<pydecode.hyper.Node object at 0x4421530>, <pydecode.hyper.Node object at 0x4421df0>], None)]
#     2 (0, 2) 3 [([<pydecode.hyper.Node object at 0x4421f30>, <pydecode.hyper.Node object at 0x44215d0>], None), ([<pydecode.hyper.Node object at 0x44218a0>, <pydecode.hyper.Node object at 0x44218c8>], None)]
#     2 (1, 3) 3 [([<pydecode.hyper.Node object at 0x4421580>, <pydecode.hyper.Node object at 0x4421828>], None), ([<pydecode.hyper.Node object at 0x44214b8>, <pydecode.hyper.Node object at 0x4421df0>], None)]
#     3 (0, 3) 3 [([<pydecode.hyper.Node object at 0x4421f30>, <pydecode.hyper.Node object at 0x4421670>], None), ([<pydecode.hyper.Node object at 0x44218a0>, <pydecode.hyper.Node object at 0x4421828>], None), ([<pydecode.hyper.Node object at 0x4421b70>, <pydecode.hyper.Node object at 0x4421df0>], None)]
# 

# In[20]:

def build_weights(_):
    return random.random()
weights = ph.Weights(hypergraph).build(build_weights)

# phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.5)


# In[8]:

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


# Out[8]:

#     5.07134788465
#     5.07065375625 True
#     5.07065375625 True
#     5.04054715632 True
#     4.23460222487 True
#     5.04054715632 True
#     4.23460222487 True
#     5.11171809051 False
#     4.34094076451 True
#     5.11171809051 False
#     4.34094076451 True
#     4.59180413369 True
#     4.85890805657 True
#     3.7861513258 True
#     4.85890805657 True
#     5.11171809051 False
#     4.14731077975 True
#     4.75810766025 True
#     4.23460222487 True
#     5.11171809051 False
#     5.04054715632 True
#     3.31389319798 True
#     4.75810766025 True
#     5.11171809051 False
#     5.07065375625 True
#     4.85890805657 True
#     4.75810766025 True
#     4.34094076451 True
#     5.11171809051 False
#     5.11171809051 False
# 

# In[23]:

phyper, pweights = ph.prune_hypergraph(hypergraph, weights, 0.9)


# In[10]:

#path, _ = ph.best_path(phyper, pweights)


# In[28]:

import pydecode.lp as lp
hyperlp = lp.HypergraphLP.make_lp(phyper, pweights)
hyperlp.lp.writeLP("parse.lp")
# with open("parse.lp") as w:
#     print >>w, open("/tmp/tmp.lp").read()


# In[30]:

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
                "color": "red" if edge in self.path else "black",
                "penwidth": 5 if edge in self.path else 1}
        #return {"arrowhead": "none", "style": "" if edge in self.path else "invis" }
# "shape": "polygon",
#                 "skew" : 0.5 if label.dir == Left  else -0.5,
#                 "sides" : 3 if label.type == Tri else 4,
                
#display.to_ipython(phyper, ParseFormat(phyper, sentence, path))

# display.to_image(hypergraph, "parse_hypergraph.png", ParseFormat(hypergraph, sentence, path))
# display.to_image(hypergraph, "parse_hypergraph_no_path.png", ParseFormat(hypergraph, sentence, []))
display.to_ipython(hypergraph, ParseFormat(hypergraph, sentence, path))


# Out[30]:

#     <IPython.core.display.Image at 0x4630850>

#     <IPython.core.display.Image at 0x4630810>
