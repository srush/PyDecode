
# In[70]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple

import pydecode.chart as chart
import pydecode.semiring as semi


# A HMM Tagger Example
# -------------------------
# 
# In this example.

# 
# Construction 
# 
# We begin by constructing the HMM probabilities.

# In[71]:

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


# Next we specify the the index set using namedtuples.

# In[72]:

class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)

class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)

def bigram_weight(bigram):
    return transition[bigram.prevtag][bigram.tag] + emission[bigram.word][bigram.tag] 


# Now we write out dynamic program. 

# In[73]:

def viterbi(chart):
    words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
    c.init(Tagged(0, "ROOT", "ROOT"))    
    for i, word in enumerate(words[1:], 1):
        prev_tags = emission[words[i-1]].keys()
        for tag in emission[word].iterkeys():
            c[Tagged(i, word, tag)] =                 c.sum([c[key] * c.sr(Bigram(word, tag, prev)) 
                       for prev in prev_tags 
                       for key in [Tagged(i - 1, words[i - 1], prev)] if key in c])
    return c


# Now we are ready to build the hypergraph topology itself.

# In[74]:

# Create a chart using to compute the probability of the sentence.
c = chart.ChartBuilder(bigram_weight)
viterbi(c).finish()


# Out[74]:

#     ROOT -> V
#     the V 1.4
#     ROOT -> D
#     the D 2.2
#     ROOT -> N
#     the N 1.4
#     V -> V
#     D -> V
#     N -> V
#     dog V 2.4000000000000004
#     V -> D
#     D -> D
#     N -> D
#     dog D 2.4000000000000004
#     V -> N
#     D -> N
#     N -> N
#     dog N 3.8000000000000003
#     V -> V
#     D -> V
#     N -> V
#     walked V 5.6000000000000005
#     V -> D
#     in D 7.0
#     D -> V
#     the V 7.2
#     D -> D
#     the D 7.9
#     D -> N
#     the N 7.9
#     V -> V
#     D -> V
#     N -> V
#     park V 9.600000000000001
#     V -> N
#     D -> N
#     N -> N
#     park N 8.8
#     V -> END
#     N -> END
#     END END 10.600000000000001
# 

#     10.600000000000001

# In[75]:

# Create a chart to compute the max paths.
c = chart.ChartBuilder(bigram_weight, 
                       chart.ViterbiSemiRing)
viterbi(c).finish()


# Out[75]:

#     ROOT -> V
#     the V 0.4
#     ROOT -> D
#     the D 1.2000000000000002
#     ROOT -> N
#     the N 0.4
#     V -> V
#     D -> V
#     N -> V
#     dog V 1.4000000000000001
#     V -> D
#     D -> D
#     N -> D
#     dog D 1.4000000000000001
#     V -> N
#     D -> N
#     N -> N
#     dog N 2.8000000000000003
#     V -> V
#     D -> V
#     N -> V
#     walked V 4.6000000000000005
#     V -> D
#     in D 6.0
#     D -> V
#     the V 6.2
#     D -> D
#     the D 6.9
#     D -> N
#     the N 6.9
#     V -> V
#     D -> V
#     N -> V
#     park V 8.600000000000001
#     V -> N
#     D -> N
#     N -> N
#     park N 7.800000000000001
#     V -> END
#     N -> END
#     END END 9.600000000000001
# 

#     9.600000000000001

# In[76]:

c = chart.ChartBuilder(lambda a:a, semi.HypergraphSemiRing, 
                       build_hypergraph = True)
hypergraph = viterbi(c).finish()


# Out[76]:

#     ROOT -> V
#     make ROOT -> V
#     the V <pydecode.semiring.HypergraphSemiRing object at 0x36f4750>
#     [([<pydecode.hyper.Node object at 0x35a0a08>], Bigram(word='the', tag='V', prevtag='ROOT'))]
#     ROOT -> D
#     make ROOT -> D
#     the D <pydecode.semiring.HypergraphSemiRing object at 0x36f4810>
#     [([<pydecode.hyper.Node object at 0x35a0a08>], Bigram(word='the', tag='D', prevtag='ROOT'))]
#     ROOT -> N
#     make ROOT -> N
#     the N <pydecode.semiring.HypergraphSemiRing object at 0x36f4f50>
#     [([<pydecode.hyper.Node object at 0x35a0a08>], Bigram(word='the', tag='N', prevtag='ROOT'))]
#     V -> V
#     make V -> V
#     D -> V
#     make D -> V
#     N -> V
#     make N -> V
#     dog V <pydecode.semiring.HypergraphSemiRing object at 0x36f4750>
#     [([<pydecode.hyper.Node object at 0x35a0c38>], Bigram(word='dog', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0f30>], Bigram(word='dog', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0ee0>], Bigram(word='dog', tag='V', prevtag='N'))]
#     V -> D
#     make V -> D
#     D -> D
#     make D -> D
#     N -> D
#     make N -> D
#     dog D <pydecode.semiring.HypergraphSemiRing object at 0x36f4c90>
#     [([<pydecode.hyper.Node object at 0x35a0c38>], Bigram(word='dog', tag='D', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0f30>], Bigram(word='dog', tag='D', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0ee0>], Bigram(word='dog', tag='D', prevtag='N'))]
#     V -> N
#     make V -> N
#     D -> N
#     make D -> N
#     N -> N
#     make N -> N
#     dog N <pydecode.semiring.HypergraphSemiRing object at 0x36f4f10>
#     [([<pydecode.hyper.Node object at 0x35a0c38>], Bigram(word='dog', tag='N', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0f30>], Bigram(word='dog', tag='N', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0ee0>], Bigram(word='dog', tag='N', prevtag='N'))]
#     V -> V
#     make V -> V
#     D -> V
#     make D -> V
#     N -> V
#     make N -> V
#     walked V <pydecode.semiring.HypergraphSemiRing object at 0x36f4c90>
#     [([<pydecode.hyper.Node object at 0x35a0fd0>], Bigram(word='walked', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x35a0be8>], Bigram(word='walked', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x35a0a30>], Bigram(word='walked', tag='V', prevtag='N'))]
#     V -> D
#     make V -> D
#     in D <pydecode.semiring.HypergraphSemiRing object at 0x36f45d0>
#     [([<pydecode.hyper.Node object at 0x37008f0>], Bigram(word='in', tag='D', prevtag='V'))]
#     D -> V
#     make D -> V
#     the V <pydecode.semiring.HypergraphSemiRing object at 0x36ecb10>
#     [([<pydecode.hyper.Node object at 0x3700a08>], Bigram(word='the', tag='V', prevtag='D'))]
#     D -> D
#     make D -> D
#     the D <pydecode.semiring.HypergraphSemiRing object at 0x36ec250>
#     [([<pydecode.hyper.Node object at 0x3700a08>], Bigram(word='the', tag='D', prevtag='D'))]
#     D -> N
#     make D -> N
#     the N <pydecode.semiring.HypergraphSemiRing object at 0x36ecb10>
#     [([<pydecode.hyper.Node object at 0x3700a08>], Bigram(word='the', tag='N', prevtag='D'))]
#     V -> V
#     make V -> V
#     D -> V
#     make D -> V
#     N -> V
#     make N -> V
#     park V <pydecode.semiring.HypergraphSemiRing object at 0x36ec4d0>
#     [([<pydecode.hyper.Node object at 0x3700cb0>], Bigram(word='park', tag='V', prevtag='V')), ([<pydecode.hyper.Node object at 0x3700fd0>], Bigram(word='park', tag='V', prevtag='D')), ([<pydecode.hyper.Node object at 0x3700120>], Bigram(word='park', tag='V', prevtag='N'))]
#     V -> N
#     make V -> N
#     D -> N
#     make D -> N
#     N -> N
#     make N -> N
#     park N <pydecode.semiring.HypergraphSemiRing object at 0x36ec810>
#     [([<pydecode.hyper.Node object at 0x3700cb0>], Bigram(word='park', tag='N', prevtag='V')), ([<pydecode.hyper.Node object at 0x3700fd0>], Bigram(word='park', tag='N', prevtag='D')), ([<pydecode.hyper.Node object at 0x3700120>], Bigram(word='park', tag='N', prevtag='N'))]
#     V -> END
#     make V -> END
#     N -> END
#     make N -> END
#     END END <pydecode.semiring.HypergraphSemiRing object at 0x36ec490>
#     [([<pydecode.hyper.Node object at 0x37004b8>], Bigram(word='END', tag='END', prevtag='V')), ([<pydecode.hyper.Node object at 0x3700620>], Bigram(word='END', tag='END', prevtag='N'))]
# 

# Step 3: Construct the weights.

# In[77]:

weights = ph.Weights(hypergraph).build(bigram_weight)

# Find the best path.
path = ph.best_path(hypergraph, weights)
print weights.dot(path)

# Output the path.
#[hypergraph.label(edge) for edge in path.edges]


# Out[77]:

#     9.6
# 

# In[78]:

display.HypergraphPathFormatter(hypergraph, [path]).to_ipython()


# Out[78]:

#     <IPython.core.display.Image at 0x36cd790>

# We can also use a custom fancier formatter. These attributes are from graphviz (http://www.graphviz.org/content/attrs)

# In[79]:

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


# Out[79]:

#     <IPython.core.display.Image at 0x37b02d0>

# PyDecode also allows you to add extra constraints to the problem. As an example we can add constraints to enfore that the tag of "dog" is the same tag as "park".

# In[80]:

def cons(tag): return "tag_%s"%tag

def build_constraints(bigram):
    if bigram.word == "dog":
        return [(cons(bigram.tag), 1)]
    elif bigram.word == "park":
        return [(cons(bigram.tag), -1)]
    return []

constraints =     ph.Constraints(hypergraph).build( 
                   [(cons(tag), 0) for tag in ["D", "V", "N"]], 
                   build_constraints)


# This check fails because the tags do not agree.

# In[81]:

print "check", constraints.check(path)


# Out[81]:

#     check [<pydecode.hyper.Constraint object at 0x261dd90>, <pydecode.hyper.Constraint object at 0x36e9190>]
# 

# Solve instead using subgradient.

# In[82]:

gpath, duals = ph.best_constrained(hypergraph, weights, constraints)


# In[83]:

for d in duals:
    print d.dual, d.constraints


# Out[83]:

#     9.6 [<pydecode.hyper.Constraint object at 0x261dd90>, <pydecode.hyper.Constraint object at 0x36e9190>]
#     8.8 []
# 

# In[84]:

display.report(duals)


# Out[84]:

# image file:

# In[85]:

import pydecode.lp as lp
hypergraph_lp = lp.HypergraphLP.make_lp(hypergraph, weights)
hypergraph_lp.solve()
path = hypergraph_lp.path


# In[86]:

# Output the path.
for edge in gpath.edges:
    print hypergraph.label(edge)


# Out[86]:

#     ROOT -> D
#     D -> N
#     N -> V
#     V -> D
#     D -> D
#     D -> N
#     N -> END
# 

# In[87]:

print "check", constraints.check(gpath)
print "score", weights.dot(gpath)


# Out[87]:

#     check []
#     score 8.8
# 

# In[88]:

HMMFormat(hypergraph, [path, gpath]).to_ipython()


# Out[88]:

#     <IPython.core.display.Image at 0x43e9050>

# In[89]:

for constraint in constraints:
    print constraint.label


# Out[89]:

#     tag_D
#     tag_V
#     tag_N
# 

# In[90]:

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


# Out[90]:

#     <IPython.core.display.Image at 0x4133dd0>

# Pruning
# 

# In[91]:

pruned_hypergraph, pruned_weights = ph.prune_hypergraph(hypergraph, weights, 0.8)


# In[92]:

HMMFormat(pruned_hypergraph, []).to_ipython()


# Out[92]:

#     <IPython.core.display.Image at 0x391d350>

# In[93]:

very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, weights, 0.9)

