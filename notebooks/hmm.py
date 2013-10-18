
# A Constrained HMM Example
# ---------------------

# In[3]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple
import pydecode.chart as chart


# We begin by constructing the HMM probabilities.

# In[4]:

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


# Next we specify the the index set using namedtuples.

# In[5]:

class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)


# In[6]:

def build_weights((word, tag, prev_tag)):
    return transition[prev_tag][tag] + emission[word][tag]


# In[59]:

def viterbi(chart):
    words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
    c.init(Tagged(0, "ROOT", "ROOT"))
    for i, word in enumerate(words[1:], 1):
        prev_tags = emission[words[i - 1]].keys()
        for tag in emission[word].iterkeys():
            ls = [c[key] * c.sr(Bigram(word, tag, prev))
                     for prev in prev_tags
                     for key in [Tagged(i - 1, words[i - 1], prev)] if key in c]
            c[Tagged(i, word, tag)] = c.sum(ls)
    return c


# Now we are ready to build the  hypergraph topology itself.

# In[60]:

c = chart.ChartBuilder(lambda a: build_weights(Bigram(*a)))
the_chart = viterbi(c)
the_chart[Tagged(7 , "END", "END")].v


# Out[60]:

#     the V 0.4
#     the D 1.2000000000000002
#     the N 0.4
#     dog V 1.4000000000000001
#     dog D 1.4000000000000001
#     dog N 2.8000000000000003
#     walked V 4.6000000000000005
#     in D 6.0
#     the V 6.2
#     the D 6.9
#     the N 6.9
#     park V 8.600000000000001
#     park N 7.800000000000001
#     END END 9.600000000000001
#

#     9.600000000000001

# In[*]:

hypergraph = ph.Hypergraph()
with hypergraph.builder() as b:
    c = chart.ChartBuilder(lambda a: Bigram(*a), b, chart.HypergraphSemiRing)
    the_chart = viterbi(c)

# In[56]:




# Step 3: Construct the weights.

# In[1]:

weights = ph.Weights(hypergraph).build(build_weights)


# In[24]:

# Find the viterbi path.
path, chart = ph.best_path(hypergraph, weights)
print weights.dot(path)

# Output the path.
[hypergraph.label(edge) for edge in path.edges]


# Out[24]:

#     9.6
#

#     [Bigram(word='the', tag='D', prevtag='ROOT'),
#      Bigram(word='dog', tag='N', prevtag='D'),
#      Bigram(word='walked', tag='V', prevtag='N'),
#      Bigram(word='in', tag='D', prevtag='V'),
#      Bigram(word='the', tag='N', prevtag='D'),
#      Bigram(word='park', tag='V', prevtag='N'),
#      Bigram(word='END', tag='END', prevtag='V')]

# In[25]:

format = display.HypergraphPathFormatter(hypergraph, [path])
display.to_ipython(hypergraph, format)


# Out[25]:

#     <IPython.core.display.Image at 0x3cb5dd0>

# We can also use a custom fancier formatter. These attributes are from graphviz (http://www.graphviz.org/content/attrs)

# In[26]:

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
        return {"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])],
                "rank" : "same"}
    def graph_attrs(self): return {"rankdir":"RL"}
format = HMMFormat(hypergraph, [path])
display.to_ipython(hypergraph, format)


# Out[26]:

#     <IPython.core.display.Image at 0x4838390>

# PyDecode also allows you to add extra constraints to the problem. As an example we can add constraints to enfore that the tag of "dog" is the same tag as "park".

# In[27]:

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

# In[28]:

print "check", constraints.check(path)


# Out[28]:

#     check ['tag_V', 'tag_N']
#

# Solve instead using subgradient.

# In[29]:

gpath, duals = ph.best_constrained(hypergraph, weights, constraints)


# In[30]:

for d in duals:
    print d.dual, d.constraints


# Out[30]:

#     9.6 [<pydecode.hyper.Constraint object at 0x3ede1d0>, <pydecode.hyper.Constraint object at 0x3ede090>]
#     8.8 []
#

# In[31]:

display.report(duals)


# Out[31]:

# image file:

# In[32]:

import pydecode.lp as lp
hypergraph_lp = lp.HypergraphLP.make_lp(hypergraph, weights)
path = hypergraph_lp.solve()


# In[33]:

# Output the path.
for edge in gpath.edges:
    print hypergraph.label(edge)


# Out[33]:

#     ROOT -> D
#     D -> N
#     N -> V
#     V -> D
#     D -> D
#     D -> N
#     N -> END
#

# In[34]:

print "check", constraints.check(gpath)
print "score", weights.dot(gpath)


# Out[34]:

#     check []
#     score 8.8
#

# In[35]:

format = HMMFormat(hypergraph, [path, gpath])
display.to_ipython(hypergraph, format)


# Out[35]:

#     <IPython.core.display.Image at 0x3abe910>

# In[36]:

for constraint in constraints:
    print constraint.label


# Out[36]:

#     tag_D
#     tag_V
#     tag_N
#

# In[37]:

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


# Out[37]:

#     <IPython.core.display.Image at 0x479cb50>

# Pruning
#

# In[38]:

pruned_hypergraph, pruned_weights = ph.prune_hypergraph(hypergraph, weights, 0.8)


# In[38]:




# In[39]:

display.to_ipython(pruned_hypergraph, HMMFormat(pruned_hypergraph, []))


# Out[39]:

#     <IPython.core.display.Image at 0x3eea390>

# In[40]:

very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, weights, 0.9)
