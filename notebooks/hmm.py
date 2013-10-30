
# In[16]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple

import pydecode.chart as chart
import pydecode.semiring as semi
import pandas as pd


## Tutorial 3: HMM Tagger (with constraints)

# We begin by constructing the HMM probabilities.

# In[26]:

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


# In[27]:

pd.DataFrame(transition).fillna(0) 


# Out[27]:

#            D    N  ROOT    V
#     D    0.1  0.1   0.4  0.4
#     END  0.0  0.0   0.0  0.0
#     N    0.8  0.1   0.3  0.3
#     V    0.1  0.8   0.3  0.3

# In[28]:

pd.DataFrame(emission).fillna(0)


# Out[28]:

#           END  ROOT  dog  in  park  the  walked
#     D       0     0  0.1   1   0.0  0.8       0
#     END     1     0  0.0   0   0.0  0.0       0
#     N       0     0  0.8   0   0.1  0.1       0
#     ROOT    0     1  0.0   0   0.0  0.0       0
#     V       0     0  0.1   0   0.9  0.1       1

# Next we specify the labels for the transitions.

# In[4]:

class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)

class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)



# And the scoring function.

# In[5]:

def bigram_weight(bigram):
    return transition[bigram.prevtag][bigram.tag] +     emission[bigram.word][bigram.tag] 


# Now we write out dynamic program. 

# In[6]:

def viterbi(chart):
    words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
    c.init(Tagged(0, "ROOT", "ROOT"))    
    for i, word in enumerate(words[1:], 1):
        prev_tags = emission[words[i-1]].keys()
        for tag in emission[word].iterkeys():
            c[Tagged(i, word, tag)] =                 c.sum([c[key] * c.sr(Bigram(word, tag, prev)) 
                       for prev in prev_tags 
                       for key in [Tagged(i - 1, words[i - 1], prev)] 
                       if key in c])
    return c


# Now we are ready to build the structure itself.

# In[7]:

# The sentence to be tagged.
sentence = 'the dog walked in the park'


# In[8]:

# Create a chart using to compute the probability of the sentence.
c = chart.ChartBuilder(bigram_weight)
viterbi(c).finish()


# Out[8]:

#     10.600000000000001

# In[9]:

# Create a chart to compute the max paths.
c = chart.ChartBuilder(bigram_weight, 
                       chart.ViterbiSemiRing)
viterbi(c).finish()


# Out[9]:

#     9.600000000000001

# But even better we can construct the entrire search space.

# In[10]:

c = chart.ChartBuilder(lambda a:a, semi.HypergraphSemiRing, 
                       build_hypergraph = True)
hypergraph = viterbi(c).finish()


# In[12]:

weights = ph.Weights(hypergraph).build(bigram_weight)

# Find the best path.
path = ph.best_path(hypergraph, weights)
print weights.dot(path)


# Out[12]:

#     9.6
# 

# We can also output the path itself.

# In[13]:

print [hypergraph.label(edge) for edge in path.edges]


# Out[13]:

#     [Bigram(word='the', tag='D', prevtag='ROOT'), Bigram(word='dog', tag='N', prevtag='D'), Bigram(word='walked', tag='V', prevtag='N'), Bigram(word='in', tag='D', prevtag='V'), Bigram(word='the', tag='N', prevtag='D'), Bigram(word='park', tag='V', prevtag='N'), Bigram(word='END', tag='END', prevtag='V')]
# 

# In[78]:

display.HypergraphPathFormatter(hypergraph, [path]).to_ipython()


# Out[78]:

#     <IPython.core.display.Image at 0x36cd790>

# We can also use a custom fancier formatter. These attributes are from graphviz (http://www.graphviz.org/content/attrs)

# In[14]:

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


# Out[14]:

#     <IPython.core.display.Image at 0x4c75050>

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

