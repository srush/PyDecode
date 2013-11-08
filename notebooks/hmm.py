
# In[48]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple

import pydecode.chart as chart
import pydecode.optimization as opt
import pydecode.constraints as cons
import pydecode.semiring as semi
import pandas as pd


## Tutorial 3: HMM Tagger (with constraints)

# We begin by constructing the HMM probabilities.

# In[49]:

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


# In[50]:

pd.DataFrame(transition).fillna(0) 


# Out[50]:

#            D    N  ROOT    V
#     D    0.1  0.1   0.4  0.4
#     END  0.0  0.0   0.0  0.0
#     N    0.8  0.1   0.3  0.3
#     V    0.1  0.8   0.3  0.3

# In[51]:

pd.DataFrame(emission).fillna(0)


# Out[51]:

#           END  ROOT  dog  in  park  the  walked
#     D       0     0  0.1   1   0.0  0.8       0
#     END     1     0  0.0   0   0.0  0.0       0
#     N       0     0  0.8   0   0.1  0.1       0
#     ROOT    0     1  0.0   0   0.0  0.0       0
#     V       0     0  0.1   0   0.9  0.1       1

# Next we specify the labels for the transitions.

# In[52]:

class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)

class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)



# And the scoring function.

# In[53]:

def bigram_potential(bigram):
    return transition[bigram.prevtag][bigram.tag] +     emission[bigram.word][bigram.tag] 


# Now we write out dynamic program. 

# In[54]:

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

# In[55]:

# The sentence to be tagged.
sentence = 'the dog walked in the park'


# In[56]:

# Create a chart using to compute the probability of the sentence.
c = chart.ChartBuilder(bigram_potential)
viterbi(c).finish()


# Out[56]:

#     9.600000381469727

# In[57]:

# Create a chart to compute the max paths.
c = chart.ChartBuilder(bigram_potential, 
                       ph._InsideW)
viterbi(c).finish()


# Out[57]:

#     9.087200164794922

# But even better we can construct the entrire search space.

# In[58]:

c = chart.ChartBuilder(lambda a:a, semi.HypergraphSemiRing, 
                       build_hypergraph = True)
hypergraph = viterbi(c).finish()


# In[59]:

potentials = ph.Potentials(hypergraph).build(bigram_potential)

# Find the best path.
path = ph.best_path(hypergraph, potentials)
print potentials.dot(path)


# Out[59]:


    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)

    <ipython-input-59-5d9224c40bbb> in <module>()
    ----> 1 potentials = ph.Potentials(hypergraph).build(bigram_potential)
          2 
          3 # Find the best path.
          4 path = ph.best_path(hypergraph, potentials)
          5 print potentials.dot(path)


    /home/srush/Projects/decoding/python/pydecode/hyper.so in pydecode.hyper._LogViterbiPotentials.build (python/pydecode/hyper.cpp:10448)()


    <ipython-input-53-fc220ccbeda4> in bigram_potential(bigram)
          1 def bigram_potential(bigram):
    ----> 2     return transition[bigram.prevtag][bigram.tag] +     emission[bigram.word][bigram.tag]
    

    AttributeError: 'NoneType' object has no attribute 'prevtag'


# We can also output the path itself.

# In[ ]:

print [hypergraph.label(edge) for edge in path.edges]


# In[60]:

display.HypergraphPathFormatter(hypergraph, [path]).to_ipython()


# Out[60]:


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-60-66085a6e7465> in <module>()
    ----> 1 display.HypergraphPathFormatter(hypergraph, [path]).to_ipython()
    

    NameError: name 'path' is not defined


# We can also use a custom fancier formatter. These attributes are from graphviz (http://www.graphviz.org/content/attrs)

# In[ ]:

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


# PyDecode also allows you to add extra constraints to the problem. As an example we can add constraints to enfore that the tag of "dog" is the same tag as "park".

# In[ ]:

def cons_name(tag): return "tag_%s"%tag

def build_constraints(bigram):
    if bigram.word == "dog":
        return [(cons_name(bigram.tag), 1)]
    elif bigram.word == "park":
        return [(cons_name(bigram.tag), -1)]
    return []

constraints =     cons.Constraints(hypergraph, [(cons_name(tag), 0) for tag in ["D", "V", "N"]]).build( 
                   build_constraints)


# This check fails because the tags do not agree.

# In[ ]:

print "check", constraints.check(path)


# Solve instead using subgradient.

# In[ ]:

gpath = opt.best_constrained_path(hypergraph, potentials, constraints)


# In[ ]:

import pydecode.lp as lp
hypergraph_lp = lp.HypergraphLP.make_lp(hypergraph, potentials)
hypergraph_lp.solve()
path = hypergraph_lp.path


# In[ ]:

# Output the path.
for edge in gpath.edges:
    print hypergraph.label(edge)


# In[ ]:

print "check", constraints.check(gpath)
print "score", potentials.dot(gpath)


# In[ ]:

HMMFormat(hypergraph, [path, gpath]).to_ipython()


# In[ ]:

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

#HMMConstraintFormat(hypergraph, constraints).to_ipython()


# Pruning
# 

# In[ ]:

pruned_hypergraph, pruned_potentials = ph.prune_hypergraph(hypergraph, potentials, 0.8)


# In[ ]:

HMMFormat(pruned_hypergraph, []).to_ipython()


# In[ ]:

very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, potentials, 0.9)

