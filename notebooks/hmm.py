
# A Constrained HMM Example
# ---------------------

# In[1]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple


# We begin by constructing the HMM probabilities.

# In[2]:

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

# In[3]:

class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)


# Now we are ready to build the  hypergraph topology itself.

# In[4]:

hypergraph = ph.Hypergraph()                      
with hypergraph.builder() as b:
    node_start = b.add_node(label = Tagged(-1, "<s>", "<t>"))
    node_list = [(node_start, "ROOT")]
    words = sentence.strip().split(" ") + ["END"]
        
    for i, word in enumerate(words):
        next_node_list = []
        for tag in emission[word].iterkeys():
            edges = (([prev_node], Bigram(word, tag, prev_tag))
                     for prev_node, prev_tag in node_list)
            node = b.add_node(edges, label = Tagged(i, word, tag))
            next_node_list.append((node, tag))
        node_list = next_node_list


# Step 3: Construct the weights.

# In[5]:

def build_weights((word, tag, prev_tag)):
    return transition[prev_tag][tag] + emission[word][tag] 
weights = ph.Weights(hypergraph).build(build_weights)


# In[6]:

# Find the viterbi path.
path, chart = ph.best_path(hypergraph, weights)
print weights.dot(path)

# Output the path.
[hypergraph.label(edge) for edge in path.edges]


# Out[6]:

#     9.6
# 

#     [Bigram(word='the', tag='D', prevtag='ROOT'),
#      Bigram(word='dog', tag='N', prevtag='D'),
#      Bigram(word='walked', tag='V', prevtag='N'),
#      Bigram(word='in', tag='D', prevtag='V'),
#      Bigram(word='the', tag='N', prevtag='D'),
#      Bigram(word='park', tag='V', prevtag='N'),
#      Bigram(word='END', tag='END', prevtag='V')]

# In[7]:

format = display.HypergraphPathFormatter(hypergraph, [path])
display.to_ipython(hypergraph, format)


# Out[7]:

#     <IPython.core.display.Image at 0x364f8d0>

# We can also use a custom fancier formatter. These attributes are from graphviz (http://www.graphviz.org/content/attrs)

# In[8]:

class HMMFormat(display.HypergraphPathFormatter):
    def hypernode_attrs(self, node):
        label = self.hypergraph.node_label(node)
        return {"label": label.tag, "shape": ""}
    def hyperedge_node_attrs(self, edge):
        return {"color": "pink", "shape": "point"}
    def hypernode_subgraph(self, node):
        label = self.hypergraph.node_label(node)
        return ["cluster_" + str(label.position)]
    def subgraph_format(self, subgraph):
        return {"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])]}

format = HMMFormat(hypergraph, [path])
display.to_ipython(hypergraph, format)


# Out[8]:

#     <IPython.core.display.Image at 0x3ae07d0>

# PyDecode also allows you to add extra constraints to the problem. As an example we can add constraints to enfore that the tag of "dog" is the same tag as "park".

# In[9]:

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

# In[10]:

print "check", constraints.check(path)


# Out[10]:

#     check ['tag_V', 'tag_N']
# 

# Solve instead using subgradient.

# In[11]:

gpath, duals = ph.best_constrained(hypergraph, weights, constraints)


# In[12]:

for d in duals:
    print d.dual, d.constraints


# Out[12]:

#     9.6 [<pydecode.hyper.Constraint object at 0x3af4130>, <pydecode.hyper.Constraint object at 0x3af4170>]
#     8.8 []
# 

# In[13]:

display.report(duals)


# Out[13]:

# image file:

# In[14]:

# Output the path.
for edge in gpath.edges:
    print hypergraph.label(edge)


# Out[14]:

#     ROOT -> D
#     D -> N
#     N -> V
#     V -> D
#     D -> D
#     D -> N
#     N -> END
# 

# In[15]:

print "check", constraints.check(gpath)
print "score", weights.dot(gpath)


# Out[15]:

#     check []
#     score 8.8
# 

# In[16]:

format = HMMFormat(hypergraph, [path, gpath])
display.to_ipython(hypergraph, format)


# Out[16]:

#     <IPython.core.display.Image at 0x3c60990>

# In[17]:

for constraint in constraints:
    print constraint.label


# Out[17]:

#     tag_D
#     tag_V
#     tag_N
# 

# In[18]:

class HMMConstraintFormat(display.HypergraphConstraintFormatter):
    def hypernode_attrs(self, node):
        label = self.hypergraph.node_label(node)
        return {"label": label.tag, "shape": ""}
    def hyperedge_node_attrs(self, edge):
        return {"color": "pink", "shape": "point"}
    def hypernode_subgraph(self, node):
        label = self.hypergraph.node_label(node)
        return ["cluster_" + str(label.position)]
    def subgraph_format(self, subgraph):
        return {"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])]}

format = HMMConstraintFormat(hypergraph, constraints)
display.to_ipython(hypergraph, format)


# Out[18]:

#     <IPython.core.display.Image at 0x4464c50>

# Pruning
# 

# In[22]:

pruned_hypergraph, pruned_weights = ph.prune_hypergraph(hypergraph, weights, 0.8)


# In[ ]:




# In[25]:

display.to_ipython(pruned_hypergraph, HMMFormat(pruned_hypergraph, []))


# Out[25]:

#     <IPython.core.display.Image at 0x44648d0>

# In[32]:

very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, weights, 0.9)


# Out[32]:


    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)

    <ipython-input-32-20046aa06d46> in <module>()
    ----> 1 very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, weights, 0.9)
    

    /home/srush/Projects/decoding/python/pydecode/hyper.so in pydecode.hyper.prune_hypergraph (python/pydecode/hyper.cpp:2145)()


    IndexError: list assignment index out of range

