
# A Constrained HMM Example
# ---------------------

# In[3]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple


# We begin by constructing the HMM probabilities.

# In[5]:

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

# In[6]:

class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)


# Now we are ready to build the  hypergraph topology itself.

# In[7]:

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

# In[8]:

def build_weights((word, tag, prev_tag)):
    return transition[prev_tag][tag] + emission[word][tag]
weights = ph.Weights(hypergraph, build_weights)


# In[9]:

# Find the viterbi path.
path, chart = ph.best_path(hypergraph, weights)
print weights.dot(path)

# Output the path.
[hypergraph.label(edge) for edge in path.edges]


# Out[9]:

#     9.6
#

#     [Bigram(word='the', tag='D', prevtag='ROOT'),
#      Bigram(word='dog', tag='N', prevtag='D'),
#      Bigram(word='walked', tag='V', prevtag='N'),
#      Bigram(word='in', tag='D', prevtag='V'),
#      Bigram(word='the', tag='N', prevtag='D'),
#      Bigram(word='park', tag='V', prevtag='N'),
#      Bigram(word='END', tag='END', prevtag='V')]

# In[13]:

format = display.HypergraphPathFormatter(hypergraph, path)
display.to_ipython(hypergraph, format)


# Out[13]:

#     <IPython.core.display.Image at 0x4860410>

# We can also use a custom fancier formatter. These attributes are from graphviz (http://www.graphviz.org/content/attrs)

# In[10]:

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

format = HMMFormat(hypergraph, path)
display.to_ipython(hypergraph, format)


# Out[10]:

#     <IPython.core.display.Image at 0x3b79f50>

# PyDecode also allows you to add extra constraints to the problem. As an example we can add constraints to enfore that the tag of "dog" is the same tag as "park".

# In[9]:


def cons(tag): return "tag_%s"%tag

def build_constraints(bigram):
    if bigram.word == "dog":
        return [(cons(bigram.tag), 1)]
    elif bigram.word == "park":
        return [(cons(bigram.tag), -1)]
    return []

constraints =     ph.Constraints(hypergraph,
                   [(cons(tag), 0) for tag in ["D", "V", "N"]],
                   build_constraints)


# This check fails because the tags do not agree.

# In[10]:

print "check", constraints.check(path)


# Out[10]:

#     check ['tag_V', 'tag_N']
#

# Solve instead using subgradient.

# In[*]:

gpath, duals = ph.best_constrained(hypergraph, weights, constraints)


# In[22]:

display.report(duals)

# Out[22]:

#     <matplotlib.axes.AxesSubplot at 0x41fb9d0>

# image file:

# In[ ]:

# Output the path.
# for edge in gpath.edges():
#     print hypergraph.label(edge)


# In[15]:

# print "check", constraints.check(gpath)
# print "score", weights.dot(gpath)


# Out[15]:

#     check ['tag_V', 'tag_N']
#     score 7.8
#

# In[16]:

#display.to_ipython(hypergraph, paths=[path, gpath])


# Out[16]:

#     <IPython.core.display.Image at 0x41e5950>
