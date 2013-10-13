
# A Constrained HMM Example
# ---------------------

# In[106]:

import pydecode.hyper as ph
import pydecode.display as display
from collections import namedtuple


# We begin by constructing the HMM probabilities.

# In[107]:

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

# In[108]:

class Bigram(namedtuple("Bigram", ["word", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)


# Now we are ready to build the  hypergraph topology itself.

# In[109]:

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

# In[110]:

def build_weights((word, tag, prev_tag)):
    return transition[prev_tag][tag] + emission[word][tag] 
weights = ph.Weights(hypergraph).build(build_weights)


# In[111]:

# Find the viterbi path.
path, chart = ph.best_path(hypergraph, weights)
print weights.dot(path)

# Output the path.
[hypergraph.label(edge) for edge in path.edges]


# Out[111]:

#     9.6
# 

#     [Bigram(word='the', tag='D', prevtag='ROOT'),
#      Bigram(word='dog', tag='N', prevtag='D'),
#      Bigram(word='walked', tag='V', prevtag='N'),
#      Bigram(word='in', tag='D', prevtag='V'),
#      Bigram(word='the', tag='N', prevtag='D'),
#      Bigram(word='park', tag='V', prevtag='N'),
#      Bigram(word='END', tag='END', prevtag='V')]

# In[112]:

format = display.HypergraphPathFormatter(hypergraph, [path])
display.to_ipython(hypergraph, format)


# Out[112]:

#     <IPython.core.display.Image at 0x48a0750>

# We can also use a custom fancier formatter. These attributes are from graphviz (http://www.graphviz.org/content/attrs)

# In[124]:

class HMMFormat(display.HypergraphPathFormatter):
    def hypernode_attrs(self, node):
        label = self.hypergraph.node_label(node)
        return {"label": label.tag, "shape": ""}
    def hyperedge_node_attrs(self, edge):
        return {"color": "pink", "shape": "point"}
    def hypernode_subgraph(self, node):
        label = self.hypergraph.node_label(node)
        return [("clust_" + str(label.position), label.tag)]
    def subgraph_format(self, subgraph):
        return {#"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])],
                "rank" : "same"}
    def graph_attrs(self): return {"rankdir":"RL"}
format = HMMFormat(hypergraph, [path])
display.to_ipython(hypergraph, format)


# Out[124]:

#     <IPython.core.display.Image at 0x56e3310>

# PyDecode also allows you to add extra constraints to the problem. As an example we can add constraints to enfore that the tag of "dog" is the same tag as "park".

# In[114]:

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

# In[115]:

print "check", constraints.check(path)


# Out[115]:

#     check ['tag_V', 'tag_N']
# 

# Solve instead using subgradient.

# In[116]:

gpath, duals = ph.best_constrained(hypergraph, weights, constraints)


# In[117]:

for d in duals:
    print d.dual, d.constraints


# Out[117]:

#     9.6 [<pydecode.hyper.Constraint object at 0x4d53190>, <pydecode.hyper.Constraint object at 0x4d539f0>]
#     8.8 []
# 

# In[118]:

display.report(duals)


# Out[118]:

# image file:

# In[119]:

# Output the path.
for edge in gpath.edges:
    print hypergraph.label(edge)


# Out[119]:

#     ROOT -> D
#     D -> N
#     N -> V
#     V -> D
#     D -> D
#     D -> N
#     N -> END
# 

# In[120]:

print "check", constraints.check(gpath)
print "score", weights.dot(gpath)


# Out[120]:

#     check []
#     score 8.8
# 

# In[121]:

format = HMMFormat(hypergraph, [path, gpath])
display.to_ipython(hypergraph, format)


# Out[121]:

#     <IPython.core.display.Image at 0x4d8ae10>

# In[122]:

for constraint in constraints:
    print constraint.label


# Out[122]:

#     tag_D
#     tag_V
#     tag_N
# 

# In[123]:

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


# Out[123]:


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-123-73aaf724ef31> in <module>()
         12 
         13 format = HMMConstraintFormat(hypergraph, constraints)
    ---> 14 display.to_ipython(hypergraph, format)
    

    /home/srush/Projects/decoding/python/pydecode/display.py in to_ipython(hypergraph, graph_format)
        113     from IPython.display import Image
        114     temp_file = "/tmp/tmp.png"
    --> 115     to_image(hypergraph, temp_file, graph_format)
        116     return Image(filename = temp_file)
        117 


    /home/srush/Projects/decoding/python/pydecode/display.py in to_image(hypergraph, filename, graph_format)
         83 
         84     for node in hypergraph.nodes:
    ---> 85         for sub, rank in graph_format.hypernode_subgraph(node):
         86             subgraphs.setdefault(sub, [])
         87             subgraphs[sub].append((node.id, rank))


    ValueError: too many values to unpack


# Pruning
# 

# In[ ]:

pruned_hypergraph, pruned_weights = ph.prune_hypergraph(hypergraph, weights, 0.8)


# In[ ]:




# In[ ]:

display.to_ipython(pruned_hypergraph, HMMFormat(pruned_hypergraph, []))


# In[ ]:

very_pruned_hypergraph, _ = ph.prune_hypergraph(hypergraph, weights, 0.9)

