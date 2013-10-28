
# Building a Hypergraph
# ---------------------

# In[ ]:

import pydecode.hyper as ph


# In[ ]:

hyper1 = ph.Hypergraph()


# The code assumes that the hypergraph is immutable. The python interface enforces this by using a builder pattern. The important function to remember is add_node. 
# 
# * If there no arguments, then a terminal node is created. Terminal nodes must be created first.
# * If it is given an iterable, it create hyperedges to the new node. Each element in the iterable is a pair
#    * A list of tail nodes for that edge. 
#    * A label for that edge. 

# In[ ]:

with hyper1.builder() as b:
    node_a = b.add_node(label = "a")
    node_b = b.add_node(label = "b")
    node_c = b.add_node(label = "c")
    node_d = b.add_node(label = "d")
    node_e = b.add_node([([node_b, node_c], "First Edge")], label = "e")
    b.add_node([([node_a, node_e], "Second Edge"),
                ([node_a, node_d], "Third Edge")], label = "f")


# Outside of the `with` block the hypergraph is considered finished and no new nodes can be added. 

# We can also display the hypergraph to see our work.

# In[ ]:

import pydecode.display as display
display.HypergraphFormatter(hyper1).to_ipython()


# Out[]:

#     <IPython.core.display.Image at 0x3bf8110>

# After creating the hypergraph we can assign additional property information. One useful property is to add weights. We do this by defining a function to map labels to weights.

# In[ ]:

def build_weights(label):
    if "First" in label: return 1
    if "Second" in label: return 5
    if "Third" in label: return 5
    return 0
weights = ph.Weights(hyper1).build(build_weights)


# In[ ]:

for edge in hyper1.edges:
    print hyper1.label(edge), weights[edge]


# Out[]:

#     First Edge 1.0
#     Second Edge 5.0
#     Third Edge 5.0
# 

# We use the best path.

# In[ ]:

path = ph.best_path(hyper1, weights)


# In[ ]:

print weights.dot(path)


# Out[]:

#     6.0
# 

# In[ ]:

display.HypergraphFormatter(hyper1).to_ipython()


# Out[]:

#     <IPython.core.display.Image at 0x3be4d90>
