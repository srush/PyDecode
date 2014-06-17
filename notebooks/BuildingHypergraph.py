
## Hypergraph Interface

# In[8]:

import pydecode.hyper as ph


# In[9]:

hyper1 = ph.Hypergraph()


# The code assumes that the hypergraph is immutable. The python interface enforces this by using a builder pattern. The important function to remember is add_node. 
# 
# * If there no arguments, then a terminal node is created. Terminal nodes must be created first.
# * If it is given an iterable, it create hyperedges to the new node. Each element in the iterable is a pair
#    * A list of tail nodes for that edge. 
#    * A label for that edge. 

# In[10]:

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

# In[11]:

import pydecode.display as display
display.HypergraphFormatter(hyper1).to_ipython()


# Out[11]:

#     <IPython.core.display.Image at 0x2b86b10>

# After creating the hypergraph we can assign additional property information. One useful property is to add potentials. We do this by defining a function to map labels to potentials.

# In[12]:

def build_potentials(label):
    if "First" in label: return 1
    if "Second" in label: return 5
    if "Third" in label: return 5
    return 0
potentials = ph.LogViterbiPotentials(hyper1).from_vector((build_potentials(edge.label) 
                                                for edge in hyper1.edges))


# In[13]:

for edge in hyper1.edges:
    print edge.label, potentials[edge]


# Out[13]:

#     First Edge 1.0
#     Second Edge 5.0
#     Third Edge 5.0
# 

# We use the best path.

# In[14]:

path = ph.best_path(hyper1, potentials)


# In[15]:

print potentials.dot(path)


# Out[15]:

#     6.0
# 

# In[16]:

display.HypergraphFormatter(hyper1).to_ipython()


# Out[16]:

#     <IPython.core.display.Image at 0x269ef10>
