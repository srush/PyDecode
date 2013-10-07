
## Simple Hypergraph Example

# In[15]:

import pydecode.hyper as ph
import pydecode.display as display


# In[27]:

hyp = ph.Hypergraph()
with hyp.builder() as b:
     n1 = b.add_node(label = "a")
     n2 = b.add_node(label = "b")
     n3 = b.add_node(label = "c")
     n4 = b.add_node(label = "d")
     n5 = b.add_node((([n1, n2], "edge1"),), label = "e")
     b.add_node([([n5], "edge3"), ([n3, n4], "edge2")], label = "root")

def build_weights(label):
     return {"edge1" : 3, "edge2" : 1, "edge3" : 1}[label]
weights = ph.Weights(hyp, build_weights)


# Draw the graph

# In[28]:

display.to_ipython(hyp, extra=[weights])


# Out[28]:

#     <IPython.core.display.Image at 0x3503790>

# In[30]:

path, _ = ph.best_path(hyp, weights)
display.to_ipython(hyp, paths=[path])


# Out[30]:

#     <IPython.core.display.Image at 0x3503e10>
