
## Simple Hypergraph Example

# In[19]:

import pydecode.hyper as hyper
import pydecode.display as display
import networkx as nx 
import matplotlib.pyplot as plt 
from IPython.display import Image


# In[7]:

hyp = hyper.Hypergraph()
with hyp.builder() as b:
     n1 = b.add_node("first", terminal=True)
     n2 = b.add_node("second")
     b.add_edge(n2, [n1], label = "Edge")


# Draw the graph
# 

# In[23]:

G = display.to_networkx(hyp)
d = nx.drawing.to_agraph(G)
d.layout("dot")
d.draw("/tmp/tmp.png")
Image(filename ="/tmp/tmp.png")


# Out[23]:

#     <IPython.core.display.Image at 0x4755750>
