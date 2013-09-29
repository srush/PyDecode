
## Simple Hypergraph Example

# In[2]:

import pydecode.hyper as hyper
import pydecode.display as display
import networkx as nx 
import matplotlib.pyplot as plt 
from IPython.display import Image


# In[3]:

hyp = hyper.Hypergraph()
with hyp.builder() as b:
     n1 = b.add_node()
     n2 = b.add_node((([n1], "Label"),))


# Draw the graph
# 

# In[5]:

G = display.to_networkx(hyp)
d = nx.drawing.to_agraph(G)
d.layout("dot")
d.draw("/tmp/tmp.png")
Image(filename ="/tmp/tmp.png")


# Out[5]:

#     <IPython.core.display.Image at 0x33b4f90>

# In[ ]:



