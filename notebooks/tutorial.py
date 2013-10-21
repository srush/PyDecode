
## Tutorial 

# Building a Hypergraph
# ---------------------

# In[9]:

import pydecode.hyper as ph


# In[10]:

hyper1 = ph.Hypergraph()


# The main data structure provided by pydecode is a hypergraph. Hypergraphs provide a graph-theoretical representation of dynamic programming problems. 

# The code assumes that the hypergraph is immutable. The python interface enforces this by using a builder pattern. The important function to remember is add_node. 
# 
# * If there no arguments, then a terminal node is created. Terminal nodes must be created first.
# * If it is given an iterable, it create hyperedges to the new node. Each element in the iterable is a pair
#    * A list of tail nodes for that edge. 
#    * A label for that edge. 

# In[11]:

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

# In[12]:

import pydecode.display as display
display.HypergraphFormatter(hyper1).to_ipython()


# Out[12]:

#     <IPython.core.display.Image at 0x3bf8110>

# After creating the hypergraph we can assign additional property information. One useful property is to add weights. We do this by defining a function to map labels to weights.

# In[13]:

def build_weights(label):
    if "First" in label: return 1
    if "Second" in label: return 5
    if "Third" in label: return 5
    return 0
weights = ph.Weights(hyper1).build(build_weights)


# In[14]:

for edge in hyper1.edges:
    print hyper1.label(edge), weights[edge]


# Out[14]:

#     First Edge 1.0
#     Second Edge 5.0
#     Third Edge 5.0
# 

# We use the best path.

# In[15]:

path = ph.best_path(hyper1, weights)


# In[16]:

print weights.dot(path)


# Out[16]:

#     6.0
# 

# In[17]:

display.HypergraphFormatter(hyper1).to_ipython()


# Out[17]:

#     <IPython.core.display.Image at 0x3be4d90>

# Hypergraph for Dynamic Programming
# ----------------------------------
# 
# The next question is how we might use this in practice.

# Here is a simple dynamic programming example take from wikipedia:
# 
#     int LevenshteinDistance(string s, string t)
#     {
#       int len_s = length(s);
#       int len_t = length(t);
# 
#       /* test for degenerate cases of empty strings */
#       if (len_s == 0) return len_t;
#       if (len_t == 0) return len_s;
# 
#       /* test if last characters of the strings match */
#       if (s[len_s-1] == t[len_t-1]) cost = 0;
#       else                          cost = 1;
# 
#       /* return minimum of delete char from s, delete char from t, and delete char from both */
#       return minimum(LevenshteinDistance(s[0..len_s-1], t) + 1,
#                      LevenshteinDistance(s, t[0..len_t-1]) + 1,
#                      LevenshteinDistance(s[0..len_s-1], t[0..len_t-1]) + cost)
#     }

# In[18]:

def make_ld_hyper(s, t):
    ld_hyper = ph.Hypergraph()
    
    with ld_hyper.builder() as b:
        nodes = {}
        s2 = s + "$"
        t2 = t + "$"
        for i, s_char in enumerate(s2):
            for j, t_char in enumerate(t2):
                edges = [([nodes[k, l]], m) 
                         for k, l, m in [(i-1, j, "<"), (i, j-1, ">"), (i-1, j-1, "m")] 
                         if k >= 0 and l >= 0
                         if m != "m" or s2[k] == t2[l]]
                nodes[i, j] = b.add_node(edges, label=str((s2[:i], t2[:j])))
        b.add_node([([nodes[len(s2) - 1, len(t2) - 1]], "")])
    return ld_hyper


# In[19]:

hyper2 = make_ld_hyper("ab", "bb")
display.HypergraphFormatter(hyper2).to_ipython()


# Out[19]:

#     <IPython.core.display.Image at 0x4146790>

# In[20]:

def build_weights(label):
    if label in ["<", ">"]: return 0.0
    if label == "m": return 1.0
    return 0.0
weights2 = ph.Weights(hyper2).build(build_weights)


# In[21]:

path = ph.best_path(hyper2, weights2)
display.HypergraphPathFormatter(hyper2, [path]).to_ipython()


# Out[21]:


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-21-9d5702cb485c> in <module>()
    ----> 1 path, chart = ph.best_path(hyper2, weights2)
          2 display.HypergraphPathFormatter(hyper2, [path]).to_ipython()


    ValueError: too many values to unpack

