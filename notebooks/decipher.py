
## A Decipherment Example

# This is a note on running decipherment.

# In[101]:

from nltk.util import ngrams
from nltk.model.ngram import NgramModel
from nltk.probability import LidstoneProbDist
import random, math

class Problem:
    def __init__(self, corpus):
        self.t = corpus
        t = list(self.t)
        est = lambda fdist, bins: LidstoneProbDist(fdist, 0.0001)
        self.lm = NgramModel(2, t, estimator = est)

        self.letters = set(t) #[chr(ord('a') + i) for i in range(26)]
        self.letters.remove(" ")
        shuffled = list(self.letters)
        random.shuffle(shuffled)
        self.substitution_table = dict(zip(self.letters, shuffled))
        self.substitution_table[" "] = " "

    def make_cipher(self, plaintext):
        self.ciphertext = "".join([self.substitution_table[l] for l in plaintext])
        self.plaintext = plaintext
simple_problem = Problem("ababacabac ")
simple_problem.make_cipher("abac")


# In[102]:

import pydecode.hyper as hyper
import pydecode.display as display
import pydecode.constraints as cons
from collections import namedtuple        


# In[103]:

class Conversion(namedtuple("Conversion", ["i", "cipherletter", "prevletter", "letter"])):
    __slots__ = ()
    def __str__(self):
        return "%s %s %s"%(self.cipherletter, self.prevletter, self.letter)
class Node(namedtuple("Node", ["i", "cipherletter", "letter"])):
    __slots__ = ()
    def __str__(self):
        return "%s %s %s"%(self.i, self.cipherletter, self.letter)


# In[104]:

def build_cipher_graph(problem):
    ciphertext = problem.ciphertext
    letters = problem.letters
    hypergraph = hyper.Hypergraph()
    with hypergraph.builder() as b:
        prev_nodes = [(" ", b.add_node([], label=Node(-1, "", "")))]
        for i, c in enumerate(ciphertext):
            nodes = []
            possibilities = letters
            if c == " ": possibilities = [" "]
            for letter in possibilities:
                edges = [([prev_node], Conversion(i, c, old_letter, letter))
                         for (old_letter, prev_node) in prev_nodes]
                
                node = b.add_node(edges, label = Node(i, c, letter))
                nodes.append((letter, node))
            prev_nodes = nodes
        letter = " "
        final_edges = [([prev_node], Conversion(i, c, old_letter, letter))
                       for (old_letter, prev_node) in prev_nodes]
        b.add_node(final_edges, label=Node(len(ciphertext), "", ""))
    return hypergraph


# In[105]:

hyper1 = build_cipher_graph(simple_problem)


# In[106]:

class CipherFormat(display.HypergraphPathFormatter):
    def hypernode_attrs(self, node):
        return {"label": "%s -> %s"%(node.label.cipherletter, node.label.letter)}
    def hyperedge_node_attrs(self, edge):
        return {"color": "pink", "shape": "point"}
    def hypernode_subgraph(self, node):
        return [("cluster_" + str(node.label.i), node.label.i)]
    # def subgraph_format(self, subgraph):
    #     return {"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])]}

CipherFormat(hyper1, []).to_ipython()


# Out[106]:

#     <IPython.core.display.Image at 0x33a4f90>

# In[106]:




# Constraint is that the sum of edges with the conversion is equal to the 0.
# 
# l^2 constraints

# In[107]:

def build_constraints(hypergraph, problem):
    ciphertext = problem.ciphertext
    letters = problem.letters
    def transform(from_l, to_l): return "letter_%s_from_letter_%s"%(to_l, from_l)
    constraints = cons.Constraints(hypergraph, [(transform(l, l2), 0)
                       for l  in letters 
                       for l2 in letters])

    first_position = {}
    count = {}
    for i, l in enumerate(ciphertext):
        if l not in first_position:
            first_position[l] = i
        count.setdefault(l, 0)
        count[l] += 1
    def build(conv):
        l = conv.cipherletter
        if l == " " or l == "": return []
        if conv.letter == " ": return []
        if first_position[l] == conv.i:
            return [(transform(conv.cipherletter, conv.letter), count[l] - 1)]
        else:
            return [(transform(conv.cipherletter, conv.letter), -1)]
    constraints.from_vector([build(edge.label) for edge in hypergraph.edges])
    return constraints

constraints = build_constraints(hyper1, simple_problem)


# In[108]:

def build_potentials(edge):
    return random.random()
potentials = hyper.Potentials(hyper1).from_vector([build_potentials(node.label) 
                                                   for node in hyper1.nodes])


# In[109]:

for edge in hyper1.edges:
    print potentials[edge]


# Out[109]:

#     0.18738485935
#     0.472104200234
#     0.601083619451
#     0.926199823358
#     0.472726063032
#     0.272371839332
#     0.6717332954
#     0.919803900784
#     0.400676874512
#     0.319032345118
#     0.0858441054396
#     0.17732106776
#     0.166267944361
#     0.677231007635
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
#     0.0
# 

# In[110]:

path = hyper.best_path(hyper1, potentials)
potentials.dot(path)


# Out[110]:

#     2.0691391086535877

# In[111]:

import pydecode.optimization as opt
cpath = opt.best_constrained_path(hyper1, potentials, constraints)


# In[112]:

CipherFormat(hyper1, [cpath]).to_ipython()


# Out[112]:

#     <IPython.core.display.Image at 0x34180d0>

# In[113]:

print potentials.dot(cpath)
constraints.check(cpath)


# Out[113]:

#     0.0
# 

#     []

# Real Problem

# In[114]:

complicated_problem = Problem("this is the president calling blah blah abadadf adfadf")
complicated_problem.make_cipher("this is the president calling")


# In[115]:

hyper2 = build_cipher_graph(complicated_problem)


# In[116]:


potentials2 = hyper.Potentials(hyper2).from_vector([math.log(complicated_problem.lm.prob(edge.label.letter, edge.label.prevletter)) for edge in hyper2.edges])


# In[117]:

print len(hyper2.edges)


# Out[117]:

#     4650
# 

# In[118]:

path2 = hyper.best_path(hyper2, potentials2)

for edge in path2.edges:
    print edge.id
    print potentials2[edge]
potentials2.dot(path2)


# Out[118]:

#     11
#     -2.07941654387
#     221
#     0.0
#     298
#     0.0
#     648
#     -1.09861228867
#     702
#     -0.405481773803
#     709
#     -1.45088787965
#     814
#     -0.510852289188
#     951
#     -0.69314718056
#     971
#     -2.07941654387
#     1181
#     0.0
#     1258
#     0.0
#     1428
#     -1.09861228867
#     1451
#     -2.07941654387
#     1661
#     0.0
#     1738
#     0.0
#     1908
#     -0.693234675638
#     2190
#     -0.693172179622
#     2449
#     -0.510852289188
#     2586
#     -0.69314718056
#     2865
#     -0.693172179622
#     3124
#     -0.510852289188
#     3261
#     -0.69314718056
#     3281
#     -2.07941654387
#     3491
#     0.0
#     3568
#     0.0
#     3888
#     -1.09861228867
#     3970
#     -0.693234675638
#     4245
#     -0.693172179622
#     4504
#     -0.510852289188
#     4641
#     -0.69314718056
# 

#     -21.751856464057795

# In[119]:

# new_hyper, new_potentials = hyper.prune_hypergraph(hyper2, potentials2, 0.2)
# constraints2 = build_constraints(new_hyper, complicated_problem)


# In[120]:

# print hyper2.edges_size
# new_hyper.edges_size


# In[121]:

# display.report(duals)


# In[122]:

# path2, duals = hyper.best_constrained(new_hyper, new_potentials, constraints2)


# Potentials are the bigram language model scores.

# In[123]:

# path2 = hyper.best_path(hyper2, potentials2)
# print potentials2.dot(path2)
# for edge in path2.edges:
#     print hyper2.label(edge).letter, 

