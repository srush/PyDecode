#!/usr/bin/env python

import decoding_ext as d
import networkx as nx
import matplotlib.pyplot as plt

# The emission probabilities.
emission = {'the' : [('D', 0.8), ('N', 0.1), ('V', 0.1)],
		    'dog' : [('D', 0.1), ('N', 0.8), ('V', 0.1)],
			'walked' : [('V', 1)],
			'in' : [('D', 1)],
			'park' : [('N',0.1), ('V',0.9) ],
			'END' : [('END', 0)]}

# The transition probabilities.
transition = {'D' : {'D' : 0.1, 'N' : 0.8, 'V' : 0.1, 'END' : 0},
	   		  'N' : {'D' : 0.1, 'N' : 0.1, 'V' : 0.8, 'END' : 0},
			  'V' : {'D' : 0.4, 'N' : 0.3, 'V' : 0.3, 'END' : 0},
			  'ROOT' : {'D' : 0.4, 'N' : 0.3, 'V' : 0.3}}

# The sentence to be tagged.
sentence = 'the dog walked in the park'

node_tags = {}

hypergraph = d.HGraph()
constraints = d.HConstraints(hypergraph)

constraints_set = {}
for tag in ["D", "V", "N"]:
	constraint = constraints.add("tag_" + tag)
	constraint.set_constant(0)
	constraints_set[tag] = constraint

wb = d.WeightBuilder(hypergraph)
with hypergraph.builder() as b:
	node_start = b.add_terminal_node()
	node_tags[node_start] = "ROOT"
	node_list = []
	node_list.append(node_start)
	words = sentence.strip().split(" ")
	words.append("END")
	for word in words :
		current_node_list = []
		for tag_pair in emission[word]:
			with b.add_node() as node:
				node_tags[node] = tag_pair[0]
				for prev_node in node_list:
					edge = node.add_edge([prev_node], node_tags[prev_node] + "|" + tag_pair[0])
					if word == "dog":
						constraints_set[tag_pair[0]].add_edge_term(edge, 1)
					elif word == "park":
						constraints_set[tag_pair[0]].add_edge_term(edge, -1)
					wb.set_weight(edge, transition[node_tags[prev_node]][tag_pair[0]] + tag_pair[1])
			current_node_list.append(node)
		node_list = current_node_list

# Find the viterbi path.
weights = wb.weights()
path = d.viterbi(hypergraph, weights)
print weights.dot(path)

print "check", constraints.check(path)

# Output the path.
for edge in path.edges():
    print edge.label()

G = nx.Graph()

# Draw the graph using networkx.
for edge in hypergraph.edges():
    name = str(edge.id()) + " : " + edge.label()
    head = edge.head()
    G.add_edge(head.id(), name)
    for t in edge.tail():
        G.add_edge(name, t.id())
nx.draw_networkx(G)
plt.show()
plt.savefig("graph.png")
