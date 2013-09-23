import decoding_ext as d
import networkx as nx
import matplotlib.pyplot as plt
hypergraph = d.HGraph()
d.HConstraints(hypergraph)
wb = d.WeightBuilder(hypergraph)
with hypergraph.builder() as b:
    node_a = b.add_terminal_node()

    with b.add_node() as node_b:
        edge = node_b.add_edge([node_a], "hello")
        wb.set_weight(edge, 10)

    with b.add_node() as node_c:
        edge = node_c.add_edge([node_b], "hello")
        wb.set_weight(edge, 15)
    print hypergraph.edges_size()
weights = wb.weights()
path = d.viterbi(hypergraph, weights)
print weights.dot(path)

for edge in path.edges():
    print edge.label()


G = nx.Graph()

for edge in hypergraph.edges():
    name = str(edge.id()) + " : " + edge.label()
    head = edge.head()
    G.add_edge(head.id(), name)
    for t in edge.tail():
        G.add_edge(name, t.id())
nx.draw(G)
plt.show()
