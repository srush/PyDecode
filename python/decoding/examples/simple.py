import decoding_ext as d
import networkx as nx
import matplotlib.pyplot as plt
import decoding.display as draw
hypergraph = d.HGraph()
constraints = d.HConstraints(hypergraph)

constraint_a = constraints.add("check_a")
constraint_a.set_constant(-1)

constraint_b = constraints.add("check_b")
constraint_b.set_constant(-2)

wb = d.WeightBuilder(hypergraph)
with hypergraph.builder() as b:
    node_a1 = b.add_terminal_node()
    node_a2 = b.add_terminal_node()

    with b.add_node() as node_b:
        edge = node_b.add_edge([node_a1], "hello")
        wb.set_weight(edge, 10)

        edge = node_b.add_edge([node_a2], "cool")
        constraint_a.add_edge_term(edge, 1)
        wb.set_weight(edge, 5)

    with b.add_node() as node_c:
        edge = node_c.add_edge([node_b], "good")
        constraint_b.add_edge_term(edge, 2)
        wb.set_weight(edge, 15)

    with b.add_node() as node_d:
        edge = node_d.add_edge([node_c], "bye")
        wb.set_weight(edge, 15)

print hypergraph.edges_size()
print hypergraph.nodes_size()

weights = wb.weights()

for i, node in enumerate(hypergraph.nodes()):
    assert node.id() == i
for i, edge in enumerate(hypergraph.edges()):
    assert edge.id() == i


G = draw.to_networkx(hypergraph)
nx.draw(G)
#plt.show()
# G = nx.Graph()

# for edge in hypergraph.edges():
#     name = str(edge.id()) + " : " + edge.label()
#     head = edge.head()
#     G.add_edge(head.id(), name)
#     for t in edge.tail():
#         G.add_edge(name, t.id())



path, chart = d.best_path(hypergraph, weights)
print weights.dot(path)

gpath = d.best_constrained(hypergraph, weights, constraints)

print "check", constraints.check(path)

for edge in path.edges():
    print edge.label()
