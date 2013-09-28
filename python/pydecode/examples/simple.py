import pydecode.hyper as hyper
import networkx as nx
import matplotlib.pyplot as plt
import pydecode.display as draw

hypergraph = hyper.Hypergraph()
weights = hyper.Weights(hypergraph)

constraints = hyper.Constraints(hypergraph)
constraint = [constraints.add("check_a", -1),
              constraints.add("check_b", -2)]

with hypergraph.builder() as b:
    node_a1 = b.add_node(terminal = True)
    node_a2 = b.add_node(terminal = True)
    
    node_b = b.add_node()
    edge = b.add_edge(node_b, [node_a1],
                      weight = 5, 
                      constraints = [(constraint[0], 1)])

    b.add_edge(node_b, [node_a2], weight = 10)
    
    node_c = b.add_node()
    b.add_edge(node_c, [node_b], weight = 15, 
               constraints = [(constraint[1], 2)])

    node_d = b.add_node()
    b.add_edge(node_d, [node_c], weight = 15)

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
#     G.add_edge(heahyper.id(), name)
#     for t in edge.tail():
#         G.add_edge(name, t.id())

path, chart = hyper.best_path(hypergraph, weights)
print weights.dot(path)

gpath = hyper.best_constrained(hypergraph, weights, constraints)

print "check", constraints.check(path)

for edge in path.edges():
    print edge.label()
