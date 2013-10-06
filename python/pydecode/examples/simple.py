import pydecode.hyper as hyper
import networkx as nx
import matplotlib.pyplot as plt
import pydecode.display as draw

hypergraph = hyper.Hypergraph()

# Build graph
with hypergraph.builder() as b:
    node_a1 = b.add_node()
    node_a2 = b.add_node()
    node_b = b.add_node((([node_a1], "1"),
                         ([node_a2], "2")))
    node_c = b.add_node((([node_b], "3"),))
    node_d = b.add_node((([node_c], "4"),))


# Specify the weights.
def weights(t): return {"1": 5,"2": 10, "3": 15, "4": 15}[t]
weights = hyper.Weights(hypergraph, weights)

# Specify the constraints.
constraints = hyper.Constraints(hypergraph)

def edges_a(t):
    if t in ["2", "4"]: return 1
    return 0
constraints.add("constraint_a", edges_a, -1)

def edges_b(t):
    if t == "3": return 2
    return 0
constraints.add("constraint_a", edges_b, -2)

G = draw.to_networkx(hypergraph)
nx.draw(G)

path, chart = hyper.best_path(hypergraph, weights)
print weights.dot(path)

gpath, output = hyper.best_constrained(hypergraph, weights, constraints)

print "check", constraints.check(path)
for edge in path.edges:
    print edge.label

for result in output:
    print result.dual
    for a in result.constraints:
        print a
