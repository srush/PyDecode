import pydecode.hyper as hyper
import random
import pydecode.display as draw
import networkx as nx
import matplotlib.pyplot as plt
import nose.tools as nt

def simple_hypergraph():
    hypergraph = hyper.Hypergraph()

    with hypergraph.builder() as b:
        term = [b.add_node() for i in range(3)]
        head_node = b.add_node([([term[0], term[1]], "")])
        head_node2 = b.add_node([([head_node, term[2]], "")])
    weights = hyper.Weights(hypergraph).build(lambda t: random.random())
    return hypergraph, weights

def random_hypergraph():
    "Generate a random hypergraph."
    hypergraph = hyper.Hypergraph()

    with hypergraph.builder() as b:
        terminals = [b.add_node() for i in range(10)]
        nodes = list(terminals)
        for node in range(10):
            node_a, node_b = random.sample(nodes, 2)
            head_node = b.add_node((([node_a, node_b], node),))
            nodes.append(head_node)
    nt.assert_greater(len(hypergraph.nodes), 0)
    assert len(hypergraph.edges) > 0

    weights = hyper.Weights(hypergraph).build(lambda t: random.random())
    return hypergraph, weights

def test_numbering():
    for hypergraph, _ in [random_hypergraph() for i in range(10)]:
        for i, node in enumerate(hypergraph.nodes):
            nt.assert_equal(node.id, i)
        for i, edge in enumerate(hypergraph.edges):
            nt.assert_equal(edge.id, i)

def valid_hypergraph(hypergraph):
    root_count = 0
    terminal = True
    children = set()

    # Check that terminal nodes are first.
    print len(hypergraph.nodes)
    print len(hypergraph.edges)
    for node in hypergraph.nodes:
        print node.id

    for node in hypergraph.nodes:
        print node.id
        if terminal == False and len(node.edges) == 0:
            assert False
        if len(node.edges) != 0:
            terminal = False

        # Check ordering.
        for edge in node.edges:
            for tail_node in edge.tail:
                nt.assert_less(tail_node.id, node.id)
                children.add(tail_node.id)

    # Only 1 root.
    nt.assert_equal(len(children), len(hypergraph.nodes) - 1)


def test_simple_valid():
    valid_hypergraph(simple_hypergraph()[0])

def test_valid():
    for hypergraph, w in [random_hypergraph() for i in range(10)]:
        valid_hypergraph(hypergraph)

def valid_path(hypergraph, path):
    "Check whether a path is valid."
    root = hypergraph.root
    nt.assert_greater(len(path.edges), 0)
    # Check there is a path to terminals.
    stack = [hypergraph.root]
    while stack:
        node = stack[0]
        stack = stack[1:]
        if node.is_terminal(): continue
        count = 0
        for edge in node.edges:
            if edge in path:
                count += 1
                for tail_node in edge.tail:
                    stack.append(tail_node)
        assert count == 1, " Count is {}. Path is {}".format(count, pretty_print_path(path))

def test_construction():
    h = random_hypergraph()

def test_inside():
    for h, w in [random_hypergraph() for i in range(10)]:
        path, chart = hyper.best_path(h, w)
        nt.assert_not_equal(w.dot(path), 0.0)

        valid_path(h, path)

def test_outside():
    for h, w in [random_hypergraph() for i in range(10)]:
        path, chart = hyper.best_path(h, w)
        best = w.dot(path)
        nt.assert_not_equal(best, 0.0)
        out_chart = hyper.outside_path(h, w, chart)
        for node in h.nodes:
            other = chart[node] + out_chart[node]
            nt.assert_less_equal(other, best  + 1e-4)
            if node.is_terminal():
                nt.assert_almost_equal(other, best)

def test_maxmarginals():
    for h, w in [random_hypergraph() for i in range(10)]:
        path, chart = hyper.best_path(h, w)
        best = w.dot(path)
        nt.assert_not_equal(best, 0.0)
        max_marginals = hyper.compute_max_marginals(h, w)
        for node in h.nodes:
            other = max_marginals.node_marginal(node)
            nt.assert_less_equal(other, best + 1e-4)

        for edge in h.edges:
            other = max_marginals.edge_marginal(edge)
            nt.assert_less_equal(other, best + 1e-4)
            if edge in path:
                nt.assert_almost_equal(other, best)

def random_constraint(hypergraph):
    edge, = random.sample(hypergraph.edges, 1)
    def build_constraints(label):
        l = hypergraph.label(edge)
        if label == l:
            return [("have", 1), ("not", 1)]
        return []
    constraints = hyper.Constraints(hypergraph).build(
                                    [("have", -1), ("not", 0)],
                                    build_constraints)
    return constraints, edge

def test_constraint():
    for h, w in [random_hypergraph() for i in range(10)]:
        constraints, edge = random_constraint(h)
        path, chart = hyper.best_path(h, w)
        match = constraints.check(path)
        if edge not in path:
            nt.assert_equal(match[0], "have")
        else:
            nt.assert_equal(match[0], "not")


def test_pruning():
    for h, w in [random_hypergraph() for i in range(10)]:
        new_hyper, new_weights = hyper.prune_hypergraph(h, w, 0.9)
        path, chart = hyper.best_path(new_hyper, new_weights)
        valid_path(new_hyper, path)

if __name__ == "__main__":
    test_inside()
    test_pruning()
