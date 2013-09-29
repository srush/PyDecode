import pydecode.hyper as hyper
import random
import pydecode.display as draw
import networkx as nx 
import matplotlib.pyplot as plt


def random_hypergraph():
    "Generate a random hypergraph."
    hypergraph = hyper.Hypergraph()    
    
    with hypergraph.builder() as b:
        terminals = [b.add_node() for i in range(10)]
        nodes = list(terminals)
        for node in range(10):
            node_a, node_b = random.sample(nodes, 2)
            head_node = b.add_node((([node_a, node_b], ""),))
            nodes.append(head_node)
    assert len(hypergraph.nodes()) > 0
    assert len(hypergraph.edges()) > 0

    weights = hyper.Weights(hypergraph, lambda t: random.random())
    return hypergraph, weights

def test_numbering():
    for hypergraph, _ in [random_hypergraph() for i in range(10)]:
        for i, node in enumerate(hypergraph.nodes()):
            assert node.id() == i
        for i, edge in enumerate(hypergraph.edges()):
            assert edge.id() == i

def valid_hypergraph(hypergraph):
    root_count = 0 
    terminal = True
    
    children = set()
    # Check that terminal nodes are first.
    for node in hypergraph.nodes():
        if terminal == False and len(node.edges()) == 0:
            assert False
        if len(node.edges()) != 0:
            terminal = False

        # Check ordering.
        for edge in node.edges():
            for tail_node in edge.tail():
                assert tail_node.id() < node.id() 
                children.add(tail_node.id())

    # Only 1 root.  
    assert len(children) == len(hypergraph.nodes()) - 1

def test_valid():
    for hypergraph, w in [random_hypergraph() for i in range(10)]:
        valid_hypergraph(hypergraph)

def valid_path(hypergraph, path):
    "Check whether a path is valid."
    root = hypergraph.root()
    assert len(path.edges()) > 0
    # Check there is a path to terminals.
    stack = [hypergraph.root()]
    while stack:
        node = stack[0]
        stack = stack[1:]
        if node.is_terminal(): continue
        count = 0
        for edge in node.edges():
            if edge in path:
                count += 1
                for tail_node in edge.tail():
                    stack.append(tail_node)
        assert count == 1, " Count is {}. Path is {}".format(count, pretty_print_path(path))

def test_construction():
    h = random_hypergraph()

def test_inside():
    for h, w in [random_hypergraph() for i in range(10)]:
        path, chart = hyper.best_path(h, w)
        assert w.dot(path) != 0.0        

        valid_path(h, path)

def test_outside():
    for h, w in [random_hypergraph() for i in range(10)]:
        path, chart = hyper.best_path(h, w)
        best = w.dot(path) 
        assert best != 0.0
        out_chart = hyper.outside_path(h, w, chart)
        for node in h.nodes():
            other = chart.score(node) + out_chart.score(node)
            assert other <= best + 1e-4, \
                "Best: {}. Other: {}".format(other, best)

            if node.is_terminal():
                assert abs(other - best) < 1e-4, \
                    "Best: {}. Other: {}".format(other, best)
    
def random_constraint(hypergraph, constraints):
    constrainta = constraints.add("have", lambda t : 0 , -1)
    constraintb = constraints.add("not", lambda t : 0 , 0)
    edge = random.sample(hypergraph.edges(), 1)
    return edge[0]

def test_constraint():
    for h, w in [random_hypergraph() for i in range(10)]:
        constraints = hyper.Constraints(h)
        edge = random_constraint(h, constraints)
        path, chart = hyper.best_path(h, w)
        match = constraints.check(path)
        # if edge not in path:
        #     assert match[0] == "have"
        # else: 
        #     assert match[0] == "not"

if __name__ == "__main__":
    test_inside()
    test_constraint()
    
