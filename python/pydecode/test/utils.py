import pydecode.hyper as ph
import random
import itertools
from collections import defaultdict
from pydecode.hyper import EdgeDesc


def simple_hypergraph():
    """
    Create a simple fixed hypergraph.
    """
    hypergraph = ph.Hypergraph()
    with hypergraph.builder() as b:
        term = [b.add_node([], label="start " + str(i)) for i in range(4)]
        head_node = b.add_node([EdgeDesc([term[0], term[1]], "0"),
                                EdgeDesc([term[0]], "1")],
                               label="mid")
        head_node2 = b.add_node([EdgeDesc([head_node, term[2]], "2"),
                                 EdgeDesc([head_node, term[3]], "3"),
                                 EdgeDesc([head_node], "4")],
                                label="top")

    for edge in hypergraph.edges:
        assert edge.label in ["0", "1", "2", "3", "4"]
    return hypergraph


def random_hypergraph(size=50):
    """
    Generate a random hypergraph.

    Parameters
    ----------
    size : Int

    """
    hypergraph = ph.Hypergraph()
    children = defaultdict(lambda: set())
    with hypergraph.builder() as b:
        terminals = []
        for i in range(size):
            n = b.add_node()
            terminals.append(n)
            children[n.id] = set([n.id])
        nodes = list(terminals)
        for node in range(size):
            node_a, node_b = random.sample(nodes, 2)
            if len(children[node_a.id] & children[node_b.id]) > 0:
                continue
            head_node = b.add_node((([node_a, node_b], node),))
            children[head_node.id] = \
                set([head_node.id]) | children[node_a.id] | children[node_b.id]
            nodes.append(head_node)

    assert len(hypergraph.nodes) > 0
    assert len(hypergraph.edges) > 0
    return hypergraph


def chain_hypergraph(size=100):
    """
    Returns a simple chain hypergraph.

    Returns
    --------
    graph : Hypergraph
       The chain hypergraph.
    """
    hypergraph = ph.Hypergraph()
    with hypergraph.builder() as b:
        term = b.add_node()
        last_node = term
        for i in range(size):
            head_node = b.add_node([([last_node], "0")])
            last_node = head_node
    return hypergraph


def valid_path(hypergraph, path):
    """
    Check whether a path is valid for a given hypergraph.

    Parameters
    ------------
    graph : Hypergraph

    path : Path
    """
    root = hypergraph.root
    assert len(path.edges) > 0
    # Check there is a path to terminals.
    stack = [hypergraph.root]
    while stack:
        node = stack[0]
        stack = stack[1:]
        if node.is_terminal:
            continue
        count = 0
        for edge in node.edges:
            if edge in path:
                count += 1
                for tail_node in edge.tail:
                    stack.append(tail_node)
        assert count == 1,\
            " Count is {}. Path is {}".format(count,
                                              pretty_print_path(path))


def hypergraphs():
    """
    Returns several different small hypergraphs (for testing).
    """
    for i in range(10):
        h = random_hypergraph()
        yield h
    h = simple_hypergraph()
    yield h


def all_paths(graph):
    """
    Constructs all possible hyperpaths through a hypergraph (for testing).

    Parameters
    ------------
    graph : Hypergraph

    Returns
    ----------
    paths : list of hyperpaths
        All hyperpaths in the graph.

    """
    def paths(node):
        if node.is_terminal:
            yield tuple()
        else:
            for edge in node.edges:
                t = [paths(node) for node in edge.tail]
                for below in itertools.product(*t):
                    yield (edge,) + sum(below, ())
    paths = [ph.Path(graph, list(edges)) for edges in paths(graph.root)]
    return paths


def random_path(graph):
    """
    Constructs a randomly-generated valid hyperpath.

    Parameters
    ------------

    graph : Hypergraph

    Returns
    ----------
    path : Hyperpath
        A randomly generated hyperpath.

    """
    def random_path_edges(node):
        edges = []
        if node.edges:
            edge = random.sample(node.edges, 1)
            edges.append(edge[0])
            for t in edge[0].tail:
                edges += random_path_edges(t)
        return edges
    edges = random_path_edges(graph.root)
    return ph.Path(graph, edges)


def random_inside_potentials(hypergraph):
    return ph.InsidePotentials(hypergraph)\
        .from_vector([random.random()
                      for e in hypergraph.edges])


def random_viterbi_potentials(hypergraph):
    return ph.ViterbiPotentials(hypergraph)\
        .from_vector([random.random()
                      for e in hypergraph.edges])


def random_log_viterbi_potentials(hypergraph):
    return ph.LogViterbiPotentials(hypergraph)\
        .from_vector([random.random()
                      for e in hypergraph.edges])


def random_bool_potentials(hypergraph):
    return ph.BoolPotentials(hypergraph)\
        .from_vector([random.random() > 0.5
                      for e in hypergraph.edges])
