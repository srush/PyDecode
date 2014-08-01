import pydecode
import random
import itertools
from collections import defaultdict
import numpy as np

def check_fully_connected(graph):
    """
    Checks that a hypergraph is fully-connected.

    Parameters
    ----------
    graph : Hypergraph
        Hypergraph to check.

    Returns
    --------
    checks : bool
       True, if the hypergraph is fully-connected.
    """
    seen_vertices = set()
    queue = [graph.root]
    while queue:
        cur_vertex = queue[0]
        seen_vertices.add(cur_vertex)
        queue = queue[1:]
        for edge in cur_vertex.edges:
            for child in edge.tail:
                queue.append(child)
    return len(seen_vertices) == len(graph.vertices), \
        "%d %d"%(len(seen_vertices), len(graph.vertices))


def check_reference_set_properties(graph, reference_sets):
    """
    Checks that a hypergraph satisifies reference set properties.

    Parameters
    ----------
    graph : Hypergraph
        Hypergraph to check.

    reference_sets : list
        The reference set associated with each vertex.

    Returns
    --------
    checks : bool
       True, if the hypergraph satisfies this reference set.
    """
    for vertex in graph.vertices:
        for edge in vertex.edges:
            children_set = set()
            for child in edge.tail:

                if not len(children_set & reference_sets) == 0:
                    return False

                children_set |= reference_sets[child.id]
            if not children_set.issubset(reference_sets[vertex.id]):
                return False

    full_set = set()
    for vertex_set in reference_sets:
        full_set |= vertex_set
    if full_set != reference_sets[graph.root.id]:
        return False

    return True


def simple_hypergraph():
    """
    Create a simple hypergraph.
    """
    enc = np.arange(6)
    c = pydecode.ChartBuilder(enc, np.arange(10))


    c.init(enc[:4])

    c.set(enc[4], enc[0:2], enc[1:3])
    c.set(enc[5], np.repeat(enc[4], 1), enc[[3]])

    dp = c.finish()
    # for edge in hypergraph.edges:
    #     assert edge.label in ["0", "1", "2", "3", "4"]
    return dp.hypergraph


import numpy as np
def random_hypergraph(size=50):
    """
    Generate a random hypergraph.

    Parameters
    ----------
    size : integer
    """
    # children = defaultdict(lambda: set())

    # complete_reference_set = range(0, size)
    reference_sets = defaultdict(lambda: set())
    enc = np.arange(2*size  + 1)

    c = pydecode.ChartBuilder(enc, np.arange(10))
    used = set()

    c.init(enc[:size])

    for i in range(size):
        reference_sets[i] = set([i])

    nodes = range(size)
    for node in range(size):
        head_node = size + node
        node_a, node_b = random.sample(nodes, 2)
        if reference_sets[node_a] & reference_sets[node_b]:
            continue

        c.set(enc[head_node], enc[[node_a]], enc[[node_b]])
        used.update([node_a, node_b])
        reference_sets[head_node] |= \
            reference_sets[node_a] | reference_sets[node_b]
        nodes.append(head_node)
    unused = set(nodes) -  used
    c.set(enc[2*size], enc[list(unused)])

    dp = c.finish()
    assert len(dp.hypergraph.nodes) > 0
    assert len(dp.hypergraph.edges) > 0
    return dp.hypergraph


def chain_hypergraph(size=100):
    """
    Returns a simple chain hypergraph.

    Returns
    --------
    graph : Hypergraph
       The chain hypergraph.
    """
    hypergraph = pydecode.Hypergraph()
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
    assert len(list(path.edges)) > 0
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
        assert check_fully_connected(h)
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
    paths = [pydecode.Path(graph, list(edges)) for edges in paths(graph.root)]
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
    return pydecode.Path(graph, edges)


# def random_inside_potentials(hypergraph):
#     return ph.InsidePotentials(hypergraph)\
#         .from_vector([random.random()
#                       for e in hypergraph.edges])

# def random_viterbi_potentials(hypergraph):
#     return ph.ViterbiPotentials(hypergraph)\
#         .from_vector([random.random()
#                       for e in hypergraph.edges])


# def random_log_viterbi_potentials(hypergraph):
#     return ph.LogViterbiPotentials(hypergraph)\
#         .from_vector([random.random()
#                       for e in hypergraph.edges])

# def random_log_viterbi_potentials_array(hypergraph):
#     return ph.LogViterbiPotentials(hypergraph)\
#         .from_array(np.random.rand(len(hypergraph.edges)))


# def random_bool_potentials(hypergraph):
#     return ph.BoolPotentials(hypergraph)\
#         .from_vector([random.random() > 0.5
#                       for e in hypergraph.edges])
