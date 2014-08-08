import os
import pydecode
import functools
from collections import defaultdict
import numpy as np
import random
import itertools
import numpy.testing

# def forall(tries=100, **kwargs):
#     def wrap(f):
#         @functools.wraps(f)
#         def wrapped(*inargs, **inkwargs):
#             for _ in xrange(tries):
#                 random_kwargs = (dict((name, gen.next())
#                                  for (name, gen) in kwargs.iteritems()))
#                 if forall.verbose or os.environ.has_key('QC_VERBOSE'):
#                     from pprint import pprint
#                     pprint(random_kwargs)
#                 random_kwargs.update(**inkwargs)
#                 f(*inargs, **random_kwargs)
#         return wrapped
#     return wrap
# forall.verbose = False # if enabled will print out the random test cases

assert_almost_equal = numpy.testing.assert_almost_equal

def _random_weighted_graphs(viterbi=False):
    while True:
        graph = random_hypergraph(20)
        weight_type = random_weight_type(viterbi)
        weights = random_weights(weight_type, len(graph.edges))
        yield graph, weights, weight_type


def property(tries=10, viterbi=False):
    def wrap(f):
        @functools.wraps(f)
        def wrapped(*inargs, **inkwargs):
            gen = _random_weighted_graphs(viterbi)
            for _ in xrange(tries):
                graph, weights, weight_type = gen.next()
                random_kwargs = {"graph": graph,
                                 "weights" : weights,
                                 "weight_type": weight_type}

                # random_kwargs = (dict((name, gen.next())
                #                  for (name, gen) in kwargs.iteritems()))
                # if forall.verbose or os.environ.has_key('QC_VERBOSE'):
                #     from pprint import pprint
                #     pprint(random_kwargs)
                random_kwargs.update(**inkwargs)
                f(*inargs, **random_kwargs)
        return wrapped
    return wrap

testWeights = [pydecode.Real,
               pydecode.Viterbi,
               pydecode.LogViterbi,
               pydecode.Log, pydecode.Boolean,
               pydecode.Counting]

testViterbi = [pydecode.LogViterbi]

def random_weight_type(viterbi = False):
    if not viterbi:
        return random.sample(testWeights, 1)[0]
    else:
        return random.sample(testViterbi, 1)[0]

def random_weights(weight_type, size):
    if weight_type == pydecode.Counting:
        return np.array(np.random.randint(100, size=size), dtype=np.int32)
    if weight_type == pydecode.Boolean:
        return np.array(np.random.randint(1, size=size), dtype=np.uint8)
    return np.random.random(size)


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

        c.set_t(enc[head_node], enc[[node_a]], enc[[node_b]],
                labels=np.array([random.randint(0, 100)]))
        used.update([node_a, node_b])
        reference_sets[head_node] |= \
            reference_sets[node_a] | reference_sets[node_b]
        nodes.append(head_node)
    unused = set(nodes) -  used
    c.set_t(enc[2*size], enc[list(unused)])

    dp = c.finish()
    assert len(dp.nodes) > 0
    assert len(dp.edges) > 0
    return dp

def _paths(node, bottom_vertex=None):
    if node.is_terminal:
        yield tuple()
    elif bottom_vertex is not None and node.id == bottom_vertex.id:
        yield tuple()
    else:
        for edge in node.edges:
            t = [_paths(node, bottom_vertex) for node in edge.tail]
            for below in itertools.product(*t):
                yield (edge,) + sum(below, ())

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
    paths = [pydecode.Path(graph, list(edges))
             for edges in _paths(graph.root)]
    return paths

def inside_paths(graph, vertex):
    paths = [pydecode.Path(graph, list(edges))
             for edges in _paths(vertex)]
    return paths

def outside_paths(graph, vertex):
    if vertex.id == graph.root.id: return [None]
    def has_vertex(path):
        for edge in path.edges:
            for tail in edge.tail:
                if vertex.id == tail.id:
                    return True
        return False

    paths = [pydecode.Path(graph, list(edges))
             for edges in _paths(graph.root, vertex)]
    paths = [path for path in paths if has_vertex(path)]
    return paths

def simple_hypergraph():
    """
    Create a simple hypergraph.
    """
    enc = np.arange(6)
    c = pydecode.ChartBuilder(enc, np.arange(10))

    c.init(enc[:4])

    c.set_t(enc[4], enc[0:2], enc[1:3], labels=np.arange(2))
    c.set_t(enc[5], np.repeat(enc[4], 1), enc[[3]], labels=np.array([2]))

    dp = c.finish()
    # for edge in hypergraph.edges:
    #     assert edge.label in ["0", "1", "2", "3", "4"]
    return dp


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
