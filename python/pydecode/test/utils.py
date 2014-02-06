import pydecode.hyper as ph
import random
import itertools

def all_paths(graph):
    """
    Constructs all possible hyperpaths for testing.

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
    return ph.InsidePotentials(hypergraph).build(_rand_gen)

def random_viterbi_potentials(hypergraph):
    return ph.ViterbiPotentials(hypergraph).build(_rand_gen)

def random_log_viterbi_potentials(hypergraph):
    return ph.LogViterbiPotentials(hypergraph).build(_rand_gen)

def rand_bool_gen(arg=None):
    return random.random() > 0.5

def random_bool_potentials(hypergraph):
    return ph.BoolPotentials(hypergraph).build(rand_bool_gen)

def _rand_gen(arg=None):
    return random.random()
