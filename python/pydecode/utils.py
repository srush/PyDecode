import pydecode.hyper as ph
import random
import itertools
import json

def hypergraph_to_json(graph):
    data = []
    for node in graph.nodes:
        if not node.edges:
            data.append([])
        else:
            data.append([([tail.id for tail in edge.tail], edge.label)
                         for edge in node.edges])
    return json.dumps(data)

def json_to_hypergraph(s):
    hypergraph = ph.Hypergraph()
    nodes = {}
    
    with hypergraph.builder() as b:
        for i, edge_ls in enumerate(json.loads(s)):
            if not edge_ls:
                nodes[i] = b.add_node()
            else:
                nodes[i] = b.add_node(
                    [([nodes[node_id] for node_id in edge], lab) 
                     for edge, lab in edge_ls])
    return hypergraph

def all_paths(graph):
    """
    Gets all possible hyperpaths.    
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
    Gets a random valid hyperpath.
    """
    def random_path_edges(node):
        edges = []
        if node.edges:
            edge = random.sample(node.edges, 1)
            edges.append(edge)
            for t in edge.tail:
                edges += random_path_edges(t)
        return edges
    edges = random_path_edges(graph.root)
    return ph.Path(graph, edges)

def random_inside_potentials(hypergraph):
    return ph.InsidePotentials(hypergraph).build(rand_gen)

def random_viterbi_potentials(hypergraph):
    return ph.ViterbiPotentials(hypergraph).build(rand_gen)

def random_log_viterbi_potentials(hypergraph):
    return ph.LogViterbiPotentials(hypergraph).build(rand_gen)

def rand_bool_gen(arg=None):
    return random.random() > 0.5

def random_bool_potentials(hypergraph):
    return ph.BoolPotentials(hypergraph).build(rand_bool_gen)

def rand_gen(arg=None):
    return random.random()
