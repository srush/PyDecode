import pydecode.hyper as ph
import random
import itertools


def hypergraph_to_json(graph):
    """
    Converts a hypergraph to a JSON encodable data structure.

    Parameters
    ------------
    graph : hypergraph
        The hypergraph to encode.

    Returns
    -------
    A JSON encodable object.

    """

    data = []
    for node in graph.nodes:
        if not node.edges:
            data.append([])
        else:
            data.append([([tail.id for tail in edge.tail], edge.label)
                         for edge in node.edges])
    return data


def json_to_hypergraph(obj):
    """
    Parameters
    -----------
    obj : An object returned by hypergraph_to_json

    Returns
    --------
    graph : Hypergraph
    """

    hypergraph = ph.Hypergraph()
    nodes = {}

    with hypergraph.builder() as b:
        for i, edge_ls in enumerate(obj):
            if not edge_ls:
                nodes[i] = b.add_node()
            else:
                nodes[i] = b.add_node(
                    [([nodes[node_id] for node_id in edge], lab)
                     for edge, lab in edge_ls])
    return hypergraph


def json_to_potentials(s, potentials,
                       potential_type,
                       val_convert=lambda a: a):
    data = json.loads(s)
    return potential_type.from_vector(
        [val_convert(value) for value in data["values"]],
        val_convert(data["bias"]))


def potentials_to_json(graph, potentials,
                       val_convert=lambda a: a):
    data = {"values": [val_convert(potentials.score(edge))
                       for edge in graph.edges],
            "bias": val_convert(potentials.bias)}
    return json.dumps(data)
