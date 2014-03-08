from pydecode.hypergraph import *
from pydecode.potentials import *
#from pydecode.beam import *

from collections import defaultdict, namedtuple


class EdgeDesc(namedtuple("EdgeDesc", ("tail", "label"))):
    pass


def binarize(hypergraph):
    new_hyper = Hypergraph()
    d = {}
    with new_hyper.builder() as b:
        for node in hypergraph.nodes:
            if node.is_terminal:
                d[node.id] = b.add_node(label=node.label)
                continue
            new_edges = []
            for edge in node.edges:
                if len(edge.tail) <= 2:
                    new_edges.append(([d[tnode.id] for tnode in edge.tail],
                                      edge.label))
                    continue
                new_edge = [d[edge.tail[0].id], d[edge.tail[1].id]]
                new_node = b.add_node([(new_edge, None)])
                for tnode in edge.tail[2:-1]:
                    new_edge = [new_node, d[tnode.id]]
                    new_node = b.add_node([(new_edge, None)])
                new_edges.append(([new_node, d[edge.tail[-1].id]], edge.label))
            d[node.id] = b.add_node(new_edges, label=node.label)
    return new_hyper


def intersect(hypergraph, states, valid_states):
    """
    Intersect a hypergraph with a finite state automaton.
    """
    new_hyper = Hypergraph()
    d = {}
    seen = set()
    node_states = defaultdict(list)
    node_states_right = defaultdict(list)
    with new_hyper.builder() as b:
        for node in hypergraph.nodes:
            print node.id
            if node.is_terminal:
                for s1, s2 in valid_states(node.label):
                    d[node.id, s1, s2] = b.add_node(label=node.label)
                    node_states[node.id].append((s1, s2))
                    node_states_right[node.id, s1].append(s2)
                continue

            new_edges = defaultdict(list)
            for edge in node.edges:
                if len(edge.tail) == 1:
                    for s1, s2 in node_states[edge.tail[0].id]:
                        new_edges[s1, s2].append(([d[edge.tail[0].id, s1, s2]],
                                                  edge.label))
                else:
                    # key1 = (edge.tail[0].id, s1, s3)
                    # key2 = (edge.tail[1].id, s3, s2)
                    # if key1 in d and key2 in d:
                    for s1, s3 in node_states[edge.tail[0].id]:
                        for s2 in node_states_right[edge.tail[1].id, s3]:
                            a = [d[edge.tail[0].id, s1, s3],
                                 d[edge.tail[1].id, s3, s2]]
                            new_edges[s1, s2].append((a, edge.label))
            for (s1, s2), edges in new_edges.iteritems():
                d[node.id, s1, s2] = b.add_node(edges, label=node.label)
                node_states[node.id].append((s1, s2))
                node_states_right[node.id, s1].append(s2)
    return new_hyper
