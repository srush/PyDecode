"""
Tests for hypergraph construction and basic data structures.
"""

import pydecode
import pydecode.test.utils as utils
import nose.tools as nt
from pydecode.test.utils import hypergraphs


def test_main():
    for graph in hypergraphs():
        # yield check_all_valid, graph
        yield check_numbering, graph
        yield check_hypergraph, graph
        assert utils.check_fully_connected(graph)

# def check_all_valid(graph):
#     for path in utils.all_paths(graph):
#         utils.valid_path(graph, path)


def check_numbering(graph):
    """
    Check that the numbering nodes and edges is correct.
    """
    for i, node in enumerate(graph.nodes):
        nt.assert_equal(node.id, i)
    for i, edge in enumerate(graph.edges):
        nt.assert_equal(edge.id, i)


def check_hypergraph(graph):
    """
    Check the assumptions about the hypergraph.
    """

    terminal = True
    children = set()

    # Check that terminal nodes are first.
    print len(graph.nodes)
    print len(graph.edges)
    for node in graph.nodes:
        print node.id

    for node in graph.nodes:
        if not terminal and len(node.edges) == 0:
            assert False
        if len(node.edges) != 0:
            terminal = False

        # Check ordering.
        for edge in node.edges:
            for tail_node in edge.tail:
                nt.assert_less(tail_node.id, node.id)
                children.add(tail_node.id)

    # Only 1 root.
    nt.assert_equal(len(children), len(graph.nodes) - 1)


# def check_serialization(graph):
#     s = pydecode.io.hypergraph_to_json(graph)
#     hyper2 = pydecode.io.json_to_hypergraph(s)
#     nt.assert_equal(len(graph.edges), len(hyper2.edges))
#     nt.assert_equal(len(graph.nodes), len(hyper2.nodes))


@nt.raises(Exception)
def test_diff_potentials_fail():
    h1, w1 = utils.random_hypergraph()
    h2, w2 = utils.random_hypergraph()
    pydecode.best_path(h1, w2)


@nt.raises(Exception)
def test_outside_fail():
    h1, w1 = utils.random_hypergraph()
    h2, w2 = utils.random_hypergraph()
    pydecode.outside_path(h1, w2)


@nt.raises(Exception)
def test_builder():
    h = pydecode.Hypergraph()
    b = h.builder()
    b.add_node([])


@nt.raises(Exception)
def test_bad_edge():
    h = pydecode.Hypergraph()
    with h.builder() as b:
        n1 = b.add_node()
        b.add_node(([n1],))

if __name__ == "__main__":
    for a in test_main():
        print a[0]
        a[0](*a[1:])
