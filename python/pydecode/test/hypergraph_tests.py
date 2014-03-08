"""
Tests for hypergraph construction and basic data structures.
"""

import pydecode.hyper as ph
import pydecode.test.utils as utils
import pydecode.io
import nose.tools as nt
from pydecode.test.utils import hypergraphs
from collections import defaultdict


def test_main():
    for hypergraph in hypergraphs():
        yield check_all_valid, hypergraph
        yield check_numbering, hypergraph
        yield check_hypergraph, hypergraph
        yield check_serialization, hypergraph


def check_all_valid(hypergraph):
    for path in utils.all_paths(hypergraph):
        utils.valid_path(hypergraph, path)


def check_numbering(hypergraph):
    """
    Check that the numbering nodes and edges is correct.
    """
    for i, node in enumerate(hypergraph.nodes):
        nt.assert_equal(node.id, i)
    for i, edge in enumerate(hypergraph.edges):
        nt.assert_equal(edge.id, i)


def check_hypergraph(hypergraph):
    """
    Check the assumptions about the hypergraph.
    """

    root_count = 0
    terminal = True
    children = set()

    # Check that terminal nodes are first.
    print len(hypergraph.nodes)
    print len(hypergraph.edges)
    for node in hypergraph.nodes:
        print node.id

    for node in hypergraph.nodes:
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
    nt.assert_equal(len(children), len(hypergraph.nodes) - 1)


def check_serialization(hypergraph):
    s = pydecode.io.hypergraph_to_json(hypergraph)
    hyper2 = pydecode.io.json_to_hypergraph(s)
    nt.assert_equal(len(hypergraph.edges), len(hyper2.edges))
    nt.assert_equal(len(hypergraph.nodes), len(hyper2.nodes))


@nt.raises(Exception)
def test_diff_potentials_fail():
    h1, w1 = utils.random_hypergraph()
    h2, w2 = utils.random_hypergraph()
    ph.best_path(h1, w2)


@nt.raises(Exception)
def test_outside_fail():
    h1, w1 = utils.random_hypergraph()
    h2, w2 = utils.random_hypergraph()
    ph.outside_path(h1, w2)


@nt.raises(Exception)
def test_builder():
    h = ph.Hypergraph()
    b = h.builder()
    b.add_node([])


@nt.raises(Exception)
def test_bad_edge():
    h = ph.Hypergraph()
    with h.builder() as b:
        n1 = b.add_node()
        b.add_node(([n1],))


if __name__ == "__main__":
    test_variables()
