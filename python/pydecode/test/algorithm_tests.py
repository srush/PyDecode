import pydecode
import pydecode.test.utils as utils
from collections import defaultdict
import nose.tools as nt
import numpy as np
import numpy.random
import numpy.testing

def test_main():
    for hypergraph in utils.hypergraphs():
        log_pot = numpy.random.random(len(hypergraph.edges))
        inside = numpy.random.random(len(hypergraph.edges))
        yield check_best_path, hypergraph, log_pot
        yield check_best_path_matrix, hypergraph
        yield check_inside, hypergraph, inside
        yield check_inside, hypergraph, log_pot
        yield check_outside, hypergraph, log_pot
        yield check_posteriors, hypergraph, inside
        yield check_max_marginals, hypergraph, log_pot
        yield check_semirings, hypergraph


def check_best_path_matrix(graph):
    """
    Test viterbi path finding using matrix representation.
    """
    scores = numpy.random.random(len(graph.edges))
    path = pydecode.best_path(graph, scores)
    path.v

def check_best_path(graph, max_potentials):
    """
    Test viterbi path finding.
    """
    path = pydecode.best_path(graph, max_potentials)
    nt.assert_not_equal(max_potentials.T * path.v, 0.0)
    utils.valid_path(graph, path)
    same = False
    for other_path in utils.all_paths(graph):
        assert max_potentials.T * path.v >= max_potentials.T * other_path.v
        if path == other_path:
            same = True
    assert same


def check_inside(graph, pot):
    """
    Test inside chart gen.
    """
    inside = pydecode.inside(graph, pot)


def check_outside(graph, pot):
    """
    Test outside chart properties.
    """
    print graph
    path = pydecode.best_path(graph, pot)
    chart = pydecode.inside(graph, pot)
    print pot.shape, path.v.shape
    best = pot.T * path.v
    print path.v
    print best
    nt.assert_almost_equal(best, chart[graph.root.id])
    nt.assert_not_equal(best, 0.0)

    out_chart = pydecode.outside(graph, pot, chart)

    # Array-form
    for vertex in graph.vertices:
        other = chart[vertex.id] + out_chart[vertex.id]
        nt.assert_less_equal(other, best + 1e-4,
                             "%f %f %d %f %f"%(other, best, vertex.id,
                                         chart[vertex.id], out_chart[vertex.id]))

    # Matrix-form
    m = chart + out_chart
    assert (m < best + 1e4).all()

    # for node in graph.nodes:
    #     other = chart[node] * out_chart[node]
    #     nt.assert_less_equal(other, best + 1e-4)
    print chart
    print out_chart

    for edge in path.edges:
        for node in edge.tail:
            if node.is_terminal:
                other = out_chart[node.id]
                nt.assert_almost_equal(other, best)


def check_posteriors(graph, pot):
    """
    Check the posteriors by enumeration.
    """

    node_marg = pydecode.marginals(graph, pot)

    paths = utils.all_paths(graph)
    m = defaultdict(lambda: 0.0)
    total_score = 0.0
    for path in paths:
        #path_score = prod([pot[edge.id] for edge in path.edges])
        path_score = np.exp(np.log(pot.T) * path.v)
        total_score += path_score
        for edge in path:
            m[edge.id] += path_score

    # for edge in graph.edges:
    #     nt.assert_almost_equal(
    #         edge_marg[edge.id] / node_marg[graph.root.id],
    #         m[edge.id] / total_score, places=4)

    # chart = ph.inside(graph, pot, kind=ph.Inside)
    # nt.assert_almost_equal(chart[graph.root.id], total_score, places=4)


def check_max_marginals(graph, pot):
    """
    Test that max-marginals are correct.
    """

    path = pydecode.best_path(graph, pot)
    best = pot.T * path.v
    # print "BEST"
    # print "\n".join(["%20s : %s" % (edge.label, pot[edge.id])
    #                  for edge in path.edges])
    # print best
    nt.assert_not_equal(best, 0.0)
    max_marginals = pydecode.marginals(graph, pot)

    # Array-form.
    for edge in graph.edges:
        other = max_marginals[edge.id]
        nt.assert_less_equal(other, best + 1e-4)

    # Matrix-form.
    assert (max_marginals < best + 1e-4).all()
    # for edge in graph.edges:
    #     other = max_marginals[edge]
    #     nt.assert_less_equal(other, best + 1e-4)
    #     if edge in path:
    #         nt.assert_almost_equal(other, best)

    # assert (max_marginals.as_edge_array() < best + 1e-4).all()
    # assert (len(max_marginals.as_edge_array()) == len(hypergraph.edges))

def check_semirings(graph):
    weights = [10.0] * len(graph.edges)
    weights2 = [0.5] * len(graph.edges)
    potentials = np.array(weights)
    edge_marg = pydecode.marginals(graph, potentials,
                                   weight_type=pydecode.Viterbi)

    log_potentials = np.array(weights)
    potentials = np.array(weights)
    chart = pydecode.inside(graph, log_potentials)
    chart2 = pydecode.inside(graph, potentials)

    # Array-form.
    for node in graph.nodes:
        nt.assert_equal(chart[node.id], chart2[node.id])

    # Matrix-form.
    numpy.testing.assert_array_almost_equal(
        chart, chart2, decimal=4)


    marg = pydecode.marginals(graph, log_potentials)
    marg2 = pydecode.marginals(graph, potentials)

    for edge in graph.edges:
        nt.assert_almost_equal(marg[edge.id], marg2[edge.id])

    potentials = np.array(weights2)
    # chart = ph.inside(graph, potentials, kind=ph.Inside)

if __name__ == "__main__":
    for a in test_main():
        print a[0]
        a[0](*a[1:])
