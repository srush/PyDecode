import pydecode.hyper as ph
import pydecode.test.utils as utils
from collections import defaultdict
import nose.tools as nt
import numpy as np

def test_main():
    for hypergraph in utils.hypergraphs():
        log_pot = utils.random_log_viterbi_potentials_array(hypergraph)
        inside = utils.random_inside_potentials(hypergraph)
        yield check_best_path, hypergraph, log_pot
        yield check_inside, hypergraph, inside
        yield check_inside, hypergraph, log_pot
        yield check_outside, hypergraph, log_pot
        yield check_posteriors, hypergraph, inside
        yield check_max_marginals, hypergraph, log_pot
        yield check_semirings, hypergraph


def check_best_path(hypergraph, pot):
    """
    Test viterbi path finding.
    """
    path = ph.best_path(hypergraph, pot)
    nt.assert_not_equal(pot.dot(path), 0.0)
    utils.valid_path(hypergraph, path)
    same = False
    for other_path in utils.all_paths(hypergraph):
        assert pot.dot(path) >= pot.dot(other_path)
        if path == other_path:
            same = True
    assert same


def check_inside(hypergraph, pot):
    """
    Test inside chart gen.
    """
    inside = ph.inside(hypergraph, pot)


def check_outside(hypergraph, pot):
    """
    Test outside chart properties.
    """
    path = ph.best_path(hypergraph, pot)
    chart = ph.inside_values(hypergraph, pot)
    best = pot.dot(path)
    nt.assert_not_equal(best, 0.0)
    out_chart = ph.outside_values(hypergraph, pot, chart)

    # Array-form
    for node in hypergraph.nodes:
        other = chart[node] * out_chart[node]
        nt.assert_less_equal(other, best + 1e-4)

    # Matrix-form
    m = chart.as_array() * out_chart.as_array()
    assert (m < best + 1e4).all()

    # for node in hypergraph.nodes:
    #     other = chart[node] * out_chart[node]
    #     nt.assert_less_equal(other, best + 1e-4)


    for edge in path.edges:
        for node in edge.tail:
            if node.is_terminal:
                other = out_chart[node]
                nt.assert_almost_equal(other, best)


def check_posteriors(hypergraph, pot):
    """
    Check the posteriors by enumeration.
    """

    marg = ph.compute_marginals(hypergraph, pot)

    paths = utils.all_paths(hypergraph)
    m = defaultdict(lambda: 0.0)
    total_score = 0.0
    for path in paths:
        path_score = pot.dot(path)
        total_score += path_score
        for edge in path:
            m[edge.id] += path_score

    for edge in hypergraph.edges:
        nt.assert_almost_equal(
            marg[edge] / marg[hypergraph.root],
            m[edge.id] / total_score, places=4)

    chart = ph.inside(hypergraph, pot)
    nt.assert_almost_equal(chart[hypergraph.root], total_score, places=4)


def check_max_marginals(hypergraph, pot):
    """
    Test that max-marginals are correct.
    """
    print pot.show(hypergraph)

    path = ph.best_path(hypergraph, pot)
    best = pot.dot(path)
    print "BEST"
    print "\n".join(["%20s : %s" % (edge.label, pot[edge])
                     for edge in path.edges])
    print best
    nt.assert_not_equal(best, 0.0)
    max_marginals = ph.compute_marginals(hypergraph, pot)

    # Array-form.
    for node in hypergraph.nodes:
        other = max_marginals[node]
        nt.assert_less_equal(other, best + 1e-4)

    # Matrix-form.
    assert (max_marginals.as_array() < best + 1e-4).all()

    for edge in hypergraph.edges:
        other = max_marginals[edge]
        nt.assert_less_equal(other, best + 1e-4)
        if edge in path:
            nt.assert_almost_equal(other, best)

    assert (max_marginals.as_edge_array() < best + 1e-4).all()
    assert (len(max_marginals.as_edge_array()) == len(hypergraph.edges))

def check_semirings(hypergraph):
    weights = [10.0] * len(hypergraph.edges)
    weights2 = [0.5] * len(hypergraph.edges)
    potentials = ph.ViterbiPotentials(hypergraph).from_vector(weights)
    marg = ph.Viterbi.compute_marginals(hypergraph, potentials)

    log_potentials = ph.LogViterbiPotentials(hypergraph).from_vector(weights)
    potentials = ph.LogViterbiPotentials(hypergraph).from_vector(weights)
    chart = ph.inside(hypergraph, log_potentials)
    chart2 = ph.inside_values(hypergraph, potentials)

    # Array-form.
    for node in hypergraph.nodes:
        nt.assert_equal(chart[node], chart2[node])

    # Matrix-form.
    assert (chart.as_array() == chart2.as_array()).all()

    marg = ph.LogViterbi.compute_marginals(hypergraph, log_potentials)
    marg2 = ph.compute_marginals(hypergraph, potentials)
    for edge in hypergraph.edges:
        nt.assert_almost_equal(marg[edge], marg2[edge])

    potentials = ph.Inside.Potentials(hypergraph).from_vector(weights2)
    chart = ph.Inside.inside(hypergraph, potentials)

    potentials = ph.Inside.Potentials(hypergraph).from_vector(weights2)
