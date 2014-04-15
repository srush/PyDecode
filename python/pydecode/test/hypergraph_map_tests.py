"""
Tests for hypergraph mapping and pruning.
"""

import pydecode.hyper as ph
import nose.tools as nt


@nt.nottest
def test_pruning():
    for h in hypergraphs():
        w = utils.random_viterbi_potentials(h)

        original_path = ph.best_path(h, w)
        prune_projection = ph.prune_hypergraph(h, w, -0.99)
        new_hyper = prune_projection.small_hypergraph
        new_potentials = w.project(h, prune_projection)
        prune_path = ph.best_path(new_hyper, new_potentials)
        assert len(original_path.edges) > 0
        for edge in original_path.edges:
            assert edge in prune_path
        valid_path(new_hyper, prune_path)

        original_score = w.dot(original_path)
        print original_score
        print new_potentials.dot(prune_path)
        nt.assert_almost_equal(original_score,
                               new_potentials.dot(prune_path))

        # Test pruning amount.
        prune = 0.001
        max_marginals = ph.marginals(h, w)
        prune_projection = ph.prune(h, w, prune)

        new_hyper = prune_projection.small_hypergraph
        new_potentials = w.project(h, prune_projection)

        assert (len(new_hyper.edges) > 0)
        original_edges = {}
        for edge in h.edges:
            original_edges[edge.label] = edge

        new_edges = {}
        for edge in new_hyper.edges:
            new_edges[edge.label] = edge

        for name, edge in new_edges.iteritems():
            orig = original_edges[name]
            nt.assert_almost_equal(w[orig], new_potentials[edge])
            m = max_marginals[orig]
            nt.assert_greater(m, prune)
