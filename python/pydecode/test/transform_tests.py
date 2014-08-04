"""
Tests for hypergraph mapping and pruning.
"""

import pydecode
import pydecode.test.utils as utils
import nose.tools as nt
import numpy as np
import numpy.random
import numpy.testing

def test_pruning():
    for h in utils.hypergraphs():

        w = numpy.random.random(len(h.edges))

        original_path = pydecode.best_path(h, w)
        marginals = pydecode.marginals(h, w)
        best = w.T * original_path.v
        print marginals[1]
        a = np.array(marginals > 0.99* best, dtype=np.uint8)

        # new_hyper = pydecode.filter(h, a)
        # # print project.shape, w.shape
        # # print project.todense()
        # # new_potentials = project * w
        # prune_path = pydecode.best_path(new_hyper, new_potentials)
        # assert len(original_path.edges) > 0


        # # print "proect ", project.shape
        # # print project * prune_path.v, (project * prune_path.v).shape
        # # print original_path.v, original_path.v.shape
        # numpy.testing.assert_equal(
        #     prune_path.v.todense(),
        #     (project * original_path.v).todense())

        # # nt.assert_almost_equal(
        #     best,
        #     new_potentials.T * prune_path.v)

        # Test pruning amount.
        # prune = 0.001
        # max_marginals = ph.marginals(h, w)
        # prune_projection = ph.prune(h, w, prune)

        # new_hyper = prune_projection.small_hypergraph
        # new_potentials = w.project(h, prune_projection)

        # assert (len(new_hyper.edges) > 0)
        # original_edges = {}
        # for edge in h.edges:
        #     original_edges[edge.label] = edge

        # new_edges = {}
        # for edge in new_hyper.edges:
        #     new_edges[edge.label] = edge

        # for name, edge in new_edges.iteritems():
        #     orig = original_edges[name]
        #     nt.assert_almost_equal(w[orig], new_potentials[edge])
        #     m = max_marginals[orig]
        #     nt.assert_greater(m, prune)

if __name__ == "__main__":
    test_pruning()
