"""
Tests for constrained hypergraph optimization.
"""

import pydecode.hyper as ph
import pydecode.test.utils as utils
import nose.tools as nt
import pydecode.constraints as cons
import pydecode.optimization as opt
import random


def test_main():
    for hypergraph in utils.hypergraphs():
        path = utils.random_path(hypergraph)
        yield check_constraint, hypergraph, path
        # yield check_variables, hypergraph, path
        yield check_lp, hypergraph


def check_constraint(hypergraph, path):
    """
    Test constraint checking.
    """
    constraints, edge = random_constraint(hypergraph)
    match = constraints.check(path)
    if edge not in path:
        print "Should not have", edge.id
        assert "have" in match
        assert "not" not in match
    else:
        print "Should have", edge.id
        assert "have" not in match
    nt.assert_equal(len(match), 1)


def check_variables(hypergraph, path):
    """
    Test variable constraint checking.
    """
    variables, edge = random_constraint_trans(hypergraph)
    match = list(variables.check(path))
    if edge not in path:
        print "Should not have", edge.id
        assert "have" in match
        assert "not" not in match
    else:
        print "Should have", edge.id
        assert "have" not in match
    nt.assert_equal(len(match), 1)

### SUBGRADIENT OPTIMIZATION CODE


@nt.nottest
def test_subgradient():
    for h in utils.hypergraphs():
        w = utils.random_log_viterbi_potentials(h)
        constraints, edge = random_have_constraint(h)
        path = ph.best_path(h, w)
        match = constraints.check(path)
        if edge not in path:
            nt.assert_equal(match[0], "have")

        cpath = opt.best_constrained_path(h, w,
                                          constraints)
        assert edge in cpath

### LINEAR PROGRAMMING CODE


def check_lp(hypergraph):
    import pydecode.lp as lp
    w = utils.random_log_viterbi_potentials(hypergraph)

    g = lp.HypergraphLP.make_lp(hypergraph, w)
    g.solve()
    path = g.path
    opath = ph.best_path(hypergraph, w)

    nt.assert_almost_equal(w.dot(path), w.dot(opath))
    for edge in path.edges:
        assert edge in opath

    # Constraint.
    constraints, edge = random_have_constraint(hypergraph)
    g = lp.HypergraphLP.make_lp(hypergraph, w)
    g.add_constraints(constraints)
    g.solve()
    assert edge in g.path


@nt.raises(Exception)
def test_bad_constraints():
    h1, w1 = utils.random_hypergraph()
    h2, w2 = utils.random_hypergraph()
    c1, _ = random_have_constraint(h1)
    best_constrained(h2, w2, c1)


# Beam search tests.

def test_future_constraints():
    """
    Test constraint checking.
    """
    hypergraph = utils.simple_hypergraph()

    def build_constraints(label):
        if label == "1":
            return [("one", 1)]
        return []
    constraints = cons.Constraints(hypergraph, [("one", -1)]). \
        from_vector([build_constraints(edge.label)
                     for edge in hypergraph.edges])

    # Compute min and max potentials.
    min_potentials = ph.MinSparseVectorPotentials(hypergraph).\
        from_potentials(constraints.potentials)
    max_potentials = ph.MaxSparseVectorPotentials(hypergraph).\
        from_potentials(constraints.potentials)

    print "sparse"
    for edge in hypergraph.edges:
        print edge.label, constraints.potentials[edge]

    # Compute min and max potentials.
    print "min"
    in_chart = ph.inside(hypergraph, min_potentials)
    out_chart = ph.outside(hypergraph, min_potentials, in_chart)
    for node in hypergraph.nodes:
        print "%20s %20s %20s" % (node.label, in_chart[node], out_chart[node])

    print "max"
    in_chart = ph.inside(hypergraph, max_potentials)
    out_chart = ph.outside(hypergraph, max_potentials, in_chart)
    for node in hypergraph.nodes:
        print "%20s %20s %20s" % (node.label, in_chart[node], out_chart[node])


def random_have_constraint(hypergraph):
    orig_edge, = random.sample(hypergraph.edges, 1)

    def build_constraints(label):
        l = orig_edge.label
        if label == l:
            return [("have", 1)]
        return []
    constraints = cons.Constraints(hypergraph, [("have", -1)]).\
        from_vector([build_constraints(edge.label)
                     for edge in hypergraph.edges])
    return constraints, orig_edge


def random_constraint(hypergraph):
    "Produce a random constraint on an edge."

    orig_edge, = random.sample(hypergraph.edges, 1)
    l = orig_edge.label

    def build_constraints(label):
        if label == l:
            print "label", label
            return [("have", 1), ("not", 1)]
        return []
    constraints = cons.Constraints(hypergraph, [("have", -1), ("not", 0)]).\
        from_vector([build_constraints(edge.label)
                     for edge in hypergraph.edges])
    return constraints, orig_edge


def random_constraint_trans(hypergraph):
    "Produce a random constraint on an edge."

    orig_edge, = random.sample(hypergraph.edges, 1)
    l = orig_edge.label

    def build_variables(label):
        if label == l:
            b = ph.Bitset()
            b[0] = 1
            return b
        return None
    constraints = [cons.Constraint("have", [0], [1], -1),
                   cons.Constraint("not", [0], [1], 0)]
    variables = cons.Variables(hypergraph, 1, constraints)\
        .from_vector([build_variables(edge.label)
                      for edge in hypergraph.edges])
    return variables, orig_edge
