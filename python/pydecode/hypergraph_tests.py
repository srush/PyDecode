import pydecode.hyper as ph
import random
import pydecode.display as draw
import networkx as nx
import matplotlib.pyplot as plt
import nose.tools as nt
import pydecode.constraints as cons
import pydecode.optimization as opt
import itertools
from collections import defaultdict

def get_all_hyperpaths(h):
    "Get all possible hyperpaths."
    def paths(node):
        if node.is_terminal:
            yield tuple()
        else:
            for edge in node.edges:
                t = [paths(node) for node in edge.tail]
                for below in itertools.product(*t):
                    yield (edge,) + sum(below, ())
    paths = [ph.Path(h, list(edges)) for edges in paths(h.root)]
    return paths


def test_all():
    s = simple_hypergraph()
    for path in get_all_hyperpaths(s):
        valid_path(s, path)

def simple_hypergraph():
    """
    Create a simple fixed hypergraph.
    """
    hypergraph = ph.Hypergraph()
    with hypergraph.builder() as b:
        term = [b.add_node() for i in range(4)]
        head_node = b.add_node([([term[0], term[1]], "0"),
                                ([term[0]], "1")])
        head_node2 = b.add_node([([head_node, term[2]], "2"),
                                 ([head_node, term[3]], "3"),
                                 ([head_node], "4")])
    return hypergraph

def random_hypergraph():
    """
    Generate a random hypergraph.
    """
    hypergraph = ph.Hypergraph()
    children = defaultdict(lambda: set())
    with hypergraph.builder() as b:
        terminals = []
        for i in range(50):
            n = b.add_node()
            terminals.append(n)
            children[n.id] = set([n.id])
        nodes = list(terminals)
        for node in range(50):
            node_a, node_b = random.sample(nodes, 2)
            if len(children[node_a.id] & children[node_b.id]) > 0: continue
            head_node = b.add_node((([node_a, node_b], node),))
            children[head_node.id] = \
                set([head_node.id]) | children[node_a.id] | children[node_b.id]
            nodes.append(head_node)
    nt.assert_greater(len(hypergraph.nodes), 0)
    assert len(hypergraph.edges) > 0
    return hypergraph

def hypergraphs():
    for i in range(10):
        h = random_hypergraph()
        print h
        yield h
    h = simple_hypergraph()
    print h
    yield h


# TESTS FOR HYPERGRAPH CONSTRUCTION.

def test_numbering():
    """
    Check that the numbering nodes and edges is correct.
    """
    for hypergraph in hypergraphs():
        for i, node in enumerate(hypergraph.nodes):
            nt.assert_equal(node.id, i)
        for i, edge in enumerate(hypergraph.edges):
            nt.assert_equal(edge.id, i)

def valid_hypergraph(hypergraph):
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


def test_simple_valid():
    valid_hypergraph(simple_hypergraph())


def test_valid():
    for hypergraph in hypergraphs():
        valid_hypergraph(hypergraph)


def valid_path(hypergraph, path):
    "Check whether a path is valid."
    root = hypergraph.root
    nt.assert_greater(len(path.edges), 0)
    # Check there is a path to terminals.
    stack = [hypergraph.root]
    while stack:
        node = stack[0]
        stack = stack[1:]
        if node.is_terminal:
            continue
        count = 0
        for edge in node.edges:
            if edge in path:
                count += 1
                for tail_node in edge.tail:
                    stack.append(tail_node)
        assert count == 1,\
            " Count is {}. Path is {}".format(count,
                                              pretty_print_path(path))

def test_construction():
    h = random_hypergraph()

def rand_gen(arg=None):
    return random.random()

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


def test_best_path():
    """
    Test viterbi path finding.
    """
    for h in hypergraphs():
        w = random_viterbi_potentials(h)
        path = ph.best_path(h, w)
        nt.assert_not_equal(w.dot(path), 0.0)
        valid_path(h, path)
        same = False
        for other_path in get_all_hyperpaths(h):
            assert w.dot(path) >= w.dot(other_path)
            if path == other_path: same = True
        assert same


def test_inside():
    """
    Test inside chart gen.
    """
    for h in hypergraphs():
        w = random_inside_potentials(h)
        inside = ph.inside(h, w)


def test_outside():
    """
    Test outside chart properties.
    """
    for h in hypergraphs():
        w = random_viterbi_potentials(h)
        path = ph.best_path(h, w)
        chart = ph.inside_values(h, w)
        best = w.dot(path)
        nt.assert_not_equal(best, 0.0)
        out_chart = ph.outside_values(h, w, chart)
        for node in h.nodes:
            other = chart[node] * out_chart[node]
            nt.assert_less_equal(other.value, best + 1e-4)
        for edge in path.edges:
            for node in edge.tail:
                if node.is_terminal:
                    nt.assert_almost_equal(other.value, best)


def test_posteriors():
    "Check the posteriors by enumeration."
    for h in hypergraphs():
        w = random_inside_potentials(h)
        marg = ph.compute_marginals(h, w)


        paths = get_all_hyperpaths(h)
        m = defaultdict(lambda: 0.0)
        total_score = 0.0
        for path in paths:
            path_score = w.dot(path)
            total_score += path_score
            for edge in path:
                m[edge.id] += path_score

        for edge in h.edges:
            nt.assert_almost_equal(
                marg[edge].value / marg[h.root].value,
                m[edge.id] / total_score, places=4)

        chart = ph.inside(h, w)
        nt.assert_almost_equal(chart[h.root].value, total_score, places=4)

def test_max_marginals():
    """
    Test that max-marginals are correct.
    """
    for h in hypergraphs():
        w = random_viterbi_potentials(h)
        print w.show(h)

        path = ph.best_path(h, w)
        best = w.dot(path)
        print "BEST"

        print "\n".join(["%20s : %s"%(h.label(edge), w[edge]) for edge in path.edges])
        print best
        nt.assert_not_equal(best, 0.0)
        max_marginals = ph.compute_marginals(h, w)
        for node in h.nodes:
            other = max_marginals[node]
            nt.assert_less_equal(other.value, best + 1e-4)

        for edge in h.edges:
            other = max_marginals[edge]
            nt.assert_less_equal(other.value, best + 1e-4)
            if edge in path:
                nt.assert_almost_equal(other.value, best)

### PRUNING CODE

def test_pruning():
    for h in hypergraphs():
        w = random_viterbi_potentials(h)

        original_path = ph.best_path(h, w)
        new_hyper, new_potentials = ph.prune_hypergraph(h, w, -0.99)
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
        prune = random.random()
        max_marginals = ph.compute_marginals(h, w)
        new_hyper, new_potentials = ph.prune_hypergraph(h, w, 0.0)

        assert (len(new_hyper.edges) > 0)
        original_edges = {}
        for edge in h.edges:
            original_edges[h.label(edge)] = edge

        new_edges = {}
        for edge in new_hyper.edges:
            new_edges[h.label(edge)] = edge

        for name, edge in new_edges.iteritems():

            orig = original_edges[name]
            nt.assert_almost_equal(w[orig], new_potentials[edge])
            m = max_marginals[orig]
            nt.assert_greater(m, prune * original_score)


### CONSTRAINT CODE

def random_have_constraint(hypergraph):
    edge, = random.sample(hypergraph.edges, 1)

    def build_constraints(label):
        l = hypergraph.label(edge)
        if label == l:
            return [("have", 1)]
        return []
    constraints = cons.Constraints(hypergraph, [("have", -1)]).build(
        build_constraints)
    return constraints, edge



def random_constraint(hypergraph):
    "Produce a random constraint on an edge."

    edge, = random.sample(hypergraph.edges, 1)
    print edge.id
    l = hypergraph.label(edge)

    def build_constraints(label):
        if label == l:
            print "label", label
            return [("have", 1), ("not", 1)]
        return []
    constraints = cons.Constraints(hypergraph, [("have", -1), ("not", 0)]).build(
        build_constraints)
    return constraints, edge

def test_constraint():
    """
    Test constraint checking.
    """
    for h in hypergraphs():
        w = random_viterbi_potentials(h)
        constraints, edge = random_constraint(h)
        path = ph.best_path(h, w)
        match = constraints.check(path)

        if edge not in path:
            print "Should not have", edge.id
            assert "have" in match
            assert "not" not in match
        else:
            print "Should have", edge.id
            assert "have" not in match

        nt.assert_equal(len(match), 1)

### SUBGRADIENT OPTIMIZATION CODE

def test_subgradient():
    for h in hypergraphs():
        w = random_log_viterbi_potentials(h)
        constraints, edge = random_have_constraint(h)
        path = ph.best_path(h, w)
        match = constraints.check(path)
        if edge not in path:
            nt.assert_equal(match[0], "have")

        cpath = opt.best_constrained_path(h, w,
                                          constraints)
        assert edge in cpath

### LINEAR PROGRAMMING CODE

def test_lp():
    import pydecode.lp as lp
    for h in hypergraphs():
        w = random_log_viterbi_potentials(h)
        g = lp.HypergraphLP.make_lp(h, w)
        g.solve()
        path = g.path
        opath = ph.best_path(h, w)
        nt.assert_almost_equal(w.dot(path), w.dot(opath))

        for edge in path.edges:
            assert(edge in opath)
        constraints, edge = random_have_constraint(h)
        g = lp.HypergraphLP.make_lp(h, w)
        g.add_constraints(constraints)
        g.solve()
        assert edge in g.path

### LINEAR PROGRAMMING CODE

def test_semirings():
    for hypergraph in hypergraphs():
        potentials = ph.ViterbiPotentials(hypergraph).build(lambda l: 10.0)
        marg = ph.Viterbi.compute_marginals(hypergraph, potentials)

        log_potentials = ph.LogViterbiPotentials(hypergraph).build(lambda l: 10.0)
        potentials = ph.LogViterbiPotentials(hypergraph).build(lambda l: 10.0)
        chart = ph.inside(hypergraph, log_potentials)
        chart2 = ph.inside_values(hypergraph, potentials)
        for node in hypergraph.nodes:
            nt.assert_equal(chart[node], chart2[node])

        marg = ph.LogViterbi.compute_marginals(hypergraph, log_potentials)
        marg2 = ph.compute_marginals(hypergraph, potentials)
        for edge in hypergraph.edges:
            nt.assert_almost_equal(marg[edge], marg2[edge])


        potentials = ph.Inside.Potentials(hypergraph).build(lambda l: 0.5)
        chart = ph.Inside.inside(hypergraph, potentials)

        potentials = ph.Inside.Potentials(hypergraph).build(lambda l: 0.5)


## CONSTRUCTION CODE

@nt.raises(Exception)
def test_diff_potentials_fail():
    h1, w1 = random_hypergraph()
    h2, w2 = random_hypergraph()
    ph.best_path(h1, w2)


@nt.raises(Exception)
def test_outside_fail():
    h1, w1 = random_hypergraph()
    h2, w2 = random_hypergraph()
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


@nt.raises(Exception)
def test_bad_constraints():
    h1, w1 = random_hypergraph()
    h2, w2 = random_hypergraph()
    c1, _ = random_have_constraint(h1)
    best_constrained(h2, w2, c1)


if __name__ == "__main__":
    test_subgradient()
