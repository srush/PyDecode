import pydecode.hyper as ph
import random
import pydecode.display as draw
import networkx as nx
import matplotlib.pyplot as plt
import nose.tools as nt
import pydecode.constraints as cons
import pydecode.optimization as opt


def simple_hypergraph():
    hypergraph = ph.Hypergraph()

    with hypergraph.builder() as b:
        term = [b.add_node() for i in range(3)]
        head_node = b.add_node([([term[0], term[1]], "")])
        head_node2 = b.add_node([([head_node, term[2]], "")])
    weights = ph.Weights(hypergraph, kind=ph.LogViterbi).build(lambda t: random.random())
    return hypergraph, weights


def random_hypergraph():
    "Generate a random hypergraph."
    hypergraph = ph.Hypergraph()

    with hypergraph.builder() as b:
        terminals = [b.add_node() for i in range(10)]
        nodes = list(terminals)
        for node in range(10):
            node_a, node_b = random.sample(nodes, 2)
            head_node = b.add_node((([node_a, node_b], node),))
            nodes.append(head_node)
    nt.assert_greater(len(hypergraph.nodes), 0)
    assert len(hypergraph.edges) > 0

    weights = ph.Weights(hypergraph, kind=ph.LogViterbi).build(lambda t: random.random())
    return hypergraph, weights

def test_semirings():
    hypergraph = simple_hypergraph()[0]
    weights = ph.Weights(hypergraph, kind=ph.Viterbi).build(lambda l: 10.0)
    marg = ph.Viterbi.compute_marginals(hypergraph, weights)
    # for edge in hypergraph.edges:
    #     marg[edge]

    log_weights = ph.Weights(hypergraph, kind=ph.LogViterbi).build(lambda l: 10.0)
    weights = ph.Weights(hypergraph, kind=ph.LogViterbi).build(lambda l: 10.0)
    chart = ph.LogViterbi.inside(hypergraph, log_weights)
    chart2 = ph.inside_values(hypergraph, weights)
    for node in hypergraph.nodes:
        nt.assert_equal(chart[node], chart2[node])

    marg = ph.LogViterbi.compute_marginals(hypergraph, log_weights)
    marg2 = ph.compute_marginals(hypergraph, weights)
    for edge in hypergraph.edges:
        nt.assert_almost_equal(marg[edge], marg2[edge])


    weights = ph.Inside.Weights(hypergraph).build(lambda l: 0.5)
    chart = ph.Inside.inside(hypergraph, weights)

    weights = ph.Inside.Weights(hypergraph).build(lambda l: 0.5)



def test_numbering():
    for hypergraph, _ in [random_hypergraph() for i in range(10)]:
        for i, node in enumerate(hypergraph.nodes):
            nt.assert_equal(node.id, i)
        for i, edge in enumerate(hypergraph.edges):
            nt.assert_equal(edge.id, i)


def valid_hypergraph(hypergraph):
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
    valid_hypergraph(simple_hypergraph()[0])


def test_valid():
    for hypergraph, w in [random_hypergraph() for i in range(10)]:
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


def test_inside():
    for h, w in [random_hypergraph() for i in range(10)]:
        path = ph.best_path(h, w)
        nt.assert_not_equal(w.dot(path), 0.0)

        valid_path(h, path)


def test_outside():
    for h, w in [random_hypergraph() for i in range(10)]:
        path = ph.best_path(h, w)
        chart = ph.inside_values(h, w)
        best = w.dot(path)
        nt.assert_not_equal(best, 0.0)
        out_chart = ph.outside_values(h, w, chart)
        for node in h.nodes:
            other = chart[node] + out_chart[node]
            nt.assert_less_equal(other.value, best + 1e-4)
            if node.is_terminal:
                nt.assert_almost_equal(other.value, best)


def test_maxmarginals():
    for h, w in [random_hypergraph() for i in range(10)]:
        path = ph.best_path(h, w)
        best = w.dot(path)
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


def random_constraint(hypergraph):
    edge, = random.sample(hypergraph.edges, 1)

    def build_constraints(label):
        l = hypergraph.label(edge)
        if label == l:
            return [("have", 1), ("not", 1)]
        return []
    constraints = cons.Constraints(hypergraph, [("have", -1), ("not", 0)]).build(
        build_constraints)
    return constraints, edge


def test_constraint():
    for h, w in [random_hypergraph() for i in range(10)]:
        constraints, edge = random_constraint(h)
        for edge in h.edges:
            print edge.id
            print constraints.weights[edge]
        path = ph.best_path(h, w)
        match = constraints.check(path)
        print match
        if edge not in path:
            assert "have" in match
        else:
            assert "not" in match


def test_pruning():
    for h, w in [random_hypergraph() for i in range(10)]:
        original_path = ph.best_path(h, w)
        new_hyper, new_weights = ph.prune_hypergraph(h, w, -0.99)
        prune_path = ph.best_path(new_hyper, new_weights)
        assert len(original_path.edges) > 0
        for edge in original_path.edges:
            assert edge in prune_path
        valid_path(new_hyper, prune_path)

        original_score = w.dot(original_path)
        print original_score
        print new_weights.dot(prune_path)
        nt.assert_almost_equal(original_score,
                               new_weights.dot(prune_path))

        # Test pruning amount.
        prune = random.random()
        max_marginals = ph.compute_marginals(h, w)
        new_hyper, new_weights = ph.prune_hypergraph(h, w, 0.0)

        assert (len(new_hyper.edges) > 0)
        original_edges = {}
        for edge in h.edges:
            original_edges[h.label(edge)] = edge

        new_edges = {}
        for edge in new_hyper.edges:
            new_edges[h.label(edge)] = edge

        for name, edge in new_edges.iteritems():

            orig = original_edges[name]
            nt.assert_almost_equal(w[orig], new_weights[edge])
            m = max_marginals[orig]
            nt.assert_greater(m, prune * original_score)


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


def test_subgradient():
    for h, w in [random_hypergraph() for i in range(10)]:
        constraints, edge = random_have_constraint(h)
        path = ph.best_path(h, w)
        match = constraints.check(path)
        if edge not in path:
            nt.assert_equal(match[0], "have")

        cpath = opt.best_constrained_path(h, w,
                                          constraints)
        assert edge in cpath


def test_lp():
    import pydecode.lp as lp
    for h, w in [simple_hypergraph()]:
        g = lp.HypergraphLP.make_lp(h, w)
        g.solve()
        path = g.path
        opath = ph.best_path(h, w)
        nt.assert_almost_equal(w.dot(path), w.dot(opath))

        for edge in path.edges:
            assert(edge in opath)


@nt.raises(Exception)
def test_diff_weights_fail():
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
