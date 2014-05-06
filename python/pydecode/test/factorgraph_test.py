import pydecode.factorgraph as fg
import pydecode.hyper as ph
import random
import numpy as np

def make_simple_factor_graph():
    """
    Make a factor graph with three variables, each taking 5 labels,
    and two factors.
    """
    labels = 3
    graph = fg.FactorGraph(3, [labels, labels, labels], [[random.random()
                                  for i in range(labels)] for j in range(3)])
    factor1 = fg.TabularFactor(2, [labels, labels],  [[random.random()  for j in range(labels)]
                                 for i in range(labels)] )
    factor2 = fg.TabularFactor(2, [labels, labels], [[random.random()  for j in range(labels)]
                                 for i in range(labels)] )
    graph.register_factor(factor1, [0, 1])
    graph.register_factor(factor2, [1, 2])
    return graph

def make_hypergraph_factor_graph():
    hypergraph = ph.make_lattice(2, 3, [[0, 1 ,2]] * 3)

    weights = ph.LogViterbiPotentials(hypergraph) \
        .from_array(np.random.random(len(hypergraph.edges)))
    labels_pot = ph.SparseVectorPotentials(hypergraph) \
        .from_vector([[(edge.tail[0].label.i, edge.tail[0].label.j + 1)]
                      for edge in hypergraph.edges])

    labels = 4
    variables = 3
    possible_labels = [labels] * variables
    graph = fg.FactorGraph(variables, possible_labels,
                           [[random.random() for i in range(labels)]
                            for j in range(variables)])
    factor_hyp = fg.HypergraphFactor(variables, possible_labels,
                                     hypergraph, weights, labels_pot)

    graph.register_factor(factor_hyp, range(variables))
    return graph

def make_decipher_factorgraph():
    hypergraph = ph.make_lattice(5, 3, [[0, 1 ,2]] * 3)

    weights = ph.LogViterbiPotentials(hypergraph) \
        .from_array(np.random.random(len(hypergraph.edges)))

    labels_pot = ph.SparseVectorPotentials(hypergraph) \
        .from_vector([[(edge.tail[0].label.i, edge.tail[0].label.j + 1)]
                      for edge in hypergraph.edges])

    labels = 4
    variables = 6
    possible_labels = [labels] * variables
    graph = fg.FactorGraph(variables, possible_labels,
                           [[0.0 for i in range(labels)]
                            for j in range(variables)])
    factor_hyp = fg.HypergraphFactor(variables, possible_labels,
                                     hypergraph, weights, labels_pot)

    agree_hyp = fg.AgreeFactor(2, [labels]*2)
    agree_hyp2 = fg.AgreeFactor(2, [labels]*2)

    graph.register_factor(factor_hyp, range(variables))
    graph.register_factor(agree_hyp, [1,3])
    graph.register_factor(agree_hyp2, [2,4])
    return graph


def run_mplp(graph):
    print "US"
    fg.mplp(graph)
    print "EXHAUSTIVE"
    print fg.exhaustive_search(graph)

    # print fg.exhaustive_max_marginals(graph, graph.factors[0])
    # print fg.exhaustive_max_marginals(graph, graph.factors[1])
    # print fg.exhaustive_max_marginals(graph, graph.factors[2])


if __name__ == "__main__":
    # graph = make_simple_factor_graph()
    # run_mplp(graph)

    # graph = make_hypergraph_factor_graph()
    # run_mplp(graph)
    graph = make_decipher_factorgraph()
    run_mplp(graph)
