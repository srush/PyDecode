import decoding_ext as d


hypergraph = d.HGraph()
d.HConstraints(hypergraph)
wb = d.WeightBuilder(hypergraph)
with hypergraph.builder() as b:
    node_a = b.add_terminal_node()

    with b.add_node() as node_b:
        edge = node_b.add_edge([node_a], "hello")
        wb.set_weight(edge, 10)

    with b.add_node() as node_c:
        edge = node_c.add_edge([node_b], "hello")
        wb.set_weight(edge, 15)
    print hypergraph.edges_size()
weights = wb.weights()
d.viterbi(hypergraph, weights)
