import networkx as nx

def to_networkx(hypergraph):
    """Convert hypergraph to networkx representation.

    :param hypergraph: The hypergraph to convert.
    """
    graph = nx.Graph()
    for node in hypergraph.nodes():
        head_node = node.id()
        graph.add_node(head_node)
        for edge in node.edges():
            artificial_node = str(edge.id()) + " * " + edge.label()
            graph.add_node(artificial_node)
            graph.add_edge(head_node, artificial_node)
            for tail_nodes in edge.tail() :
                graph.add_edge(artificial_node, tail_nodes.id())
    return graph

def pretty_print(hypergraph):
    graph = nx.Graph()
    for node in hypergraph.nodes():
        head_node = node.id()
        graph.add_node(head_node)
        for edge in node.edges():
            artificial_node = str(edge.id) + "*"
            graph.add_node(artificial_node)
            graph.add_edge(head_node, artificial_node)
            for tail_node_id in edge.tail_node_ids :
                graph.add_edge(artificial_node, tail_node_id)

def pretty_print_path(path):
    out = ""
    for edge in path.edges():
        out += "{} -> {}\n".format(edge.head().id(), 
                                " ".join([node.id() for node in edge.tail()]))
    return out 
