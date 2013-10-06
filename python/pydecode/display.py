import networkx as nx

def to_networkx(hypergraph, extra=[], node_extra=[], paths=[], constraints = None):
    """Convert hypergraph to networkx graph representation.

    :param hypergraph: The hypergraph to convert.
    :param extra: Extra naming information for edges.
    :param node_extra: Extra naming information for nodes.
    :param paths: Paths to highlight in the graph.
    """

    colors = ["red", "blue", "green"]
    def node_label(node):
        return "{}".format(" ".join(
                [node.label] + [extra[node] for extra in node_extra]))

    graph = nx.DiGraph()
    for node in hypergraph.nodes:
        label = node_label(node)
        graph.add_node(node.id, label = label)
        for edge in node.edges:
            artificial_node = "[e{}]".format(edge.id)
            color = ""
            for path, c in zip(paths, colors):
                if edge in path:
                    color = c
            label = "{}".format(hypergraph.label(edge))
            for labeler in extra:
                label += " : " + str(labeler[edge])
            graph.add_node(artificial_node, shape = "rect", label = label)
            graph.add_edge(node.id, artificial_node, color=color)
            for tail_nodes in edge.tail:
                graph.add_edge(artificial_node, tail_nodes.id, color=color)
    return graph

def to_image(hypergraph, filename, extra=[], node_extra=[], paths=[], constraints = None):
    G = to_networkx(hypergraph, extra, node_extra, paths, constraints)
    agraph = nx.drawing.to_agraph(G)
    agraph.graph_attr.update({"rankdir":  "RL"})
    agraph.layout("dot")
    agraph.draw(filename)

def to_ipython(hypergraph, extra=[], node_extra=[], paths=[], constraints = None):
    """Display a hypergraph in iPython.

    :param hypergraph: The hypergraph to convert.
    :param extra: Extra naming information for edges.
    :param node_extra: Extra naming information for nodes.
    :param paths: Paths to highlight in the graph.
    """

    from IPython.display import Image
    temp_file = "/tmp/tmp.png"
    to_image(hypergraph, temp_file, extra, node_extra, paths, constraints)
    return Image(filename = temp_file)

def pretty_print(hypergraph):
    graph = nx.Graph()
    for node in hypergraph.nodes:
        head_node = node.id
        graph.add_node(head_node)
        for edge in node.edges:
            artificial_node = str(edge.id) + "*"
            graph.add_node(artificial_node)
            graph.add_edge(head_node, artificial_node)
            for tail_node_id in edge.tail_node_ids :
                graph.add_edge(artificial_node, tail_node_id)

def pretty_print_path(path):
    out = ""
    for edge in path.edges:
        out += "{} -> {}\n".format(edge.head.id,
                                " ".join([node.id for node in edge.tail]))
    return out


def report(duals):
    import pandas as pd
    df = pd.DataFrame(
        data = {"dual": [d.dual for d in duals],
                "ncons": [len(d.constraints) for d in duals]},
        index = range(len(duals)))
    df.plot()
