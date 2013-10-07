import networkx as nx

class HypergraphFormatter:
    def __init__(self, hypergraph):
        self.hypergraph = hypergraph
    def hypernode_attrs(self, node):
        return {"shape": "ellipse", "label":str(self.hypergraph.node_label(node))}
    def hyperedge_node_attrs(self, edge):
        return {"shape": "rect",
                "label": str(self.hypergraph.label(edge))}
    def hyperedge_attrs(self, edge):
        return {}
    def hypernode_subgraph(self, node): return []
    def hyperedge_subgraph(self, edge): return []
    def subgraph_format(self, subgraph): return {}

class HypergraphPathFormatter(HypergraphFormatter):
    def __init__(self, hypergraph, path):
        self.hypergraph = hypergraph
        self.path = path
    def hyperedge_attrs(self, edge):
        if edge in self.path:
            return {"color":"blue"}
        return {}


def to_networkx(hypergraph, graph_format):
    """Convert hypergraph to networkx graph representation.

    :param hypergraph: The hypergraph to convert.
    :param graph_format: A dictionary of formatting options.
    """


    graph = nx.DiGraph()
    def e(edge): return "e" + str(edge.id)
    for node in hypergraph.nodes:
        graph.add_node(node.id)
        graph.node[node.id].update(
            graph_format.hypernode_attrs(node))
        for edge in node.edges:
            graph.add_node(e(edge))
            graph.node[e(edge)].update(
                graph_format.hyperedge_node_attrs(edge))

            graph.add_edge(node.id, e(edge))
            graph[node.id][e(edge)].update(
                graph_format.hyperedge_attrs(edge))

            for tail_nodes in edge.tail:
                graph.add_edge(e(edge), tail_nodes.id)
                graph[e(edge)][tail_nodes.id].update(
                    graph_format.hyperedge_attrs(edge))
    return graph

def to_image(hypergraph, filename, graph_format):
    subgraphs = {}
    G = to_networkx(hypergraph, graph_format)
    agraph = nx.drawing.to_agraph(G)

    for node in hypergraph.nodes:
        for sub in graph_format.hypernode_subgraph(node):
            subgraphs.setdefault(sub, [])
            subgraphs[sub].append(node.id)
    for sub, nodes in subgraphs.iteritems():
        subgraph = agraph.subgraph(nodes, name = sub)
        subgraph.graph_attr.update(graph_format.subgraph_format(sub))
    agraph.graph_attr.update({"rankdir": "RL"})
    agraph.layout("dot")
    agraph.draw(filename)

def to_ipython(hypergraph, graph_format):
    """Display a hypergraph in iPython.

    :param hypergraph: The hypergraph to convert.
    :param extra: Extra naming information for edges.
    :param node_extra: Extra naming information for nodes.
    :param paths: Paths to highlight in the graph.
    """

    from IPython.display import Image
    temp_file = "/tmp/tmp.png"
    to_image(hypergraph, temp_file, graph_format)
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
