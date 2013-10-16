import networkx as nx

class HypergraphFormatter:
    """
    The base class for hypergraph formatters.
    Define the style-sheet for graphviz representation.

    Full list available - http://www.graphviz.org/content/attrs
    """
    def __init__(self, hypergraph):
        self.hypergraph = hypergraph

    def graph_attrs(self):
        "Returns a dictionary of graph properties."
        return {"rankdir": "RL"}

    def hypernode_attrs(self, node):
        """
        Returns a dictionary of node properties for style hypernode.

        :param node: The hypernode to style.
        """
        return {"shape": "ellipse", "label":str(self.hypergraph.node_label(node))}

    def hyperedge_node_attrs(self, edge):
        """
        Returns a dictionary of node properties for styling intermediate hyperedge nodes.

        :param edge: The hyperedge to style.
        """
        return {"shape": "rect",
                "label": str(self.hypergraph.label(edge))}
    def hyperedge_attrs(self, edge):
        """
        Returns a dictionary of edge properties for styling hyperedge.

        :param edge: The hyperedge to style.
        """
        return {}

    def hypernode_subgraph(self, node): return []
    def hyperedge_subgraph(self, edge): return []
    def subgraph_format(self, subgraph): return {}

class HypergraphPathFormatter(HypergraphFormatter):
    def __init__(self, hypergraph, paths):
        self.hypergraph = hypergraph
        self.paths = paths
    def hyperedge_attrs(self, edge):
        colors = ["red", "green", "blue", "purple", "orange", "yellow"]
        for path, color in zip(self.paths, colors):
            if edge in path:
                return {"color": color}
        return {}


class HypergraphWeightFormatter(HypergraphFormatter):
    def __init__(self, hypergraph, weights):
        self.hypergraph = hypergraph
        self.weights = weights
    def hyperedge_node_attrs(self, edge):
        return {"label": self.weights[edge]}

class HypergraphConstraintFormatter(HypergraphFormatter):
    def __init__(self, hypergraph, constraints):
        self.hypergraph = hypergraph
        self.constraints = constraints
    def hyperedge_attrs(self, edge):
        colors = ["red", "green", "blue", "purple"]
        for constraint, color in zip(self.constraints, colors):
            if edge in constraint:
                return {"color": color}
        return {}


def to_networkx(hypergraph, graph_format):
    """Convert hypergraph to networkx graph representation.

    :param hypergraph: The hypergraph to convert.
    :type hypergraph: :py:class:`Hypergraph`

    :param graph_format: A hypergraph formatter.
    :type graph_format: :py:class:`HypergraphFormatter`

    :rtype: NetworkX Graph
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
    """
    :param hypergraph: The hypergraph to convert.
    :type hypergraph: :py:class:`Hypergraph`

    :param filename: A filename to writeout image.

    :param graph_format: A hypergraph formatter.
    :type graph_format: :py:class:`HypergraphFormatter`

    :rtype: NetworkX Graph
    """
    subgraphs = {}
    G = to_networkx(hypergraph, graph_format)
    agraph = nx.drawing.to_agraph(G)

    for node in hypergraph.nodes:
        for sub, rank in graph_format.hypernode_subgraph(node):
            subgraphs.setdefault(sub, [])
            subgraphs[sub].append((node.id, rank))

    for sub, node_ranks in subgraphs.iteritems():
        node_ranks.sort(key = lambda (node, rank): rank)
        nodes = [node for node, rank in node_ranks]
        subgraph = agraph.subgraph(nodes, name = sub)
        if sub[:7] != "cluster":
            # for node in nodes:
            #     agraph.get_node(node).attr.update({"group": sub})

            for n_a, n_b in zip(nodes, nodes[1:]):
                edge = subgraph.add_edge(n_a, n_b,
                                         weight = 1000, style="invis")
        subgraph.graph_attr.update(graph_format.subgraph_format(sub))
    agraph.graph_attr.update(graph_format.graph_attrs())

    agraph.layout("dot")
    agraph.write("/tmp/tmp.dot")
    agraph.draw(filename)

def to_ipython(hypergraph, graph_format):
    """Display a hypergraph in iPython.

    :param hypergraph: The hypergraph to convert.
    :type hypergraph: :py:class:`Hypergraph`

    :param graph_format: A hypergraph formatter.
    :type graph_format: :py:class:`HypergraphFormatter`

    :rtype: IPython display object.
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
    """Display a constrained result in IPython
    """

    import pandas as pd
    df = pd.DataFrame(
        data = {"dual": [d.dual for d in duals],
                "ncons": [len(d.constraints) for d in duals]},
        index = range(len(duals)))
    df.plot()
