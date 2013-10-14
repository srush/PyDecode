import pulp
import pydecode.hyper as ph
from collections import defaultdict

class HypergraphLP:
    """
    Representation of a hypergraph LP.
    Requires the pulp library.
    """
    def __init__(self, lp, hypergraph, node_vars, edge_vars):
        self.lp = lp
        self.hypergraph = hypergraph
        self.node_vars = node_vars
        self.edge_vars = edge_vars

    def solve(self, solver=None):
        status = self.lp.solve()
        path_edges = [edge
                      for edge in self.hypergraph.edges
                      if pulp.value(self.edge_vars[edge.id]) == 1.0]
        return ph.Path(self.hypergraph, path_edges)

    def add_constraints(constraints):
        for constraint in constraints:
            self.lp += constraint.constrant == \
                sum([coeff * self.edge_vars[edge.id]
                     for (coeff, edge) in constraint])

    @staticmethod
    def make_lp(hypergraph, weights,
                name="Hypergraph Problem",
                var_type=pulp.LpContinuous):
        prob = pulp.LpProblem("Hypergraph Problem", pulp.LpMinimize)

        def node_name(node):
            return "node_{}".format(node.id)
        def edge_name(edge):
            return "edge_{}".format(edge.id)
        #hypergraph.label(edge))

        # Make variables for the nodes.
        node_vars = {node.id :
                     pulp.LpVariable(node_name(node), 0, 1,
                                     var_type)
                     for node in hypergraph.nodes}

        edge_vars = {edge.id :
                     pulp.LpVariable(edge_name(edge), 0, 1,
                                     var_type)
                     for edge in hypergraph.edges}

        # Build table of incoming edges
        in_edges = defaultdict(lambda: [])
        for edge in hypergraph.edges:
            for node in edge.tail:
                in_edges[node.id].append(edge)

        # max \theta x

        prob += sum([weights[edge] * edge_vars[edge.id]
                     for edge in hypergraph.edges])

        # x(r) = 1
        prob += node_vars[hypergraph.root.id] == 1

        # x(v) = \sum_{e : h(e) = v} x(e)
        for node in hypergraph.nodes:
            if node.is_terminal(): continue
            prob += node_vars[node.id] == sum([edge_vars[edge.id]
                                            for edge in node.edges])

        # x(v) = \sum_{e : v \in t(e)} x(e)
        for node in hypergraph.nodes:
            if node.id == hypergraph.root.id: continue
            prob += node_vars[node.id] == sum([edge_vars[edge.id]
                                            for edge in in_edges[node.id]])

        return HypergraphLP(prob, hypergraph, node_vars, edge_vars)
