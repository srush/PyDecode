"""
Linear programming (LP) library for combinatorial search.
Requires the gurobi library.
"""

from gurobipy import *
import pydecode.hyper as ph
from collections import defaultdict


class HypergraphLP:
    """
    Manages the linear program for a hypergraph search problem.

    Attributes
    -----------

    objective : float
       Get the objective of the solved LP.

    path : The hyperpath (if ILP)
       Get the hyperpath of the solved ILP.

    edge_variables : Dict of edge values.
       Get the (fractional) path of the solved LP.

    """

    def __init__(self, lp, hypergraph, node_vars, edge_vars,
                 integral=False):
        r"""
        Initialize the Hypergraph LP.

        Call with HypergraphLP.make_lp().

        Parameters
        ------------

        lp : PuLP linear program
           The linear program for the hypergraph

        hypergraph : :py:class:`Hypergraph`
           The hypergraph.

        node_vars : map of Node to LP variable.
           The node variables :math:`y(v)`
           for all :math:`v \in {\cal V}`.

        edge_vars : map of Edge to LP variable.
           The hyperedge variables :math:`y(e)`
           for all :math:`e \in {\cal E}`.

        integral : bool
        """

        self.lp = lp
        self.hypergraph = hypergraph
        self.node_vars = node_vars
        self.edge_vars = edge_vars
        self.integral = integral

    def solve(self, solver=None):
        r"""
        Solve the underlying hypergraph linear program
        and return the best path.

         :math:`\arg \max_{y \in {\cal X}} \theta^{\top}y`.

        Parameters
        ----------

        solver : LP solver
           A PuLP LP solver (glpsol, Gurobi, etc.).
        """
        # if solver is None:
        #     _solver = pulp.solvers.GLPK(msg=0)
        # else:
        #     _solver = solver
        # self._status = self.lp.solve(_solver)
        self.lp.optimize()

    @property
    def path(self):
        # if self._status != pulp.LpStatusOptimal:
        #     raise Exception("No optimal solution.")
        # else:
        path_edges = [edge
                      for edge in self.hypergraph.edges
                      if self.edge_vars[edge.id].x == 1.0]
        return ph.Path(self.hypergraph, path_edges)

    @property
    def objective(self):
        return self.lp.objective

    @property
    def edge_variables(self):
        return {edge: self.edge_vars[edge.id].x
                for edge in self.hypergraph.edges
                if self.edge_vars[edge.id].x > 0.0}

    # def decode_fractional(self):
    #     vec = [self.edge_vars[edge.id])
    #            for edge in self.hypergraph.edges]
    #     weights = ph.LogViterbiPotentials(self.hypergraph).from_vector(vec)
    #     return ph.best_path(self.hypergraph, weights)

    def add_constraints(self, constraints):
        """
        Add hard constraints to the hypergraph.

        Parameters
        -----------

        constraints : :py:class:`Constraints`
        """
        for i, constraint in enumerate(constraints):
            self.lp.addConstr(0 == \
                constraints.bias[i][1] + \
                quicksum([coeff * self.edge_vars[edge.id]
                            for (coeff, edge) in constraint]))

    @staticmethod
    def make_lp(hypergraph, potentials, name="", integral=False):
        r"""
        Construct a linear program from a hypergraph.

        .. math::
          \max \theta^{\top} x \\
          x(1) = 1 \\
          x(v) = \sum_{e \in {\cal E} : h(e) = v} x(e) \\
          x(v) = \sum_{e \in {\cal E} : v \in t(e)} x(e)

        Parameters
        ----------

        hypergraph : :py:class:`pydecode.hyper.Hypergraph`
          The hypergraph search.


        potentials : :py:class:`pydecode.hyper.LogViterbiPotentials`
          The potentials.

        name : string
          A debugging name for linear program.

        integral : bool
          Construct as an integer linear program.

        Returns
        --------

        lp : :py:class:`HypergraphLP`
          Returns the hypergraph LP (or ILP)
        """
        
        if integral:
            var_type = GRB.BINARY
        else:
            var_type = GRB.CONTINUOUS
        prob = Model("Hypergraph Problem")

        def node_name(node):
            return "node_{}".format(node.id)

        def edge_name(edge):
            return "edge_{}".format(edge.id)

        # Make variables for the nodes.
        node_vars = {node.id: prob.addVar(name=node_name(node),
                                          vtype=var_type)
                     for node in hypergraph.nodes}

        edge_vars = {edge.id: prob.addVar(name=edge_name(edge), 
                                          vtype=var_type)
                     for edge in hypergraph.edges}

        prob.update()

        # Build table of incoming edges
        in_edges = defaultdict(lambda: [])
        for edge in hypergraph.edges:
            for node in edge.tail:
                in_edges[node.id].append(edge)

        # for edge in hypergraph.edges:
        #     p = potentials[edge]
        #     v = edge_vars[edge.id]

        # max \theta x
        prob.setObjective(quicksum(
            [potentials[edge] * edge_vars[edge.id]
             for edge in hypergraph.edges]) + potentials.bias, GRB.MAXIMIZE)

        # x(r) = 1
        prob.addConstr(node_vars[hypergraph.root.id] == 1)

        # x(v) = \sum_{e : h(e) = v} x(e)
        for node in hypergraph.nodes:
            if node.is_terminal:
                continue
            prob.addConstr(node_vars[node.id] == \
                quicksum((edge_vars[edge.id]
                            for edge in node.edges)))

        # x(v) = \sum_{e : v \in t(e)} x(e)
        for node in hypergraph.nodes:
            if node.id == hypergraph.root.id:
                continue
            prob.addConstr(node_vars[node.id] == \
                quicksum((edge_vars[edge.id]
                            for edge in in_edges[node.id])))

        return HypergraphLP(prob, hypergraph, node_vars,
                            edge_vars, integral)
