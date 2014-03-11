import pydecode.hyper as ph
from collections import defaultdict

class WeightedConstrainedGraph:
    def __init__(self, graph, weight, constraint):
        self.graph = graph
        self.weight = weight
        self.constraint = constraint

    def project(self, range_hypergraph, projection):
        return WeightedConstrainedGraph(range_hypergraph,
                                        projection[self.weight],
                                        projection[self.constraint])

    def up_project(self, projection):
        return WeightedConstrainedGraph(
            projection.domain_hypergraph,
            self.weight.up_project(projection.domain_hypergraph, projection),
            self.constraint.up_project(projection.domain_hypergraph, projection))



class Constraint:
    def __init__(self, name, variables, coeffs, bias):
        self.name = name
        self.variables = variables
        self.coeffs = coeffs
        self.bias = bias


class Variables():
    def __init__(self, graph, num_variables, constraints):
        self.hypergraph = graph
        self.variables = num_variables
        self.potentials = ph.BinaryVectorPotentials(graph)
        self.constraints = constraints

    # def build(self, build):
    #     self.potentials.build(build)
    #     return self

    def from_vector(self, vector):
        #vector = build(self.hypergraph)
        self.potentials.from_vector(vector)
        return self

    def check(self, path):
        sparse_vars = self.potentials.dot(path)
        print sparse_vars
        #d = dict(sparse_vars)
        for cons in self.constraints:
            val = cons.bias
            for var, coeff in zip(cons.variables, cons.coeffs):
                if sparse_vars[var]:
                    val += coeff
            if val != 0:
                yield cons.name


class Constraints:
    r"""
    Stores linear hyperedge constraints of the form :math:`A y = b`,
    where :math:`A \in R^{|b| \times |{\cal E}|}` and :math:`b \in
    R^{|b|}`.

    To iterate through individual constraints  ::

       >> [c for c in constraints]

    """
    def __init__(self, graph, constraints):
        """
        A set of constraints associated with a hypergraph.

        :param graph: The associated hypergraph.
        :type graph: :py:class:`Hypergraph`
        """
        self.hypergraph = graph

        self.bias = []
        self.by_label = {}
        self.by_id = {}
        for i, (label, constant) in enumerate(constraints):
            self.by_label[label] = i
            self.by_id[i] = label
            self.bias.append((i, constant))
        self.size = len(constraints)
        self.potentials = ph.SparseVectorPotentials(graph)

    def check(self, path):
        # for edge in path.edges:
        #     print edge.id, self.potentials[edge]
        constraints = self.potentials.dot(path)
        #print "Constraints", constraints
        return [self.by_id[i] for i, val in constraints
                if val != 0]

    def name(self, cons_id):
        return self.by_id[cons_id]

    def id(self, label):
        return self.by_label[label]

    def from_vector(self, vector):
        final = []
        self.all_constraints = defaultdict(lambda: [])
        edge_values = {}
        for i, edge in enumerate(self.hypergraph.edges):
            semi = []
            for name, coeff in vector[i]:
                constraint = self.by_label[name]
                semi.append((constraint, coeff))
                self.all_constraints[constraint].append((coeff, edge))
            final.append(semi)

        self.potentials.from_vector(final, bias=self.bias)
        return self

    def __iter__(self):
        return self.all_constraints.itervalues()
