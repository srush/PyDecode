import pydecode
from collections import defaultdict
from itertools import izip
import scipy.sparse
import numpy as np

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

        Parameters
        -----------

        graph :
            The associated hypergraph.

        """
        self.graph = graph

        self.bias = np.zeros([len(constraints), 1])
        self.by_label = {}
        self.by_id = {}
        for i, (label, constant) in enumerate(constraints):
            self.by_label[label] = i
            self.by_id[i] = label
            self.bias[i] = constant
            #self.bias.append((i, constant))
        self.size = len(constraints)
        #self.potentials = ph.SparseVectorPotentials(graph)

    def check(self, path):
        constraints = self.constraint_matrix * path.v + self.bias
        print constraints.shape
        print self.bias.shape
        #print constraints
        print constraints
        return [self.by_id[i] for i in range(self.size)
                if int(constraints[i, 0]) != 0]

    def name(self, cons_id):
        return self.by_id[cons_id]

    def id(self, label):
        return self.by_label[label]

    def from_vector(self, vector):
        final = []

        self.all_constraints = defaultdict(lambda: [])
        data = []
        indices = []
        cutoff = [0]
        for i, (edge, cons) in enumerate(izip(self.graph.edges, vector)):
            semi = []
            for name, coeff in cons:
                constraint = self.by_label[name]
                semi.append((constraint, coeff))
                self.all_constraints[constraint].append((coeff, edge))
                data.append(coeff)
                indices.append(constraint)

            cutoff.append(len(data))
            semi.sort()
            final.append(semi)

        self.constraint_matrix = \
            scipy.sparse.csc_matrix(
            (data, indices, cutoff),
            shape=(len(self.all_constraints), len(self.graph.edges)),
            dtype=np.uint8)
        #self.potentials.from_vector(final, bias=self.bias)
        return self

    def __iter__(self):
        return self.all_constraints.itervalues()


#     def to_binary_potentials(self):
#         vector = []
#         for edge in self.graph.edges:
#             b = pydecode.Bitset()
#             for i, v in self.potentials[edge]:
#                 assert(v == 1)
#                 b[i] = 1
#             vector.append(b)
#         return pydecode.BinaryVectorPotentials(self.graph).from_vector(vector)

# class WeightedConstrainedGraph:
#     def __init__(self, graph, weight, constraint):
#         self.graph = graph
#         self.weight = weight
#         self.constraint = constraint

#     def project(self, range_hypergraph, projection):
#         return WeightedConstrainedGraph(range_hypergraph,
#                                         projection[self.weight],
#                                         projection[self.constraint])

#     def up_project(self, projection):
#         return WeightedConstrainedGraph(
#             projection.domain_hypergraph,
#             self.weight.up_project(projection.domain_hypergraph, projection),
#             self.constraint.up_project(projection.domain_hypergraph, projection))



# class Constraint:
#     def __init__(self, name, variables, coeffs, bias):
#         self.name = name
#         self.variables = variables
#         self.coeffs = coeffs
#         self.bias = bias


# class Variables():
#     def __init__(self, graph, num_variables, constraints):
#         self.graph = graph
#         self.variables = num_variables
#         self.potentials = pydecode.BinaryVectorPotentials(graph)
#         self.constraints = constraints

#     # def build(self, build):
#     #     self.potentials.build(build)
#     #     return self

#     def from_vector(self, vector):
#         #vector = build(self.graph)
#         self.potentials.from_vector(vector)
#         return self

#     def check(self, path):
#         sparse_vars = self.potentials.dot(path)
#         #print sparse_vars
#         #d = dict(sparse_vars)
#         for cons in self.constraints:
#             val = cons.bias
#             for var, coeff in zip(cons.variables, cons.coeffs):
#                 if sparse_vars[var]:
#                     val += coeff
#             if val != 0:
#                 yield cons.name
