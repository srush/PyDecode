import pydecode.hyper as ph
from collections import defaultdict

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
        return [self.by_id[i]
                for i, val in constraints.iteritems()
                if val != 0]

    def build(self, builder):
        """
        build(constraints, builder)

        Build the constraints for a hypergraph.

        Parameters
        -----------

        constraints :
            A list of pairs of the form (label, constant) indicating a constraint.

        builder :
            A function from edge label to a list of pairs (constraints, coeffificient).

        Returns
        ------------

        :py:class:`Constraints`
            The constraints.
        """
        self.all_constraints = defaultdict(lambda: [])
        edge_values = {}
        for edge in self.hypergraph.edges:
            semi = []
            label = self.hypergraph.label(edge)
            for name, coeff in builder(label):
                constraint = self.by_label[name]
                semi.append((constraint, coeff))
                self.all_constraints[constraint].append((coeff, edge))
            edge_values[label] = semi

        self.potentials.build(lambda a: edge_values[a], bias=self.bias)
        return self

    def __iter__(self):
        return self.all_constraints.itervalues()

        # cdef CConstraint *cons
        # cdef Constraint hcons
        # by_label = {}
        # cdef string label
        # for label, constant in constraints:
        #     cons = self.thisptr.add_constraint(label)
        #     hcons = Constraint()
        #     hcons.init(cons)
        #     cons.set_constant(constant)
        #     by_label[label] = hcons

        # cdef vector[const CHyperedge *] edges = \
        #     self.hypergraph.thisptr.edges()
        # cdef int coeff
        # cdef Constraint constraint
        # for i, ty in enumerate(self.hypergraph.edge_labels):
        #     constraints = builder(ty)

        #     for term in constraints:
        #         if term is None: continue

        #         try:
        #             label, coeff = term
        #         except:
        #             raise HypergraphConstructionException(
        #                 "Term must be a pair of the form (label, coeff)."  + \
        #                     "Given %s."%(term))
        #         try:
        #             constraint = <Constraint> by_label[label]
        #         except:
        #             raise HypergraphConstructionException(
        #                 "Label %s is not a valid label"%label)
        #         (<CConstraint *>constraint.thisptr).add_edge_term(edges[i], coeff)
        # return self

    # def __iter__(self):
    #     return iter(convert_constraints(self.thisptr.constraints()))

    # def check(self, Path path):
    #     """
    #     check(path)

    #     Check which constraints a path violates.

    #     Parameters
    #     ----------------

    #     path : :py:class:`Path`
    #        The hyperpath to check.


    #     Returns
    #     -------------

    #     A list of :py:class:`Constraint` objects
    #         The violated constraints.
    #     """

    #     cdef vector[const CConstraint *] failed
    #     cdef vector[int] count
    #     self.thisptr.check_constraints(deref(path.thisptr),
    #                                    &failed,
    #                                    &count)
    #     return convert_constraints(failed)
