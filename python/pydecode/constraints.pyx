#cython: embedsignature=True
# from wrap cimport *
# from hypergraph cimport *
# from hypergraph import *
from cython.operator cimport dereference as deref

cdef convert_constraints(vector[const CConstraint *] c):
    return [Constraint().init(cresult) for cresult in c]


cdef class Constraint:
    r"""
    A single linear hypergraph constraint, for instance the i'th constraint
    :math:`A_i y = b_i`.

    Can get the term of an edge, :math:`A_{i, e}`,  using ::

       >> constraint[edge]

    Or iterate through the contraints with ::

       >> for term, edge in constraint:
       >>    print term, edge

    .. :math:`\{(A_{i,e}, e) : e \in {\cal E}, A_{i,e} \neq 0\}`
    Attributes
    -----------
    label : string
        The label of the constraint.

    constant : int
        The constraint constant :math:`b_i`
    """
    cdef const CConstraint *thisptr
    cdef Constraint init(self, const CConstraint *ptr):
        self.thisptr = ptr
        return self

    def __str__(self): return self.thisptr.label

    property label:
        def __get__(self):
            return self.thisptr.label

    property constant:
        def __get__(self):
            return self.thisptr.bias

    def __iter__(self):
        edges = convert_edges(self.thisptr.edges)
        return iter(zip(self.thisptr.coefficients, edges))

    def __contains__(self, Edge edge):
        return self.thisptr.has_edge(edge.edgeptr)

    def __getitem__(self, Edge edge):
        return self.thisptr.get_edge_coefficient(edge.edgeptr)

cdef class Constraints:
    r"""
    Stores linear hyperedge constraints of the form :math:`A y = b`,
    where :math:`A \in R^{|b| \times |{\cal E}|}` and :math:`b \in
    R^{|b|}`.

    To iterate through individual constraints  ::

       >> [c for c in constraints]

    """
    cdef CHypergraphConstraints *thisptr
    cdef Hypergraph hypergraph
    def __cinit__(self, Hypergraph graph):
        """
        A set of constraints associated with a hypergraph.

        :param graph: The associated hypergraph.
        :type graph: :py:class:`Hypergraph`
        """
        self.thisptr = new CHypergraphConstraints(graph.thisptr)
        self.hypergraph = graph

    def build(self, constraints, builder):
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
        cdef CConstraint *cons
        cdef Constraint hcons
        by_label = {}
        cdef string label
        for label, constant in constraints:
            cons = self.thisptr.add_constraint(label)
            hcons = Constraint()
            hcons.init(cons)
            cons.set_constant(constant)
            by_label[label] = hcons

        cdef vector[const CHyperedge *] edges = \
            self.hypergraph.thisptr.edges()
        cdef int coeff
        cdef Constraint constraint
        for i, ty in enumerate(self.hypergraph.edge_labels):
            constraints = builder(ty)

            for term in constraints:
                if term is None: continue

                try:
                    label, coeff = term
                except:
                    raise HypergraphConstructionException(
                        "Term must be a pair of the form (label, coeff)."  + \
                            "Given %s."%(term))
                try:
                    constraint = <Constraint> by_label[label]
                except:
                    raise HypergraphConstructionException(
                        "Label %s is not a valid label"%label)
                (<CConstraint *>constraint.thisptr).add_edge_term(edges[i], coeff)
        return self

    def __iter__(self):
        return iter(convert_constraints(self.thisptr.constraints()))

    def check(self, Path path):
        """
        check(path)

        Check which constraints a path violates.

        Parameters
        ----------------

        path : :py:class:`Path`
           The hyperpath to check.


        Returns
        -------------

        A list of :py:class:`Constraint` objects
            The violated constraints.
        """

        cdef vector[const CConstraint *] failed
        cdef vector[int] count
        self.thisptr.check_constraints(deref(path.thisptr),
                                       &failed,
                                       &count)
        return convert_constraints(failed)
