from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "Hypergraph/Algorithms.h":
    CHyperpath *viterbi_path(
        const CHypergraph *graph,
        const CHypergraphWeights theta,
        vector[double] *chart) except +

cdef extern from "Hypergraph/Algorithms.h":
    void outside(
        const CHypergraph *graph,
        const CHypergraphWeights weights,
        const vector[double] inside_chart,
        vector[double] *chart) except +

    const CHypergraphProjection *prune(
        const CHypergraph *original,
        const CHypergraphWeights weights,
        double ratio) except +

    cdef cppclass CConstrainedResult "ConstrainedResult":
        const CHyperpath *path
        double dual
        double primal
        vector[const CConstraint *] constraints

    const CHyperpath *best_constrained_path(
        const CHypergraph *graph,
        const CHypergraphWeights theta,
        const CHypergraphConstraints constraints,
        vector[CConstrainedResult] *constraints,
        ) except +

    cdef cppclass CMaxMarginals "MaxMarginals":
        const CHyperpath *path
        double max_marginal(const CHyperedge *edge)
        double max_marginal(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "MaxMarginals":
    CMaxMarginals *compute(const CHypergraph *hypergraph,
                           const CHypergraphWeights *weights)

cdef extern from "Hypergraph/Hypergraph.h":
    cdef cppclass CHyperedge "Hyperedge":
        string label()
        int id()
        const CHypernode *head_node()
        vector[const CHypernode *] tail_nodes()

    cdef cppclass CHypernode "Hypernode":
        vector[const CHyperedge *] edges()
        string label()
        int id()

    cdef cppclass CHypergraph "Hypergraph":
        CHypergraph()
        const CHypernode *root()
        const CHypernode *start_node(string)
        const CHypernode *add_terminal_node(string)
        vector[const CHypernode *] nodes()
        vector[const CHyperedge *] edges()
        void end_node()
        const CHyperedge *add_edge(vector[const CHypernode *],
                                   string label) except +
        void finish() except +

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[const CHyperedge *] edges)
        vector[const CHyperedge *] edges()
        int has_edge(const CHyperedge *)

    cdef cppclass CHypergraphWeights "HypergraphWeights<double>":
        double dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphWeights *project_weights(
            const CHypergraphProjection )
        CHypergraphWeights(const CHypergraph *hypergraph,
                           const vector[double] weights,
                           double bias) except +

    cdef cppclass CHypergraphProjection "HypergraphProjection":
        const CHypergraph *new_graph
        const CHyperedge *project(const CHyperedge *edge)
        const CHypernode *project(const CHypernode *node)

cdef extern from "Hypergraph/Constraints.h":
    cdef cppclass CConstraint "Constraint":
        Constraint(string label, int id)
        void set_constant(int _bias)
        int has_edge(const CHyperedge *edge)
        int get_edge_coefficient(const CHyperedge *edge)
        void add_edge_term(const CHyperedge *edge, int coefficient)
        string label
        vector[const CHyperedge *] edges
        vector[int] coefficients
        int bias


    cdef cppclass CHypergraphConstraints "HypergraphConstraints":
        CHypergraphConstraints(const CHypergraph *hypergraph)
        CConstraint *add_constraint(string label)
        const CHypergraph *hypergraph()
        int check_constraints(const CHyperpath path,
                              vector[const CConstraint *] *failed,
                              vector[int] *count)
        const vector[const CConstraint *] constraints()
