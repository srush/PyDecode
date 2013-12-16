from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

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
                   const vector[const CHyperedge *] edges) except +
        vector[const CHyperedge *] edges()
        int has_edge(const CHyperedge *)
        bool equal(const CHyperpath path)

    cdef cppclass CHypergraphWeights "HypergraphWeights<double>":
        double dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphWeights *project_weights(
            const CHypergraphProjection )
        CHypergraphWeights(const CHypergraph *hypergraph,
                           const vector[double] weights,
                           double bias) except +
