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
        int id()
        const CHyperedge *add_edge(vector[const CHypernode *],
                                   string label) except +
        void finish() except +

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[const CHyperedge *] edges) except +
        vector[const CHyperedge *] edges()
        vector[const CHypernode *] nodes()
        int has_edge(const CHyperedge *)
        bool equal(const CHyperpath path)

cdef class Hypergraph:
    cdef CHypergraph *thisptr
    cdef edge_labels
    cdef node_labels
    cdef _cached_edges

    cdef Hypergraph init(self, const CHypergraph *ptr, nodes, edges)


cdef class GraphBuilder:
    cdef CHypergraph *thisptr
    cdef Hypergraph graph
    cdef edge_labels
    cdef node_labels
    cdef started

    cdef GraphBuilder init(self, Hypergraph hyper, CHypergraph *ptr)

cdef class Node:
    cdef const CHypernode *nodeptr
    cdef CHypergraph *graphptr
    cdef Hypergraph graph

    cdef Node init(self, const CHypernode *nodeptr,
                   Hypergraph graph)

cdef class Edge:
    cdef const CHyperedge *edgeptr
    cdef Hypergraph graph

    cdef Edge init(self, const CHyperedge *ptr, Hypergraph graph)

cdef class Path:
    cdef const CHyperpath *thisptr
    cdef Hypergraph graph

    cdef Path init(self, const CHyperpath *path, Hypergraph graph)
    cdef public equal(Path self, Path other)
