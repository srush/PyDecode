from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "Hypergraph/Hypergraph.h":
    cdef cppclass CHyperedge "Hyperedge":
        int id()
        const CHypernode *head_node()
        vector[const CHypernode *] tail_nodes()


    cdef cppclass CHypernode "Hypernode":
        int id()
        vector[const CHyperedge *] edges()

    cdef cppclass CHypergraph "Hypergraph":
        CHypergraph()
        const CHypernode *root()
        const CHypernode *start_node()
        const CHypernode *add_terminal_node()
        void end_node()
        int id()
        const CHyperedge *add_edge(vector[const CHypernode *]) except +
        void finish() except +
        vector[const CHypernode *] nodes()
        vector[const CHyperedge *] edges()

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[const CHyperedge *] edges) except +
        vector[const CHyperedge *] edges()
        vector[const CHypernode *] nodes()
        int has_edge(const CHyperedge *)
        bool equal(const CHyperpath path)

cdef class Labeling:
    cdef edge_labels
    cdef node_labels

cdef class _LazyEdges:
    cdef Hypergraph _graph
    cdef vector[const CHyperedge *] _edges
    cdef init(self, vector[const CHyperedge *])

cdef class _LazyNodes:
    cdef Hypergraph _graph
    cdef vector[const CHypernode *] _nodes
    cdef init(self, vector[const CHypernode *])

cdef class Hypergraph:
    cdef CHypergraph *thisptr
    cdef Labeling labeling
    cdef _cached_edges

    cdef Hypergraph init(self, const CHypergraph *ptr, Labeling labeling)

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

cdef extern from "Hypergraph/Map.h":
    cdef cppclass CHypergraphMap "HypergraphMap":
        const CHyperedge *map(const CHyperedge *edge)
        const CHypernode *map(const CHypernode *node)
        const CHypergraph *domain_graph()
        const CHypergraph *range_graph()
        CHypergraphMap *invert()
        CHypergraphMap *compose(const CHypergraphMap &)

cdef class HypergraphMap:
    cdef const CHypergraphMap *thisptr
    cdef Hypergraph range_graph
    cdef Hypergraph domain_graph

    cdef HypergraphMap init(self, const CHypergraphMap *thisptr,
                            Hypergraph range_graph, Hypergraph domain_graph)
