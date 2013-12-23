
from wrap cimport *

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
