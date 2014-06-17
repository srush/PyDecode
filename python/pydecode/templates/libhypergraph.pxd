from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool

cdef extern from "Hypergraph/Hypergraph.hh":
    cdef cppclass CHypernode "Hypernode":
        int id()
        vector[int ] edges()

    cdef cppclass CHypergraph "Hypergraph":
        CHypergraph(bool)
        void set_expected_size(int, int, int)
        const CHypernode *root()
        int tail_nodes(int)
        const CHypernode *tail_node(int, int)
        const CHypernode *head(int)
        const CHypernode *start_node()
        const CHypernode *add_terminal_node()
        bool end_node()
        int id()
        int new_id(int)
        int add_edge(vector[const CHypernode *]) except +
        void finish(bool reconstruct) except +
        vector[const CHypernode *] nodes()
        vector[int ] edges()

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[int] edges) except +
        vector[int] edges()
        vector[const CHypernode *] nodes()
        int has_edge(int)
        bool equal(const CHyperpath path)

cdef class Labeling:
    cdef edge_labels
    cdef node_labels

cdef class _LazyEdges:
    cdef Hypergraph _graph
    cdef vector[int] _edges
    cdef init(self, vector[int])

cdef class _LazyVertices:
    cdef Hypergraph _graph
    cdef vector[const CHypernode *] _nodes
    cdef init(self, vector[const CHypernode *])

cdef class Hypergraph:
    cdef CHypergraph *thisptr
    cdef Labeling labeling
    cdef _cached_edges
    cdef bool unary
    cdef Hypergraph init(self, const CHypergraph *ptr, Labeling labeling)

cdef class Vertex:
    cdef const CHypernode *nodeptr
    cdef CHypergraph *graphptr
    cdef Hypergraph graph

    cdef Vertex init(self, const CHypernode *nodeptr,
                   Hypergraph graph)

cdef class Edge:
    cdef int edgeptr
    cdef Hypergraph graph

    cdef Edge init(self, int edge, Hypergraph graph, bool unfinished=?)
    cdef bool unfinished

cdef class Path:
    cdef const CHyperpath *thisptr
    cdef Hypergraph graph
    cdef vector

    cdef Path init(self, const CHyperpath *path, Hypergraph graph)
    cdef public equal(Path self, Path other)

cdef extern from "Hypergraph/Map.hh":
    cdef cppclass CHypergraphMap "HypergraphMap":
        int map(int)
        const CHypernode *map(const CHypernode *node)
        const CHypergraph *domain_graph()
        const CHypergraph *range_graph()
        const vector[int] edge_map()
