from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "Hypergraph/Algorithms.h":
    Hyperpath *viterbi_path(const Hypergraph *graph,
                            const HypergraphWeights theta,
                            vector[double] *chart)

cdef extern from "Hypergraph/Hypergraph.h":
    cdef cppclass Hyperedge:
        string label()
        int id()
        const Hypernode *head_node()
        vector[const Hypernode *] *tail_nodes()

    cdef cppclass Hypernode:
        vector[const Hyperedge *] *edges()

    cdef cppclass Hypergraph:
        Hypergraph()
        const Hypernode *root()
        const Hypernode *start_node()
        vector[const Hypernode *] nodes()
        vector[const Hypernode *] edges()
        void end_node()
        const Hyperedge *add_edge(vector[const Hypernode *],
                                  string label)
        void finish()

    cdef cppclass Hyperpath:
        vector[const Hyperedge *] edges()


    cdef cppclass HypergraphWeights:
        HypergraphWeights(const Hypergraph *hypergraph,
                          const vector[double] weights,
                          double bias)
        double dot(const Hyperpath &path)

cdef extern from "Hypergraph/Constraints.h":
    cdef cppclass Constraint:
        Constrint(string label, int id)

    cdef cppclass HypergraphConstraints:
        HypergraphConstraints(const Hypergraph *hypergraph)
        Constraint *add_constraint(string label)
        int check_constraints(const Hyperpath path,
                              vector[const Constraint *] *failed,
                              vector[int] *count
                              )


def viterbi(HGraph graph, Weights weights):
    cdef vector[double] chart
    viterbi_path(graph.thisptr, deref(weights.thisptr), &chart)

cdef class HGraph:
    cdef Hypergraph *thisptr
    def __cinit__(self):
        self.thisptr = new Hypergraph()

    def builder(self):
        gb = GraphBuilder()
        gb.init(self.thisptr)
        return gb

    def edges_size(self):
        return self.thisptr.edges().size()

    def nodes_size(self):
        return self.thisptr.nodes().size()

cdef class GraphBuilder:
    cdef Hypergraph *thisptr

    cdef init(self, Hypergraph *ptr):
        self.thisptr = ptr

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        self.thisptr.finish()

    def add_terminal_node(self):
        node = Node()
        cdef const Hypernode *nodeptr = self.thisptr.start_node()
        self.thisptr.end_node()
        node.init(self.thisptr, nodeptr)
        return node

    def add_node(self):
        node = Node()
        cdef const Hypernode *nodeptr = self.thisptr.start_node()
        node.init(self.thisptr, nodeptr)
        return node

cdef class Node:
    cdef const Hypernode *nodeptr
    cdef Hypergraph *graphptr
    cdef int edge_count

    def __cinit__(self):
        self.edge_count = 0

    cdef init(self, Hypergraph *graphptr, const Hypernode *nodeptr):
        self.graphptr = graphptr
        self.nodeptr = nodeptr

    def __cinit__(self):
        pass

    def __enter__(self):
        return self


    def add_edge(self, tail_nodes, label):
        cdef vector[const Hypernode *] tail_node_ptrs
        for  tail_node in tail_nodes:
            tail_node_ptrs.push_back((<Node> tail_node).nodeptr)
        edgeptr = self.graphptr.add_edge(tail_node_ptrs, label)
        self.edge_count += 1
        edge = Edge()
        edge.init(edgeptr)
        return edge

    def __exit__(self, a, b, c):
        print "exit"
        if self.edge_count == 0:
            assert False
        self.graphptr.end_node()


cdef class Edge:
    cdef const Hyperedge *thisptr

    def __cinit__(self):
        pass

    cdef init(self, const Hyperedge *ptr):
        self.thisptr = ptr


cdef class Path:
    cdef const Hyperpath *thisptr

cdef class Weights:
    cdef const HypergraphWeights *thisptr
    def __cinit__(self, HGraph hypergraph, vector[double] weights, double bias):
        self.thisptr = new HypergraphWeights(hypergraph.thisptr, weights, bias)

    def dot(self, Path path):
        cdef double result = self.thisptr.dot(deref(path.thisptr))
        return result

cdef class WeightBuilder:
    cdef vals
    cdef hypergraph
    def __cinit__(self, hypergraph):
        self.vals = {}
        self.hypergraph = hypergraph

    def set_weight(self, Edge edge, val):
        self.vals[edge.thisptr.id()] = val

    def weights(self):
        cdef vector[double] weights
        weights.resize(self.hypergraph.edges_size(), 0)
        for i, w in self.vals.iteritems():
            weights[i] = w
        return Weights(self.hypergraph, weights, 0.0)

cdef class HConstraints:
    cdef HypergraphConstraints *thisptr
    def __cinit__(self, HGraph hypergraph):
        self.thisptr = new HypergraphConstraints(hypergraph.thisptr)

    def check(self, Path path):
        cdef vector[const Constraint *] failed
        cdef vector[int] count
        self.thisptr.check_constraints(deref(path.thisptr), &failed, &count)

# cdef class Hyperedge:
#     cdef HyperedgeImpl *thisptr

#     def __cinit__(self, label,
#                   features,
#                   int id,
#                   tails,
#                   Hypernode head):
#         self.init(label, features, id, tails, head)

#     cdef init(self, string label, string features,
#               int id, tails, Hypernode head):
#         ptrs = [tail.thisptr for tail in tails]
#         cdef vector[HypernodeImpl *] tail_ptrs
#         for tail in tails:
#             tail_ptrs.push_back((<Hypernode>tail).thisptr)
#         self.thisptr = new HyperedgeImpl(label, features, id,
#                                          tail_ptrs, head.thisptr)

# cdef class Hypernode:
#     cdef HypernodeImpl *thisptr
#     def __cinit__(self, label, id):
#         self.thisptr = new HypernodeImpl(label, id)

#     def __dealloc__(self):
#         del self.thisptr
