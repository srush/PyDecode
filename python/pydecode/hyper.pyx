from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "Hypergraph/Algorithms.h":
    CHyperpath *viterbi_path(
        const CHypergraph *graph,
        const CHypergraphWeights theta,
        vector[double] *chart)

    void outside(
        const CHypergraph *graph,
        const CHypergraphWeights weights,
        const vector[double] inside_chart,
        vector[double] *chart)
    
    CHyperpath *best_constrained_path(
        const CHypergraph *graph,
        const CHypergraphWeights theta,
        const CHypergraphConstraints constraints)

cdef extern from "Hypergraph/Hypergraph.h":
    cdef cppclass CHyperedge "Hyperedge":
        string label()
        int id()
        const CHypernode *head_node()
        vector[const CHypernode *] tail_nodes()

    cdef cppclass CHypernode "Hypernode":
        vector[const CHyperedge *] edges()
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
                                  string label)
        void finish()

    cdef cppclass CHyperpath "Hyperpath":
        vector[const CHyperedge *] edges()
        int has_edge(const CHyperedge *)

    cdef cppclass CHypergraphWeights "HypergraphWeights":
        CHypergraphWeights(const CHypergraph *hypergraph,
                           const vector[double] weights,
                           double bias)
        double dot(const CHyperpath &path)

cdef extern from "Hypergraph/Constraints.h":
    cdef cppclass CConstraint "Constraint":
        Constrint(string label, int id)
        void set_constant(int _bias)
        void add_edge_term(const CHyperedge *edge, int coefficient)
        string label

    cdef cppclass CHypergraphConstraints "HypergraphConstraints":
        CHypergraphConstraints(const CHypergraph *hypergraph)
        CConstraint *add_constraint(string label)
        int check_constraints(const CHyperpath path,
                              vector[const CConstraint *] *failed,
                              vector[int] *count)

cdef class Chart:
    cdef vector[double] chart

    def score(self, Node node):
        return self.chart[node.id()]

def best_path(Hypergraph graph, 
              Weights weights):
    cdef Chart chart = Chart()
    cdef const CHyperpath *hpath = \
        viterbi_path(graph.thisptr, 
                     deref(weights.thisptr), 
                     &chart.chart)
    cdef Path path = Path()
    path.init(hpath)
    return path, chart

def outside_path(Hypergraph graph, 
                 Weights weights, 
                 Chart inside_chart):
    cdef Chart out_chart = Chart()
    outside(graph.thisptr, deref(weights.thisptr), 
            inside_chart.chart, &out_chart.chart)
    return out_chart

def best_constrained(Hypergraph graph, 
                     Weights weights, 
                     Constraints constraints):
    best_constrained_path(graph.thisptr, 
                          deref(weights.thisptr), 
                          deref(constraints.thisptr))
    
cdef class Hypergraph:
    cdef CHypergraph *thisptr
    cdef registered

    def __cinit__(self):
        self.thisptr = new CHypergraph()
        self.registered = []
    
    def register(self, builder, attr):
        self.registered.append((builder, attr))

    def builder(self):
        gb = GraphBuilder()
        gb.init(self.thisptr, self.registered)
        return gb

    def edges_size(self):
        return self.thisptr.edges().size()

    def nodes_size(self):
        return self.thisptr.nodes().size()

    def edges(self):
        return convert_edges(self.thisptr.edges())

    def nodes(self):
        return convert_nodes(self.thisptr.nodes())

    def root(self):
        return convert_node(self.thisptr.root())

# cdef class WeightedHypergraph(Hypergraph):
#     def builder(self):
#         gb = GraphBuilder()
#         gb.init(self.thisptr)
#         return gb


cdef class GraphBuilder:
    cdef CHypergraph *thisptr
    cdef table
    cdef registered
    
    cdef init(self, CHypergraph *ptr, registered):
        self.thisptr = ptr
        self.registered = registered
        self.table = {}
        for builder, attr in self.registered:
            self.table[attr] = []

    def __enter__(self): return self

    def __exit__(self, exception, b, c):
        if exception:
           return False
        self.thisptr.finish()
        for builder, attr in self.registered:
            builder.build(self.table[attr])
        
    def add_node(self, label = "", terminal = False):
        self.thisptr.end_node()
        node = Node()
        cdef const CHypernode *nodeptr 
        if terminal:
            nodeptr = self.thisptr.add_terminal_node(label)
        else:
            nodeptr = self.thisptr.start_node(label)
        node.init(nodeptr)
        return node

    cdef add_attr(self, const CHyperedge *edgeptr, name, val):
        
        self.table[name].append((convert_edge(edgeptr), val))

    def add_edge(self, head_node, tail_nodes, 
                 label = "", 
                 **keywords):
        cdef vector[const CHypernode *] tail_node_ptrs
        for tail_node in tail_nodes:
            tail_node_ptrs.push_back((<Node> tail_node).nodeptr)
        cdef const CHyperedge *edgeptr = self.thisptr.add_edge(tail_node_ptrs, label)
        for key, val in keywords.iteritems():
            self.add_attr(edgeptr, key, val)

cdef class Node:
    cdef const CHypernode *nodeptr
    cdef CHypergraph *graphptr
    cdef int edge_count

    # def __cinit__(self):
    #     self.edge_count = 0

    # cdef init_mutable(self, CHypergraph *graphptr, 
    #                   const CHypernode *nodeptr):
    #     self.graphptr = graphptr
    #     self.nodeptr = nodeptr

    cdef init(self, const CHypernode *nodeptr):
        self.nodeptr = nodeptr

    def id(self):
        assert self.nodeptr.id() != -1, "Bad node id."
        return self.nodeptr.id()

    def edges(self):
        return convert_edges(self.nodeptr.edges())

    def is_terminal(self):
        return self.nodeptr.edges().size() == 0

    def __cinit__(self):
        pass


    # def add_edge(self, tail_nodes, label = ""):
    #     cdef vector[const CHypernode *] tail_node_ptrs
    #     for tail_node in tail_nodes:
    #         tail_node_ptrs.push_back((<Node> tail_node).nodeptr)
    #     edgeptr = self.graphptr.add_edge(tail_node_ptrs, label)
    #     self.edge_count += 1
    #     edge = Edge()
    #     edge.init(edgeptr)
    #     return edge
    # def __enter__(self):
    #     return self
    # def __exit__(self, exception, b, c):
    #     if exception:
    #        return False
    #     if self.edge_count == 0:
    #         assert False
    #     self.graphptr.end_node()

cdef class Edge:
    cdef const CHyperedge *edgeptr
    
    def __cinit__(self):
        pass

    cdef init(self, const CHyperedge *ptr):
        self.edgeptr = ptr

    def tail(self):
        return convert_nodes(self.edgeptr.tail_nodes())

    def head(self):
        return convert_node(self.edgeptr.head_node())

    def label(self):
        return self.edgeptr.label()

    def removed(self):
        return (self.edgeptr.id() == -1)

    def id(self):
        assert self.edgeptr.id() != -1, "Bad edge id."
        return self.edgeptr.id()

# cdef class EdgeBuilder(Edge):
#     cdef GraphBuilder builder
    
#     cdef init_build(self, const CHyperedge *ptr, GraphBuilder builder):
#         self.builder = builder
#         self.edgeptr = ptr

#     def __setitem__(self, name, val):
#         self.builder.add_attr(self, name, val)

cdef convert_edges(vector[const CHyperedge *] edges):
    return [convert_edge(edge) for edge in edges]

cdef convert_edge(const CHyperedge * edge):
    py_edge = Edge()
    py_edge.init(edge)
    return py_edge

cdef convert_nodes(vector[const CHypernode *] nodes):
    return [convert_node(node) for node in nodes]

cdef convert_node(const CHypernode * node):
    py_node = Node()
    py_node.init(node)
    return py_node

cdef class Path:
    cdef const CHyperpath *thisptr
    cdef init(self, const CHyperpath *path):
        self.thisptr = path

    def edges(self):
        return convert_edges(self.thisptr.edges())

    def __contains__(self, Edge hyperedge):
        return self.thisptr.has_edge(hyperedge.edgeptr)
        
cdef class Weights:
    cdef Hypergraph hypergraph
    cdef const CHypergraphWeights *thisptr
    def __cinit__(self, Hypergraph hypergraph):
        self.hypergraph = hypergraph
        hypergraph.register(self, "weight")

    cdef init(self, vector[double] weights, double bias):
        self.thisptr = new CHypergraphWeights(self.hypergraph.thisptr, 
                                              weights, bias)
    
    # def builder(self):
    #     return WeightBuilder(self)

    def build(self, attrs):
        cdef vector[double] weight_vals
        weight_vals.resize(self.hypergraph.edges_size(), 0)
        for edge, w in attrs:
            if not edge.removed():
                weight_vals[edge.id()] = w
        self.init(weight_vals, 0.0)

    def dot(self, Path path):
        cdef double result = self.thisptr.dot(deref(path.thisptr))
        return result

# cdef class WeightBuilder:
#     cdef vals
#     cdef Weights weights
#     def __cinit__(self, Weights weights):
#         self.vals = []
#         self.weights = weights

#     def __setitem__(self, Edge edge, double val):
#         self.vals.append((edge, val))

    # def __enter__(self): return self

    # def __exit__(self, exception, b, c):
    #     if exception: return False
        
    #     cdef vector[double] weight_vals
    #     weight_vals.resize(self.weights.hypergraph.edges_size(), 0)
    #     for edge, w in self.vals:
    #         weight_vals[edge.id()] = w
    #     self.weights.init(weight_vals, 0.0)


cdef class Constraint:
    cdef CConstraint *thisptr
    cdef init(self, CConstraint *ptr):
        self.thisptr = ptr

    #def set_constant(self, int constant):
     #   self.thisptr.set_constant(constant)

    def __setitem__(self, Edge edge, int val):
        self.thisptr.add_edge_term(edge.edgeptr, val)

    def add_edge_term(self, Edge edge, int coefficient):
        self.thisptr.add_edge_term(edge.edgeptr, coefficient)

cdef class Constraints:
    cdef CHypergraphConstraints *thisptr
    def __cinit__(self, Hypergraph hypergraph):
        self.thisptr = new CHypergraphConstraints(hypergraph.thisptr)
        hypergraph.register(self, "constraints")

    def build(self, attrs):
        for edge, ls in attrs:
            if not edge.removed():
                for constraint, coeff in ls:
                    constraint.add_edge_term(edge, coeff)
        # cdef vector[double] weight_vals
        # weight_vals.resize(self.weights.hypergraph.edges_size(), 0)
        # for edge, w in attrs:
        #     weight_vals[edge.id()] = w
        # self.init(weight_vals, 0.0)

        

    def add(self, string label, int constant):
        cdef CConstraint *cons
        cons = self.thisptr.add_constraint(label)
        cdef Constraint hcons = Constraint()
        hcons.init(cons)
        hcons.thisptr.set_constant(constant)
        return hcons

    def check(self, Path path):
        cdef vector[const CConstraint *] failed
        cdef vector[int] count
        self.thisptr.check_constraints(deref(path.thisptr),
                                       &failed,
                                       &count)
        ret = []
        for cons in failed:
            ret.append(cons.label)
        return ret


# cdef class CHyperedge:
#     cdef CHyperedgeImpl *thisptr

#     def __cinit__(self, label,
#                   features,
#                   int id,
#                   tails,
#                   CHypernode head):
#         self.init(label, features, id, tails, head)

#     cdef init(self, string label, string features,
#               int id, tails, CHypernode head):
#         ptrs = [tail.thisptr for tail in tails]
#         cdef vector[CHypernodeImpl *] tail_ptrs
#         for tail in tails:
#             tail_ptrs.push_back((<CHypernode>tail).thisptr)
#         self.thisptr = new CHyperedgeImpl(label, features, id,
#                                          tail_ptrs, head.thisptr)

# cdef class CHypernode:
#     cdef CHypernodeImpl *thisptr
#     def __cinit__(self, label, id):
#         self.thisptr = new CHypernodeImpl(label, id)

#     def __dealloc__(self):
#         del self.thisptr
