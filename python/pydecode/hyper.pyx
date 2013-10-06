#cython: embedsignature=True
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
        const CHypergraphConstraints constraints,
        vector[CConstrainedResult] *constraints,
        )

    cdef cppclass CConstrainedResult "ConstrainedResult":
        const CHyperpath *path
        double dual
        double primal
        vector[const CConstraint *] constraints

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
        double score(const CHyperedge *edge)

cdef extern from "Hypergraph/Constraints.h":
    cdef cppclass CConstraint "Constraint":
        Constraint(string label, int id)
        void set_constant(int _bias)
        int has_edge(const CHyperedge *edge)
        void add_edge_term(const CHyperedge *edge, int coefficient)
        string label

    cdef cppclass CHypergraphConstraints "HypergraphConstraints":
        CHypergraphConstraints(const CHypergraph *hypergraph)
        CConstraint *add_constraint(string label)
        const CHypergraph *hypergraph()
        int check_constraints(const CHyperpath path,
                              vector[const CConstraint *] *failed,
                              vector[int] *count)

# cdef extern from "Hypergraph/Subgradient.h":
#     cdef cppclass CSubgradient "Subgradient":
#          vector[double] duals()

cdef class Chart:
    cdef vector[double] chart

    def __getitem__(self, Node node):
        """
        Get the chart score for a node.

        :param node: The node to check.
        :returns: A score
        """
        return self.chart[node.id]



def best_path(Hypergraph graph,
              Weights weights):
    """
    Find the highest-score path in the hypergraph.

    :param graph: The hypergraph to search.
    :param weights: The weights of the hypergraph.
    :returns: The best path and inside chart.
    """

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
    """
    Find the outside score for the hypergraph.

    :param graph: The hypergraph to search.
    :param weights: The weights of the hypergraph.
    :param inside_chart: The inside chart.
    :returns: The outside chart.
    """
    cdef Chart out_chart = Chart()
    outside(graph.thisptr, deref(weights.thisptr),
            inside_chart.chart, &out_chart.chart)
    return out_chart

def best_constrained(Hypergraph graph,
                     Weights weights,
                     Constraints constraints):
    """
    Find the highest-scoring path satisfying constraints.

    :param graph: The hypergraph to search.
    :param weights: The weights of the hypergraph.
    :param constraints: The hyperedge constraints.
    :returns: The best path and the dual values.
    """
    cdef vector[CConstrainedResult] results
    cdef CHyperpath *cpath = best_constrained_path(graph.thisptr,
                          deref(weights.thisptr),
                          deref(constraints.thisptr),
                          &results)

    cdef Path path = Path()
    path.init(cpath)
    return path, convert_results(results)

cdef convert_results(vector[CConstrainedResult] c):
    cdef results = []
    for cresult in c:
        py_res = ConstrainedResult()
        py_res.init(cresult)
        results.append(py_res)
    return results

cdef convert_constraints(vector[const CConstraint *] c):
    cdef results = []
    for cresult in c:
        py_res = Constraint()
        py_res.init(cresult)
        results.append(py_res)
    return results

cdef class ConstrainedResult:
    cdef CConstrainedResult thisptr
    cdef init(self, CConstrainedResult ptr):
        self.thisptr = ptr

    def __getattr__(self, attr):
        if attr == "path":
            path = Path()
            path.init(self.thisptr.path)
            return path
        if attr == "dual":
            return self.thisptr.dual
        if attr == "primal":
            return self.thisptr.primal
        if attr == "constraints":
            return convert_constraints(self.thisptr.constraints)

cdef class Hypergraph:
    cdef CHypergraph *thisptr
    cdef types
    def __cinit__(self):
        self.thisptr = new CHypergraph()
        self.types = []

    def builder(self):
        gb = GraphBuilder()
        gb.init(self, self.thisptr)
        return gb

    def __getattr__(self, attr):
        if attr == "nodes":
            return convert_nodes(self.thisptr.nodes())
        if attr == "root":
            return convert_node(self.thisptr.root())
        if attr == "edges":
            return convert_edges(self.thisptr.edges())

    def edges_size(self):
        return self.thisptr.edges().size()

    def label(self, edge):
        return self.types[edge.id]


cdef class GraphBuilder:
    """
    Build a hypergraph. Created using

    with hypergraph.builder() as b:
    """
    cdef CHypergraph *thisptr
    cdef Hypergraph hyper
    cdef types

    cdef init(self, Hypergraph hyper, CHypergraph *ptr):
        self.thisptr = ptr
        self.hyper = hyper
        self.types = []

    def __enter__(self): return self

    def __exit__(self, exception, b, c):
        if exception:
           return False
        self.thisptr.finish()
        self.hyper.types = [None] * self.thisptr.edges().size()
        for edge, t in self.types:
            if not edge.removed():
                self.hyper.types[edge.id] = t

    def add_node(self, edges = [], label = ""):
        """
        Add a node to the hypergraph.

        :param edges: A list of edges of the form ([v_2, v_3..], label).
        :param label: Optional label for the node.

        """

        node = Node()
        cdef const CHypernode *nodeptr
        if not edges:
            nodeptr = self.thisptr.add_terminal_node(label)
        else:
            nodeptr = self.thisptr.start_node(label)
        cdef vector[const CHypernode *] tail_node_ptrs
        cdef const CHyperedge *edgeptr
        for tail_nodes, t in edges:
            tail_node_ptrs.clear()
            for tail_node in tail_nodes:
                tail_node_ptrs.push_back((<Node> tail_node).nodeptr)
            edgeptr = self.thisptr.add_edge(tail_node_ptrs, "")
            self.types.append((convert_edge(edgeptr), t))

        node.init(nodeptr)
        self.thisptr.end_node()
        return node

cdef class Node:
    cdef const CHypernode *nodeptr
    cdef CHypergraph *graphptr

    cdef init(self, const CHypernode *nodeptr):
        self.nodeptr = nodeptr

    def __hash__(self):
        return self.id

    def __getattr__(self, attr):
        if attr == "id":
            assert self.nodeptr.id() != -1, "Bad node id."
            return self.nodeptr.id()
        if attr == "edges":
            return convert_edges(self.nodeptr.edges())
        if attr == "label":
            return self.nodeptr.label()

    def is_terminal(self):
        return self.nodeptr.edges().size() == 0

    def __cinit__(self):
        pass

cdef class Edge:
    cdef const CHyperedge *edgeptr

    def __cinit__(self):
        pass

    def __hash__(self):
        return self.id

    cdef init(self, const CHyperedge *ptr):
        self.edgeptr = ptr

    def __getattr__(self, attr):
        if attr == "label":
            return self.edgeptr.label()
        if attr == "tail":
            return convert_nodes(self.edgeptr.tail_nodes())
        if attr == "head":
            return convert_node(self.edgeptr.head_node())
        if attr == "id":
            assert self.edgeptr.id() != -1, "Bad edge id."
            return self.edgeptr.id()

    def removed(self):
        return (self.edgeptr.id() == -1)

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
    """
    A valid path in the hypergraph.
    """
    cdef const CHyperpath *thisptr
    cdef init(self, const CHyperpath *path):
        self.thisptr = path

    def __getattr__(self, attr):
        """
        Returns the edges in the path.
        """
        if attr == "edges":
            return convert_edges(self.thisptr.edges())

    def __contains__(self, Edge edge):
        """
        Check whether an edge is in the path.

        :param edge: The edge to check.
        """
        return self.thisptr.has_edge(edge.edgeptr)

cdef class Weights:
    """
    Weights associated with a hypergraph.
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphWeights *thisptr
    def __cinit__(self, Hypergraph graph, fn):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        :param fn: A function from edge labels to weights.
        """
        self.hypergraph = graph

        cdef vector[double] weights
        weights.resize(self.hypergraph.thisptr.edges().size())
        for i, ty in enumerate(self.hypergraph.types):
            weights[i] = fn(ty)
        self.thisptr =  \
          new CHypergraphWeights(self.hypergraph.thisptr,
                                 weights, 0.0)

    def __getitem__(self, Edge edge):
        return self.thisptr.score(edge.edgeptr)

    def dot(self, Path path):
        """
        Score a path with a weight vector.

        :param path: The hyperpath  to score.
        :return: The score.
        """

        cdef double result = self.thisptr.dot(deref(path.thisptr))
        return result

cdef class Constraint:
    cdef const CConstraint *thisptr
    cdef init(self, const CConstraint *ptr):
        self.thisptr = ptr

    def __str__(self): return self.thisptr.label

    def __getattr__(self, attr):
        if attr == "label": return self.thisptr.label

    def __contains__(self, Edge edge):
        self.thisptr.has_edge(edge.edgeptr)

cdef class Constraints:
    """
    A class for storing the matrix of hypergraph constraints A y = b.
    """

    cdef CHypergraphConstraints *thisptr
    cdef Hypergraph hypergraph
    def __cinit__(self, Hypergraph hypergraph):
        self.thisptr = new CHypergraphConstraints(hypergraph.thisptr)
        self.hypergraph = hypergraph

    def add(self, string label, fn, int constant):
        """
        Add a new hypergraph constraint.

        :param label: The name of the constraint.
        :param fn: A function mapping the label of an edge to its coefficient.
        :param constant: The value b_i for this constraint.
        :returns: The constraint.
        """
        cdef CConstraint *cons
        cons = self.thisptr.add_constraint(label)
        cdef Constraint hcons = Constraint()
        hcons.init(cons)
        cons.set_constant(constant)
        cdef vector[const CHyperedge *] edges = self.hypergraph.thisptr.edges()
        cdef int coefficient
        for i, ty in enumerate(self.hypergraph.types):
            coefficient = fn(ty)
            if coefficient: cons.add_edge_term(edges[i], coefficient)
        return hcons

    def check(self, Path path):
        """
        Check which constraints a path violates.

        :param path: The hyperpath to check
        :returns: The labels of violated constraints.
        """

        cdef vector[const CConstraint *] failed
        cdef vector[int] count
        self.thisptr.check_constraints(deref(path.thisptr),
                                       &failed,
                                       &count)
        ret = []
        for cons in failed:
            ret.append(cons.label)
        return ret
