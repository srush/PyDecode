#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool


class HypergraphAccessException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class HypergraphConstructionException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


cdef extern from "Hypergraph/Algorithms.h":
    CHyperpath *viterbi_path(
        const CHypergraph *graph,
        const CHypergraphWeights theta,
        vector[double] *chart) except +

    void outside(
        const CHypergraph *graph,
        const CHypergraphWeights weights,
        const vector[double] inside_chart,
        vector[double] *chart) except +

    const CHypergraphProjection *prune(
        const CHypergraph *original,
        const CHypergraphWeights weights,
        double ratio) except +

    const CHyperpath *best_constrained_path(
        const CHypergraph *graph,
        const CHypergraphWeights theta,
        const CHypergraphConstraints constraints,
        vector[CConstrainedResult] *constraints,
        ) except +

    cdef cppclass CConstrainedResult "ConstrainedResult":
        const CHyperpath *path
        double dual
        double primal
        vector[const CConstraint *] constraints

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
# cdef extern from "Hypergraph/Subgradient.h":
#     cdef cppclass CSubgradient "Subgradient":
#          vector[double] duals()

cdef class Chart:
    r"""
    A dynamic programming chart :math:`\pi \in R^{|{\cal V}|}`.

    Act as a dictionary ::

       >> print chart[node]

    """

    cdef vector[double] chart

    def __getitem__(self, Node node):
        r"""
        __getitem__(self, node)

        Get the chart score for a node.

        Parameters
        ----------
        node : :py:class:`Node`
          A node :math:`v \in {\cal V}`.

        Returns
        -------
         : float
          The score of the node, :math:`\pi[v]`.
        """
        return self.chart[node.id]



def best_path(Hypergraph graph, Weights weights):
    r"""
    Find the highest-scoring path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    weights : :py:class:`Weights`
      The weights :math:`\theta` of the hypergraph.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """

    cdef Chart chart = Chart()
    cdef const CHyperpath *hpath = \
        viterbi_path(graph.thisptr,
                     deref(weights.thisptr),
                     &chart.chart)
    cdef Path path = Path()
    path.init(hpath)
    return path

def inside_values(Hypergraph graph, Weights weights):
    r"""
    Find the inside path chart values.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    weights : :py:class:`Weights`
      The weights :math:`\theta` of the hypergraph.

    Returns
    -------

    : :py:class:`Chart`
       The inside chart.
    """
    cdef Chart chart = Chart()
    cdef const CHyperpath *hpath = \
        viterbi_path(graph.thisptr,
                     deref(weights.thisptr),
                     &chart.chart)
    cdef Path path = Path()
    path.init(hpath)
    return chart

def outside_values(Hypergraph graph,
                   Weights weights,
                   Chart inside_chart):
    """
    Find the outside scores for the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    weights : :py:class:`Weights`
       The weights of the hypergraph.

    inside_chart : :py:class:`Chart`
       The inside chart.

    Returns
    ---------

    : :py:class:`Chart`
       The outside chart.

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


    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    weights : :py:class:`Weights`
       The weights of the hypergraph.

    constraints : :py:class:`Constraints`
        The hyperedge constraints.

    Returns
    ---------

    The best path and the dual values.
    """
    cdef vector[CConstrainedResult] results
    cdef const CHyperpath *cpath = \
        best_constrained_path(graph.thisptr,
                              deref(weights.thisptr),
                              deref(constraints.thisptr),
                              &results)

    cdef Path path = Path()
    path.init(cpath)
    return path, convert_results(results)

def compute_max_marginals(Hypergraph graph,
                          Weights weights):
    """
    Compute the max-marginal value for each vertex and edge in the
    hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    weights : :py:class:`Weights`
       The weights of the hypergraph.

    Returns
    --------

    :py:class:`MaxMarginals`
       The max-marginals.
    """
    cdef const CMaxMarginals *marginals = \
        compute(graph.thisptr, weights.thisptr)
    cdef MaxMarginals max_marginals = MaxMarginals()
    max_marginals.init(marginals)
    return max_marginals

def prune_hypergraph(Hypergraph graph,
                     Weights weights,
                     double ratio):
    """
    Prune hyepredges with low max-marginal score from the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    weights : :py:class:`Weights`
       The weights of the hypergraph.

    Returns
    --------

    The new hypergraphs and weights.
    """
    cdef const CHypergraphProjection *projection = \
        prune(graph.thisptr, deref(weights.thisptr), ratio)
    cdef Hypergraph new_graph = Hypergraph()

    # Map nodes.
    node_labels = [None] * projection.new_graph.nodes().size()
    cdef vector[const CHypernode*] old_nodes = graph.thisptr.nodes()
    cdef const CHypernode *node
    for i in range(old_nodes.size()):
        node = projection.project(old_nodes[i])
        if node != NULL and node.id() >= 0:
            node_labels[node.id()] = graph.node_labels[i]

    # Map edges.
    edge_labels = [None] * projection.new_graph.edges().size()
    cdef vector[const CHyperedge *] old_edges = graph.thisptr.edges()
    cdef const CHyperedge *edge
    for i in range(old_edges.size()):
        edge = projection.project(old_edges[i])
        if edge != NULL and edge.id() >= 0:
            edge_labels[edge.id()] = graph.edge_labels[i]

    new_graph.init(projection.new_graph, node_labels, edge_labels)
    cdef Weights new_weights = Weights(new_graph)
    new_weights.init(
        weights.thisptr.project_weights(deref(projection)))
    return new_graph, new_weights

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
    r"""
    A sub-result from the constrained solver.


    Attributes
    -----------

    path : :py:class:`Path`
      The hyperpath :math:`y \in {\cal X}`
      associated with this round.

    dual : float
       The dual value for this round.

    primal : float
       The primal value for this round.

    constraints : list of :py:class:`Constraint`
       The constraints violated for this round.
    """

    cdef CConstrainedResult thisptr
    cdef init(self, CConstrainedResult ptr):
        self.thisptr = ptr

    property path:
        def __get__(self):
            path = Path()
            path.init(self.thisptr.path)
            return path

    property dual:
        def __get__(self):
            return self.thisptr.dual

    property primal:
        def __get__(self):
            return self.thisptr.primal

    property constraints:
        def __get__(self):
            return convert_constraints(self.thisptr.constraints)


cdef class Hypergraph:
    r"""

    Hypergraph consisting of a set of nodes :math:`{\cal V}`, hyperedges :math:`{\cal E}`, and a root node.

    Attributes
    -----------

    root : :py:class:`Node`
      A specialized node in :math:`{\cal V}`.

    node : list of :py:class:`Node`s
      The node set :math:`{\cal V}`. In topological-order.

    edges : list of :py:class:`Edge`s
      The edge set :math:`{\cal E}`. In topological-order.

    """

    cdef CHypergraph *thisptr
    cdef edge_labels
    cdef node_labels
    def __cinit__(self):
        """
        Create a new hypergraph.
        """
        self.thisptr = new CHypergraph()
        self.edge_labels = []
        self.node_labels = []

    cdef init(self, const CHypergraph *ptr, node_labels, edge_labels):
        self.thisptr = <CHypergraph *> ptr
        self.edge_labels = edge_labels
        self.node_labels = node_labels

    def builder(self):
        r"""
        builder()

        The builder for the hypergraph ::

           >> hypergraph = Hypergraph()
           >> with hypergraph.builder() as b:
           >>    b.add_node()

        Returns
        ---------------------
        :py:class:`GraphBuilder`
        """
        gb = GraphBuilder()
        gb.init(self, self.thisptr)
        return gb


    property nodes:
        def __get__(self):
            return convert_nodes(self.thisptr.nodes())

    property root:
        def __get__(self):
            return convert_node(self.thisptr.root())

    property edges:
        def __get__(self):
            return convert_edges(self.thisptr.edges())

    property edges_size:
        def __get__(self):
            return self.thisptr.edges().size()

    def label(self, Edge edge):
        """
        label(edge)

        The label associated with a hyperedge `edge`.
        """
        return self.edge_labels[edge.id]

    def node_label(self, Node node):
        """
        node_label(node)

        The label associated with a node `node`.
        """
        return self.node_labels[node.id]

cdef class GraphBuilder:
    r"""
    Build a hypergraph. Created using ::

           >> hypergraph = Hypergraph()
           >> with hypergraph.builder() as b:
           >>    b.add_node()

    Methods
    -------

    add_node(edges=[], label="")
        Add a node (and its hyperedges) to the hypergraph.

    """
    cdef CHypergraph *thisptr
    cdef Hypergraph hyper
    cdef edge_labels
    cdef node_labels
    cdef started

    def __init__(self):
        ""
        pass


    cdef init(self, Hypergraph hyper, CHypergraph *ptr):
        self.thisptr = ptr
        self.hyper = hyper
        self.edge_labels = []
        self.node_labels = []
        self.started = False

    def __enter__(self):
        # """
        # Start building the hypergraph.

        # Use as with hypergraph.builder() as b:
        # """
        self.started = True
        return self

    def __exit__(self, exception, b, c):
        # """End building the hypergraph

        # Automatically called when exiting with block.
        # """
        if exception:
           return False
        self.started = False
        self.thisptr.finish()
        self.hyper.edge_labels = [None] * self.thisptr.edges().size()
        self.hyper.node_labels = [None] * self.thisptr.nodes().size()

        for node, t in self.node_labels:
            if not node._removed():
                self.hyper.node_labels[node.id] = t

        for edge, t in self.edge_labels:
            if not edge._removed():
                self.hyper.edge_labels[edge.id] = t

    def add_node(self, edges=[], label=""):
        """
        add_node(edges=[], label="")

        Add a node to the hypergraph.

        Parameters
        ------------

        edges :
           An iterator where each of the items is of the form
           ([v_2, v_3..], label)  where v_2 ... are :py:class:`Node`s and
           label is an edge label of any type.

        label : any
           Optional label for the node.


        Returns
        --------------
        :py:class:`Node`
        """

        if not self.started:
            raise HypergraphConstructionException(
                "Must constuct graph in 'with' block.")

        cdef Node node = Node()
        cdef const CHypernode *nodeptr
        cdef vector[const CHypernode *] tail_node_ptrs
        cdef const CHyperedge *edgeptr
        if edges == []:
            nodeptr = self.thisptr.add_terminal_node(str(label))
        else:
            nodeptr = self.thisptr.start_node(str(label))
            for edge_cons in edges:
                try:
                    tail_nodes, t = edge_cons
                except:
                    raise HypergraphConstructionException(
                        "Edges must be pairs of the form (tail_nodes, label)." + \
                        "Received %s"%(edge_cons)
                        )
                if len(tail_nodes) == 0:
                    raise HypergraphConstructionException(
                        "An edge must have at least one tail node.")

                tail_node_ptrs.clear()
                for tail_node in tail_nodes:
                    tail_node_ptrs.push_back((<Node> tail_node).nodeptr)
                edgeptr = self.thisptr.add_edge(tail_node_ptrs, "")
                self.edge_labels.append((convert_edge(edgeptr), t))
            self.thisptr.end_node()
        self.node_labels.append((node, label))
        node.init(nodeptr)
        return node

cdef class Node:
    r"""
    Node :math:`v \in {\cal V}` associated with a :py:class:`Hypergraph`.

    Attributes
    -------------

    edge : list of edges
       The edges with :math:`v` as head node.

       :math:`\{e \in {\cal E} : h(e) = v \}`


    is_terminal : bool
       Is the node :math:`v` in terminal node.

    """
    cdef const CHypernode *nodeptr
    cdef CHypergraph *graphptr

    cdef init(self, const CHypernode *nodeptr):
        self.nodeptr = nodeptr

    def __hash__(self):
        return self.id

    property id:
        def __get__(self):
            assert self.nodeptr.id() != -1, "Bad node id."
            return self.nodeptr.id()

    property edges:
        def __get__(self):
            return convert_edges(self.nodeptr.edges())

    property label:
        def __get__(self):
            return self.nodeptr.label()

    property is_terminal:
        def __get__(self):
            return (self.nodeptr.edges().size() == 0)

    def __str__(self):
        return self.nodeptr.label()

    def __cinit__(self):
        ""
        pass

    def _removed(self):
        return (self.nodeptr.id() == -1)

cdef class Edge:
    r"""
    Hyperedge :math:`e \in {\cal E}` associated with a :py:class:`Hypergraph`.

    Attributes
    -----------

    head : :py:class:`Node`
        The head node :math:`v = h(e)`.


    tail : list of nodes
        The tail nodes :math:`v_2 \ldots v_{n} \in t(e)`.

    """

    cdef const CHyperedge *edgeptr

    def __cinit__(self):
        ""
        pass

    def __hash__(self):
        return self.id

    cdef init(self, const CHyperedge *ptr):
        self.edgeptr = ptr

    def __str__(self):
        return self.edgeptr.label()

    property tail:
        def __get__(self):
            return convert_nodes(self.edgeptr.tail_nodes())

    property head:
        def __get__(self):
            return convert_node(self.edgeptr.head_node())

    property id:
        def __get__(self):
            assert self.edgeptr.id() != -1, "Bad edge id."
            return self.edgeptr.id()

    def _removed(self):
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
    r"""
    Valid hyperpath :math:`y \in {\cal X}` in the hypergraph.

    To check if an edge is in a path ::

       >> edge in path

    To iterate over a path (in topological order) ::

       >> [edge for edge in path]

    The edges :math:`e \in {\cal E}` with :math:`y(e) = 1`.

    """
    cdef const CHyperpath *thisptr
    def __cinit__(self, Hypergraph graph=None, edges=[]):
        """
        """

        cdef vector[const CHyperedge *] cedges
        if graph and edges:
            for edge in edges:
                cedges.push_back((<Edge>edge).edgeptr)
            self.thisptr = new CHyperpath(graph.thisptr, cedges)

    cdef init(self, const CHyperpath *path):
        self.thisptr = path

    property edges:
        def __get__(self):
            return convert_edges(self.thisptr.edges())

    def __contains__(self, Edge edge):
        """
        Is the edge in the hyperpath, i.e. :math:`y(e) = 1`?
        """
        return self.thisptr.has_edge(edge.edgeptr)

    def __iter__(self):
        return iter(convert_edges(self.thisptr.edges()))

cdef class Weights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::

       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphWeights *thisptr
    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph

    def build(self, fn):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
        cdef vector[double] weights
        weights.resize(self.hypergraph.thisptr.edges().size(), 0.0)
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = 0.0
            weights[i] = result
        self.thisptr =  \
          new CHypergraphWeights(self.hypergraph.thisptr,
                                 weights, 0.0)
        return self

    cdef init(self, const CHypergraphWeights *ptr):
        self.thisptr = ptr

    def __getitem__(self, Edge edge not None):
        return self.thisptr.score(edge.edgeptr)

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        cdef double result = self.thisptr.dot(deref(path.thisptr))
        return result

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
    cdef init(self, const CConstraint *ptr):
        self.thisptr = ptr

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

cdef class MaxMarginals:
    r"""
    The max-marginal scores of a weighted hypergraph.

    .. math::

        m(e) =  max_{y \in {\cal X}: y(e) = 1} \theta^{\top} y \\
        m(v) =  max_{y \in {\cal X}: y(v) = 1} \theta^{\top} y


    Usage is
        >> max_marginals = compute_max_marginals(graph, weights)
        >> m_e = max_marginals[edge]
        >> m_v = max_marginals[node]

    """

    cdef const CMaxMarginals *thisptr

    cdef init(self, const CMaxMarginals *ptr):
        self.thisptr = ptr

    def __getitem__(self, obj):
        """
        Get the max-marginal value of a node or an edge.

        :param obj: The node/edge to check..
        :type obj: A :py:class:`Node` or :py:class:`Edge`

        :returns: The max-marginal value.
        :rtype: float

        """
        if isinstance(obj, Edge):
            return self.thisptr.max_marginal((<Edge>obj).edgeptr)
        elif isinstance(obj, Node):
            return self.thisptr.max_marginal((<Node>obj).nodeptr)
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have max-marginal values." + \
                "Passed %s."%obj)


############# This is the templated semiring part. ##############

{% for S in semirings %}

cdef extern from "Hypergraph/Algorithms.h":
    CHyperpath * inside_{{S.type}} "viterbi_path[S.ctype]" (
        const CHypergraph *graph,
        const CHypergraphWeights theta,
        vector[{{S.ctype}}] *chart) except +

    cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
        {{S.ctype}} marginal(const CHyperedge *edge)
        {{S.ctype}} marginal(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<{{S.ctype}}>":
    C{{S.type}}Marginals *{{S.type}}_compute "Marginals<{{S.ctype}}>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraph{{S.type}}Weights *weights)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass {{S.ctype}}:
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "{{S.ctype}}":
    {{S.ctype}} {{S.type}}_one "{{S.ctype}}::one" ()
    {{S.ctype}} {{S.type}}_zero "{{S.ctype}}::zero" ()

cdef extern from "Hypergraph/Algorithms.h" namespace "{{S.ctype}}":
    cdef cppclass CHypergraph{{S.type}}Weights "{{S.type}}Weights":
        {{S.ctype}} dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraph{{S.type}}Weights *project_weights(
            const CHypergraphProjection )
        CHypergraph{{S.type}}Weights(
            const CHypergraph *hypergraph,
            const vector[{{S.ctype}}] weights,
            {{S.ctype}} bias) except +



cdef class {{S.type}}Weights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraph{{S.type}}Weights *thisptr

    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph

    def build(self, fn):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
        cdef vector[{{S.ctype}}] weights = \
             vector[{{S.ctype}}](self.hypergraph.thisptr.edges().size(),
             {{S.type}}_zero())
        cdef {{S.ptype}} result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = {{S.type}}_zero()
            weights[i] = result.wrap
        self.thisptr =  \
          new CHypergraph{{S.type}}Weights(self.hypergraph.thisptr,
                                                  weights, {{S.type}}_zero())
        return self

    cdef init(self, const CHypergraph{{S.type}}Weights *ptr):
        self.thisptr = ptr

    def __getitem__(self, Edge edge not None):
        return self.thisptr.score(edge.edgeptr)

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return {{S.ptype}}().init(self.thisptr.dot(deref(path.thisptr)))

cdef class {{S.ptype}}:
    cdef {{S.ctype}} wrap

    def __cinit__(self):
        self.wrap = {{S.type}}_zero()

    cdef init(self, {{S.ctype}} wrap):
        self.wrap = wrap
        return self

cdef class {{S.type}}Chart:
    cdef vector[{{S.ctype}}] chart

    def __getitem__(self, Node node):
        return {{S.ptype}}().init(self.chart[node.id])

cdef class {{S.type}}Marginals:
    cdef const C{{S.type}}Marginals *thisptr

    cdef init(self, const C{{S.type}}Marginals *ptr):
        self.thisptr = ptr

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return {{S.ptype}}().init(self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return {{S.ptype}}().init(self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have {{S.type}} marginal values." + \
                "Passed %s."%obj)

def compute_{{S.type}}_marginals(Hypergraph graph,
                                 {{S.type}}Weights weights):
    cdef const C{{S.type}}Marginals *marginals = \
        {{S.type}}_compute(graph.thisptr, weights.thisptr)
    return {{S.type}}Marginals().init(marginals)

{% endfor %}
