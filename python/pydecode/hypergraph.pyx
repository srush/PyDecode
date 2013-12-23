#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from wrap cimport *


cdef _hypergraph_registry_counts = {}


cdef class Hypergraph:
    r"""1

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
    def __cinit__(Hypergraph self):
        """
        Create a new hypergraph.
        """
        self.thisptr = NULL
        self.edge_labels = []
        self.node_labels = []
        self._cached_edges = None

    def __dealloc__(self):
        if self.thisptr is not NULL:
            _hypergraph_registry_counts[self.thisptr.id()] -= 1
            #print "LOWERING", self.thisptr.id(), _hypergraph_registry_counts[self.thisptr.id()]
            if _hypergraph_registry_counts[self.thisptr.id()] == 0:
                #print "DEALLOCATING", self.thisptr.id()
                del self.thisptr
                self.thisptr = NULL



    cdef Hypergraph init(self, const CHypergraph *ptr,
                         node_labels, edge_labels):
        #assert ptr.id() in _hypergraph_registry[ptr.id()]
        assert self.thisptr is NULL
        if _hypergraph_registry_counts.get(ptr.id(), 0) > 0:
            _hypergraph_registry_counts[ptr.id()] += 1
        else:
            _hypergraph_registry_counts[ptr.id()] = 1
        self.thisptr = <CHypergraph *> ptr
        self.edge_labels = edge_labels
        self.node_labels = node_labels
        return self

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
        self.thisptr = new CHypergraph()
        #_hypergraph_registry[self.thisptr.id()] = self
        _hypergraph_registry_counts[self.thisptr.id()] = 1
        return GraphBuilder().init(self, self.thisptr)

    property nodes:
        def __get__(self):
            return convert_nodes(self.thisptr.nodes(), self)

    property root:
        def __get__(self):
            return Node().init(self.thisptr.root(), self)

    property edges:
        def __get__(self):
            if not self._cached_edges:
                self._cached_edges = convert_edges(self.thisptr.edges(), self)
            return self._cached_edges

    property edges_size:
        def __get__(self):
            return self.thisptr.edges().size()

    def label(self, Edge edge):
        """
        label(edge)

        Deprecated.

        The label associated with a hyperedge `edge`.
        """
        return self.edge_labels[edge.id]

    def node_label(self, Node node):
        """
        node_label(node)

        Deprecated.

        The label associated with a node `node`.
        """
        return self.node_labels[node.id]

    def __str__(self):
        s = "Hypergraph: Edges: %s Nodes: %s"%(len(self.edges),
                                               len(self.nodes)) + "\n"
        s += "Root %s"%(self.root.id) + "\n"
        for edge in self.edges:
            s += " %s %s \n"%(edge.id, self.label(edge))
            s += "\t%d -> "%(edge.head.id)
            for node in edge.tail:
                s += " %d "%(node.id)
            s += "\n"
        return s

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

    def __init__(self):
        ""
        pass


    cdef GraphBuilder init(self, Hypergraph hyper, CHypergraph *ptr):
        self.thisptr = ptr
        self.graph = hyper
        self.edge_labels = []
        self.node_labels = []
        self.started = False
        return self

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
        self.graph.edge_labels = [None] * self.thisptr.edges().size()
        self.graph.node_labels = [None] * self.thisptr.nodes().size()

        for node, t in self.node_labels:
            if not node._removed():
                self.graph.node_labels[node.id] = t

        for edge, t in self.edge_labels:
            if not edge._removed():
                self.graph.edge_labels[edge.id] = t

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


        cdef const CHypernode *nodeptr
        cdef vector[const CHypernode *] tail_node_ptrs
        cdef const CHyperedge *edgeptr
        if edges == []:
            nodeptr = self.thisptr.add_terminal_node("")
        else:
            nodeptr = self.thisptr.start_node("")
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
                self.edge_labels.append((Edge().init(edgeptr, self.graph), t))
            self.thisptr.end_node()
        cdef Node node = Node().init(nodeptr, self.graph)
        self.node_labels.append((node, label))
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

    label : any
        A piece of data associated with the edge.

    """

    cdef Node init(self, const CHypernode *nodeptr,
                   Hypergraph graph):
        self.nodeptr = nodeptr
        self.graph = graph
        return self

    def __hash__(self):
        return self.id

    def __repr__(self):
        return "NODE:%d"%(self.nodeptr.id())

    property id:
        def __get__(self):
            assert self.nodeptr.id() != -1, "Bad node id."
            return self.nodeptr.id()

    property edges:
        def __get__(self):
            return convert_edges(self.nodeptr.edges(), self.graph)

    property is_terminal:
        def __get__(self):
            return (self.nodeptr.edges().size() == 0)

    property label:
        def __get__(self):
            return self.graph.node_label(self)

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

    label : any
        A piece of data associated with the edge.

    """

    def __cinit__(self):
        ""
        pass

    def __hash__(self):
        return self.id

    cdef Edge init(self, const CHyperedge *ptr, Hypergraph graph):
        self.edgeptr = ptr
        self.graph = graph
        return self

    def __str__(self):
        return self.edgeptr.label()

    def __repr__(self):
        return "EDGE:%d"%(self.edgeptr.id())

    property tail:
        def __get__(self):
            return convert_nodes(self.edgeptr.tail_nodes(), self.graph)

    property head:
        def __get__(self):
            return Node().init(self.edgeptr.head_node(), self.graph)

    property label:
        def __get__(self):
            return self.graph.label(self)

    property id:
        def __get__(self):
            assert self.edgeptr.id() != -1, "Bad edge id."
            return self.edgeptr.id()

    def _removed(self):
        return (self.edgeptr.id() == -1)

cdef convert_edges(vector[const CHyperedge *] edges,
                   Hypergraph graph):
    return [Edge().init(edge, graph) for edge in edges]

cdef convert_nodes(vector[const CHypernode *] nodes,
                   Hypergraph graph):
    return [Node().init(node, graph) for node in nodes]


cdef class Path:
    r"""
    Valid hyperpath :math:`y \in {\cal X}` in the hypergraph.

    To check if an edge is in a path ::

       >> edge in path

    To iterate over a path (in topological order) ::

       >> [edge for edge in path]

    The edges :math:`e \in {\cal E}` with :math:`y(e) = 1`.

    """

    def __dealloc__(self):
        del self.thisptr

    def __cinit__(self, Hypergraph graph=None, edges=[]):
        """
        """

        cdef vector[const CHyperedge *] cedges
        self.graph = graph
        edges.sort(key=lambda e:e.id)
        if graph and edges:
            for edge in edges:
                cedges.push_back((<Edge>edge).edgeptr)
            self.thisptr = new CHyperpath(graph.thisptr, cedges)

    cdef Path init(self, const CHyperpath *path, Hypergraph graph):
        self.thisptr = path
        self.graph = graph
        return self

    def __str__(self):
        return ":".join([str(edge) for edge in self.edges])

    def __contains__(self, Edge edge):
        """
        Is the edge in the hyperpath, i.e. :math:`y(e) = 1`?
        """
        return self.thisptr.has_edge(edge.edgeptr)

    def __iter__(self):
        return iter(convert_edges(self.thisptr.edges(), self.graph))

    cdef public equal(Path self, Path other):
        return self.thisptr.equal(deref(other.thisptr))

    def __richcmp__(Path self, Path other, op):
        if op == 2:
            return self.equal(other)
        if op == 3:
            return not self.equal(other)
        raise Exception("No inequality on paths.")

    property edges:
        def __get__(self):
            return convert_edges(self.thisptr.edges(), self.graph)

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
