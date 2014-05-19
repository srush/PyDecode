#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef _hypergraph_registry_counts = {}

cdef class Labeling:
    def __init__(self, Hypergraph graph,
                 node_labels=None, edge_labels=None):
        self.edge_labels = edge_labels
        self.node_labels = node_labels

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            if self.edge_labels is None:
                raise HypergraphAccessException(
                    "There is no edge labeling.")
            return self.edge_labels[obj.id]

        if isinstance(obj, Vertex):
            if self.node_labels is None:
                raise HypergraphAccessException(
                    "There is no node labeling.")
            return self.node_labels[obj.id]

cdef class _LazyEdges:
    def __init__(self, graph):
        self._graph = graph

    cdef init(self, vector[int ] edges):
        self._edges = edges
        return self

    def __getitem__(self, item):
        return Edge().init(self._edges[item], self._graph)

    def __iter__(self):
        return (Edge().init(edge, self._graph) for edge in self._edges)

    def __len__(self):
        return self._edges.size()

cdef class _LazyVertices:
    def __init__(self, graph):
        self._graph = graph

    cdef init(self, vector[const CHypernode *] nodes):
        self._nodes = nodes
        return self

    def __getitem__(self, item):
        return Vertex().init(self._nodes[item], self._graph)

    def __iter__(self):
        return (Vertex().init(node, self._graph) for node in self._nodes)

    def __len__(self):
        return self._nodes.size()

cdef class Hypergraph:
    r"""
    The search space of a dynamic program.

    Hypergraph consisting of a set of nodes :math:`{\cal V}`,
    hyperedges :math:`{\cal E}`, and a root vertex.

    Attributes
    -----------

    edges : list of :py:class:`Edge`
      List of edge set :math:`{\cal E}` in topological order.

    root : :py:class:`Vertex`
      Root vertex in :math:`{\cal V}`.

    vertices : list of :py:class:`Vertex`
      List of vertex set :math:`{\cal V}` in topological order.

    labeling : :py:class:`Labeling`
      The labels associated with vertices and edges.

    """
    def __cinit__(Hypergraph self):
        """
        Create a new hypergraph.
        """
        self.thisptr = NULL
        self.labeling = None
        self._cached_edges = None

    def __dealloc__(self):
        if self.thisptr is not NULL:
            _hypergraph_registry_counts[self.thisptr.id()] -= 1
            if _hypergraph_registry_counts[self.thisptr.id()] == 0:
                del self.thisptr
                self.thisptr = NULL

    cdef Hypergraph init(self, const CHypergraph *ptr,
                         Labeling labeling):
        #assert ptr.id() in _hypergraph_registry[ptr.id()]
        assert self.thisptr is NULL
        if _hypergraph_registry_counts.get(ptr.id(), 0) > 0:
            _hypergraph_registry_counts[ptr.id()] += 1
        else:
            _hypergraph_registry_counts[ptr.id()] = 1
        self.thisptr = <CHypergraph *> ptr
        self.labeling = labeling
        return self

    def builder(self):
        self.thisptr = new CHypergraph()
        #_hypergraph_registry[self.thisptr.id()] = self
        _hypergraph_registry_counts[self.thisptr.id()] = 1
        return GraphBuilder().init(self, self.thisptr)

    property vertices:
        def __get__(self):
            return _LazyVertices(self).init(self.thisptr.nodes())

    property nodes:
        def __get__(self):
            return _LazyVertices(self).init(self.thisptr.nodes())

    property root:
        def __get__(self):
            return Vertex().init(self.thisptr.root(), self)

    property edges:
        def __get__(self):
            return _LazyEdges(self).init(self.thisptr.edges())

    property labeling:
        def __get__(self):
            return self.labeling

        def __set__(self, labeling):
            self.labeling = labeling

    def __str__(self):
        s = "Hypergraph: Edges: %s Vertices: %s" % (len(self.edges),
                                                 len(self.nodes)) + "\n"
        s += "Root %s" % (self.root.id) + "\n"
        for edge in self.edges:
            s += " %s %s \n" % (edge.id, edge.label)
            s += "\t%d -> " % (edge.head.id)
            for node in edge.tail:
                s += " %d " % (node.id)
            s += "\n"
        return s

cdef class GraphBuilder:
    r"""
    Direct constructor for hypergraphs.

    Usage ::

           >> hypergraph = Hypergraph()
           >> with hypergraph.builder() as b:
           >>    b.add_node()
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
        final_edge_labels = [None] * self.thisptr.edges().size()
        final_node_labels = [None] * self.thisptr.nodes().size()

        for node, t in self.node_labels:
            if not node._removed():
                final_node_labels[node.id] = t

        for edge, t in self.edge_labels:
            if not edge._removed():
                final_edge_labels[edge.id] = t
        self.graph.labeling = Labeling(self.graph, final_node_labels,
                                       final_edge_labels)

    def add_node(self, edges=[], label=None):
        """
        Add a node to the hypergraph.

        Parameters
        ------------

        edges :
           An iterator where each of the items is of the form
           ([v_2, v_3..], label)  where v_2 ... are :py:class:`Vertex` s and
           label is an edge label of any type.

        label : any
           Optional label for the node.


        Returns
        --------------
        :py:class:`Vertex`
        """

        if not self.started:
            raise HypergraphConstructionException(
                "Must constuct graph in 'with' block.")

        cdef const CHypernode *nodeptr
        cdef vector[const CHypernode *] tail_node_ptrs
        cdef int edgeptr
        if edges == []:
            nodeptr = self.thisptr.add_terminal_node()
        else:
            nodeptr = self.thisptr.start_node()
            for edge_cons in edges:
                try:
                    tail_nodes, t = edge_cons
                except:
                    raise HypergraphConstructionException(
                        "Edges must be pairs of the form (tail_nodes, label)."
                        + "Received %s" % (edge_cons))
                if len(tail_nodes) == 0:
                    raise HypergraphConstructionException(
                        "An edge must have at least one tail node.")

                tail_node_ptrs.clear()
                for tail_node in tail_nodes:
                    tail_node_ptrs.push_back((<Vertex> tail_node).nodeptr)
                edgeptr = self.thisptr.add_edge(tail_node_ptrs)
                self.edge_labels.append((Edge().init(edgeptr, self.graph, True), t))
            self.thisptr.end_node()
        cdef Vertex node = Vertex().init(nodeptr, self.graph)
        self.node_labels.append((node, label))
        return node

cdef class Vertex:
    r"""
    Hypergraph vertex.

    A hypergraph constains a set of vertices :math:`v \in {\cal V}`.
    Each vertex (besides the root) is in the tail of many possible
    hyperedges :math:`e \in {\cal E}`, and (besides terminal vertices)
    at the head of many other edges.

    The vertex object has access to the subedges of the vertex
    or a bit indicating it is a terminal vertex. It also optionally
    has an associated label, which may be any python object.

    Attributes
    -------------

    subedges : iterator of :py:class:`Edge` s

       The hyperedges that have this vertex as head.

       We write this as :math:`\{e \in {\cal E} : h(e) = v \}`

    is_terminal : bool
       Indicates whether this vertex is terminal (no-subedges).

    label : any
        Data associated with the vertex.
    """

    cdef Vertex init(self, const CHypernode *nodeptr,
                   Hypergraph graph):
        self.nodeptr = nodeptr
        self.graph = graph
        return self

    def __dealloc__(self):
        pass

    def __hash__(self):
        return self.id

    def __repr__(self):
        return "NODE:%d" % (self.nodeptr.id())

    property id:
        def __get__(self):
            assert self.nodeptr.id() != -1, "Bad node id."
            return self.nodeptr.id()

    property subedges:
        def __get__(self):
            return convert_edges(self.nodeptr.edges(), self.graph)

    property edges:
        def __get__(self):
            return self.subedges

    property is_terminal:
        def __get__(self):
            return (self.nodeptr.edges().size() == 0)

    property label:
        def __get__(self):
            return self.graph.labeling[self]

    def __str__(self):
        return str(self.nodeptr.id())

    def __cinit__(self):
        ""
        pass

    def _removed(self):
        return self.nodeptr == NULL or (self.nodeptr.id() == -1)

cdef class Node(Vertex):
    pass

cdef class Edge:
    r"""
    Hypergraph hyperedge.


    A hypergraph constains a set of hyperedge :math:`e \in {\cal E}`.
    at the head of many other edges.  A hyperedge is a vector
    :math:`\langle v_1 , \langle v_2 \ldots v_{n} \rangle \rangle`
    where :math:`v_1` is a head vertex and :math:`v_2 \ldots v_{n}` is
    a tail.

    We represent a hyperedge with a reference to the head vertex,
    an iterator of tail vertices, and a label which may be any
    piece of python data.

    Attributes
    -----------

    head : :py:class:`Vertex`
        The head vertex :math:`v_1`.

    tail : iterator of :py:class:`Vertex`
        The tail vertices :math:`v_2 \ldots v_{n}`.

    label : any
        Data associated with the hyperedge.
    """

    def __cinit__(self):
        ""
        pass

    def __hash__(self):
        return self.thisptr

    def __dealloc__(self):
        pass

    cdef Edge init(self, int ptr, Hypergraph graph, bool unfinished=False):
        self.edgeptr = ptr
        self.graph = graph
        self.unfinished = unfinished
        return self

    def __repr__(self):
        return "EDGE:%d" % (self.edgeptr)

    property tail:
        def __get__(self):
            return [Vertex().init(self.graph.thisptr.tail_node(self.id, i), self.graph)
                    for i in range(self.graph.thisptr.tail_nodes(self.id))]


    property head:
        def __get__(self):
            return Vertex().init(self.graph.thisptr.head(self.id), self.graph)

    property label:
        def __get__(self):
            return self.graph.labeling[self]

    property id:
        def __get__(self):
            assert self.edgeptr != -1, "Bad edge id."
            if self.unfinished:
                return self.graph.thisptr.new_id(self.edgeptr)
            else:
                return self.edgeptr

    def _removed(self):
        return (self.id == -1)

cdef convert_edges(vector[int] edges,
                   Hypergraph graph):
    return [Edge().init(edge, graph) for edge in edges]

cdef convert_nodes(vector[const CHypernode *] nodes,
                   Hypergraph graph):
    return [Vertex().init(node, graph) for node in nodes]


cdef class Path:
    r"""
    Path through the hypergraph.

    A (hyper)path representing a possible traversal of the hypergraph.
    A path is a member of the combinatorial set
    :math:`y \in {\cal Y}` satisfying the consistency conditions.

    We represent a path as an ordered list of edges and vertices

    Usage:

    To check if an edge is in a path ::

       >> edge in path

    Attributes
    -----------

    edges : iterator of :py:class:`Edge`
        The hyperedges in the path :math:`y_e = 1` in topological order.

    vertices : iterator of :py:class:`Vertex`
        The vertices in the path :math:`y_v = 1` in topological order.
    """

    def __dealloc__(self):
        del self.thisptr

    def __cinit__(self, Hypergraph graph=None, edges=[]):
        """
        """

        cdef vector[int ] cedges
        self.graph = graph
        edges.sort(key=lambda e: e.id)
        if graph and edges:
            for edge in edges:
                cedges.push_back((<Edge>edge).id)
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
        return self.thisptr.has_edge(edge.id)

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
            return _LazyEdges(self.graph).init(self.thisptr.edges())

    property vertices:
        def __get__(self):
            return _LazyVertices(self.graph).init(self.thisptr.nodes())

    property nodes:
        def __get__(self):
            return self.vertices


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


cdef class HypergraphMap:
    """
    Map between two hypergraphs.

    It is often useful to indicate the relationship between edges
    in multiple hypergraphs. Say we have two hypergraphs
    :math:`({\cal V}, {\cal E})` and :math:`({\cal V}', {\cal E}')`.
    This class represents a function :math:`m : {\cal V} \cup {\cal E} \mapsto {\cal V}' \cup {\cal E}'`.


    Usage:

    To map a vertex or edge ::

       >> hypergraph_map[edge]

       >> hypergraph_map[vertex]

    It can also be used to map objects over a hypergraph, for instance ::

       >> hypergraph_map[potentials]

    Attributes
    -----------
    domain_hypergraph : :py:class:`Hypergraph`
      Hypergraph in the domain  of the map :math:`({\cal V}, {\cal E})`

    range_hypergraph : :py:class:`Hypergraph`
      Hypergraph in the range of the map :math:`({\cal V}', {\cal E})'`
    """
    def __cinit__(self):
        self.thisptr = NULL

    cdef HypergraphMap init(self,
                            const CHypergraphMap *thisptr,
                            Hypergraph domain_graph,
                            Hypergraph range_graph):

        self.thisptr = thisptr
        assert thisptr.domain_graph().id() >= 0
        assert thisptr.range_graph().id() >= 0
        if range_graph is None:
            self.domain_graph = domain_graph
            assert self.domain_graph.thisptr.id() == \
                self.thisptr.domain_graph().id()
            self.range_graph = self._build_range_hypergraph()
        elif domain_graph is None:
            self.range_graph = range_graph
            assert self.range_graph.thisptr.id() == \
                self.thisptr.range_graph().id()
            self.domain_graph = self._build_domain_hypergraph()
        else:
            self.domain_graph = domain_graph
            self.range_graph = range_graph
            assert self.range_graph.thisptr.id() == \
                self.thisptr.range_graph().id()
            assert self.domain_graph.thisptr.id() == \
                self.thisptr.domain_graph().id()

        return self

    def compose(self, HypergraphMap other):
        """
        Compose two hypergraph maps.

        Parameters
        -----------
        other : :py:class:`HypergraphMap`
          A map of type :math:`m' : {\cal V}' \cup {\cal E}' \mapsto {\cal V}'' \cup {\cal E}''`

        Returns
        ---------
        composed_map : :py:class:`HypergraphMap`
          A map of type :math:`m'' : {\cal V} \cup {\cal E} \mapsto {\cal V}'' \cup {\cal E}''`


        """
        cdef CHypergraphMap *newptr = \
            self.thisptr.compose(deref(other.thisptr))
        return HypergraphMap().init(newptr,
                                    other.domain_graph,
                                    self.range_graph)

    def invert(self):
        """
        TODO: fill in
        """
        cdef CHypergraphMap *newptr = self.thisptr.invert()
        return HypergraphMap().init(newptr,
                                    self.range_graph,
                                    self.domain_graph)

    property domain_hypergraph:
        def __get__(self):
            return self.domain_graph

    property range_hypergraph:
        def __get__(self):
            return self.range_graph

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    def __getitem__(self, obj):
        cdef int edge
        cdef const CHypernode *node
        if isinstance(obj, Edge):
            edge = self.thisptr.map(<int>((<Edge>obj).id))
            # assert edge.id() >= 0
            # assert edge.id() == self.range_graph.edges[edge.id()].id
            if edge >= 0:
                return self.range_graph.edges[edge]
            else:
                return None
        if isinstance(obj, Vertex):
            node = self.thisptr.map((<Vertex>obj).nodeptr)
            if node != NULL and node.id() >= 0:
                return self.range_graph.nodes[node.id()]
            else:
                return None

        if isinstance(obj, Hypergraph):
            assert obj.thisptr.id() == self.domain_hypergraph.thisptr.id()
            return self.range_hypergraph

        return obj.project(self.range_graph, self)

    def _build_range_hypergraph(self):
        cdef const CHypergraphMap *projection = self.thisptr

        # Map nodes.
        node_labels = [None] * projection.range_graph().nodes().size()
        edge_labels = [None] * projection.range_graph().edges().size()
        cdef vector[const CHypernode*] old_nodes = \
            projection.domain_graph().nodes()
        cdef vector[int] old_edges = \
            projection.domain_graph().edges()

        cdef const CHypernode *node
        cdef int edge

        for i in range(old_nodes.size()):
            node = self.thisptr.map(old_nodes[i])
            if node != NULL and node.id() >= 0:
                node_labels[node.id()] = \
                    self.domain_graph.labeling.node_labels[i]

        if self.domain_graph.labeling.edge_labels:
            for i in range(old_edges.size()):
                edge = self.thisptr.map(old_edges[i])
                if edge >= 0:
                    edge_labels[edge] = \
                        self.domain_graph.labeling.edge_labels[i]

        cdef Hypergraph h = Hypergraph()
        return h.init(projection.range_graph(),
                      Labeling(h, node_labels, edge_labels))

    def _build_domain_hypergraph(self):
        cdef const CHypergraph *graph = self.thisptr.domain_graph()
        assert graph.id() >= 0
        node_labels = [None] * graph.nodes().size()
        edge_labels = [None] * graph.edges().size()
        cdef const CHypernode *node
        cdef int edge

        for i in range(graph.nodes().size()):
            node = self.thisptr.map(graph.nodes()[i])
            if node != NULL and node.id() >= 0:
                node_labels[i] = \
                    self.range_graph.labeling.node_labels[node.id()]

        if self.range_graph.labeling.edge_labels:
            for i in range(graph.edges().size()):
                edge = self.thisptr.map(graph.edges()[i])
                if  edge >= 0:
                    edge_labels[i] = \
                        self.range_graph.labeling.edge_labels[edge]

        cdef Hypergraph h = Hypergraph()
        return h.init(graph, Labeling(h, node_labels, edge_labels))


# For finite-state automaton construction.

cdef class DFA:
    def __init__(self, int num_states,
                 int num_symbols, transitions,
                 final):
        cdef vector[map[int, int]] ctransitions
        ctransitions.resize(num_states)
        for i, m in enumerate(transitions):
            for k in m:
                ctransitions[i][k] = m[k]

        cdef set[int] cfinals
        for f in final:
            cfinals.insert(f)

        self.thisptr = new CDFA(num_states, num_symbols,
                                ctransitions, cfinals)

    def is_final(self, int state):
        return self.thisptr.final(state)

    def transition(self, int state, int symbol):
        return self.thisptr.transition(state, symbol)

    def valid_transition(self, int state, int symbol):
        return self.thisptr.valid_transition(state, symbol)

cdef class DFALabel:
    cdef init(self, CDFALabel label, core):
        self.label = label
        self._core = core
        return self

    property left_state:
        def __get__(self):
            return self.label.left_state

    property right_state:
        def __get__(self):
            return self.label.right_state

    property core:
        def __get__(self):
            return self._core

    def __str__(self):
        return str(self.core) + " " + str(self.label.left_state) + " " + str(self.label.right_state)


# For lattice construction.

cdef class LatticeLabel:
    cdef init(self, CLatticeLabel label):
        self.label = label
        return self

    property i:
        def __get__(self):
            return self.label.i

    property j:
        def __get__(self):
            return self.label.j

    def __str__(self):
        return str(self.i) + " " + str(self.j)

def make_lattice(int width, int height, transitions):
    cdef vector[vector[int] ] ctrans
    cdef vector[int] tmp
    for i in range(len(transitions)):
        for j in range(len(transitions[i])):
            tmp.push_back(transitions[i][j])
        ctrans.push_back(tmp)
        tmp.clear()
    cdef Hypergraph h = Hypergraph()

    cdef vector[CLatticeLabel] clabels
    cdef CHypergraph *chyper = cmake_lattice(width, height, ctrans, &clabels)

    node_labels = [LatticeLabel().init(clabels[i])
                   for i in range(clabels.size())]
    assert(chyper.nodes().size() == len(node_labels))
    return h.init(chyper,
                  Labeling(h, node_labels, None))
