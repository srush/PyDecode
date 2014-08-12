from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray
import scipy.sparse
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
#DEFINE NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cdef _hypergraph_registry_counts = {}

cdef class _LazyEdges:
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, item):
        return Edge().init(self._graph.thisptr.edges()[item],
                           self._graph)

    def __iter__(self):
        cdef int edge
        cdef const vector[int] *edges = &self._graph.thisptr.edges()
        return (Edge().init(deref(edges)[i], self._graph)
                for i in range(edges.size()))

    def __len__(self):
        return self._graph.thisptr.edges().size()

cdef class _LazyVertices:
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, item):
        return Vertex().init(self._graph.thisptr.nodes()[item],
                             self._graph)

    def __iter__(self):
        cdef const vector[int] *nodes = &self._graph.thisptr.nodes()
        cdef int node
        return (Vertex().init(deref(nodes)[i], self._graph)
                for i in range(nodes.size()))

    def __len__(self):
        return self._graph.thisptr.nodes().size()

cdef class Hypergraph:
    r"""
    A directed hypergraph is a generalization of a directed graph
    where each (hyper)edge is allowed have multiple tail vertices,
    writen as :math:`\langle v_1 , \langle v_2 \ldots v_{n} \rangle
    \rangle`.

    Acyclic, directed hypergraphs are commonly used to represent
    dynamic programs where each vertex corresponds to an item, and
    each hyperedge corresponds to a recursive statement. Roughly
    each hyperedge is used to stand in for

    .. math::

       C_{v_1}  \oplus=  C_{v_2} \otimes C_{v_3} \ldots \otimes C_{v_n} \otimes w(l)

    where :math:`l` is a label associated with the edge.

    Each hypergraph consists of a set of vertices :math:`{\cal V}`,
    hyperedges :math:`{\cal E}`, a distinguised root vertex :math:`v_0
    \in {\cal V}`, and a label vector :math:`l \in L^{\cal E}` mapping
    each hyperedge to a label in :math:`L`.

    Attributes
    -----------

    vertices : iterator of :py:class:`Vertex`
      Vertices :math:`{\cal V}` in topological order.

    edges : iterator of :py:class:`Edge`
      Hyperedges :math:`{\cal E}` in topological order.

    root : :py:class:`Vertex`
      Root vertex :math:`v_0 \in {\cal V}`.

    labeling : int ndarray
      The labeling :math:`l`.
    """
    def __cinit__(Hypergraph self, bool unary=False):
        """
        Create a new hypergraph.
        """
        self.thisptr = NULL
        # self.labeling = None
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
        # self.labeling = labeling
        return self

    # def builder(self):
    #     self.thisptr = new CHypergraph(self.unary)
    #     #_hypergraph_registry[self.thisptr.id()] = self
    #     _hypergraph_registry_counts[self.thisptr.id()] = 1
    #     return GraphBuilder().init(self, self.thisptr)

    property vertices:
        def __get__(self):
            return _LazyVertices(self)

    property nodes:
        def __get__(self):
            return _LazyVertices(self)

    property root:
        def __get__(self):
            return Vertex().init(self.thisptr.root(), self)

    property edges:
        def __get__(self):
            return _LazyEdges(self)

    property labeling:
        def __get__(self):
            cdef int *labels = self.thisptr.labels()
            cdef int size = self.thisptr.edges().size()
            return np.asarray(<int[:size]> labels)

    property node_labeling:
        def __get__(self):
            cdef int *labels = self.thisptr.node_labels()
            cdef int size = self.thisptr.nodes().size()
            return np.asarray(<int[:size]> labels)

            # cdef int[:] array =
            #cvarray(shape=(self.thisptr.edges().size(),), itemsize=sizeof(int), format="i")

    # property heads:
    #     def __get__(self):
    #         return np.array(self.thisptr.heads())

    # property labeling:
    #     def __get__(self):
    #         return self.labeling

    #     def __set__(self, labeling):
    #         self.labeling = labeling

    # def head_labels(self):
    #     for i in range(self.thisptr.edges().size()):
    #         edge_num = self.thisptr.edges()[i]
    #         label = self.labeling.node_labels[self.thisptr.head(edge_num).id()]
    #         if label is not None:
    #             yield edge_num, label

    # def node_labels(self):
    #     for i in range(self.thisptr.edges().size()):
    #         edge_num = self.thisptr.edges()[i]
    #         label = self.labeling.node_labels[self.thisptr.head(edge_num).id()]
    #         tail_labels = [self.labeling.node_labels[self.thisptr.tail_node(edge_num, tail).id()]
    #                        for tail in range(self.thisptr.tail_nodes(edge_num))]
    #         if label is not None:
    #             yield edge_num, label, tail_labels

    def summarize(self):
        s = "Hypergraph: Edges: %s Vertices: %s" % (len(self.edges),
                                                 len(self.nodes)) + "\n"
        s += "Root %s" % (self.root.id) + "\n"
        for edge in self.edges:
            s += " %s \n" % (edge.id,)
            s += "\t%d -> " % (edge.head.id)
            for node in edge.tail:
                s += " %d " % (node.id)
            s += "\n"
        return s


cdef int_cmp(int first, int second, int cmp_type):
    if cmp_type == 0: return first <  second
    if cmp_type == 2: return first == second
    if cmp_type == 4: return first >  second
    if cmp_type == 1: return first <= second
    if cmp_type == 3: return first != second
    if cmp_type == 5: return first >= second

cdef class Vertex:
    r"""
    A hypergraph contains a set of vertices :math:`v \in {\cal V}`.
    Each vertex (besides the root) is in the tail of at least one
    hyperedges :math:`e \in {\cal E}`, and (besides terminal vertices)
    at the head of other hyperedges.

    The vertex object has access to the subedges of the vertex.

    Attributes
    -------------

    subedges : iterator of :py:class:`Edge`
       The hyperedges that have this vertex as head.

    is_terminal : bool
       Indicates whether this vertex is terminal (no-subedges).
    """

    cdef Vertex init(self, int nodeptr,
                   Hypergraph graph):
        self.nodeptr = nodeptr
        self.graph = graph
        return self

    def __dealloc__(self):
        pass

    def __hash__(self):
        return self.id

    def __richcmp__(self, Vertex other, int cmp_type):
        return int_cmp(self.id, other.id, cmp_type)

    def __repr__(self):
        return "Vertex:%d" % (self.nodeptr.id())

    property id:
        def __get__(self):
            # assert self.nodeptr.id() != -1, "Bad node id."
            return self.nodeptr

    property subedges:
        def __get__(self):
            return convert_edges(self.graph.thisptr.edges(self.nodeptr),
                                 self.graph)

    property edges:
        def __get__(self):
            return self.subedges

    property is_terminal:
        def __get__(self):
            return self.graph.thisptr.terminal(self.nodeptr)

    property label:
        def __get__(self):
            return self.graph.thisptr.node_label(self.nodeptr)

    def __str__(self):
        return str(self.nodeptr)

    def __cinit__(self):
        ""
        pass

    # def _removed(self):
    #     return self.nodeptr == NULL or (self.nodeptr.id() == -1)

cdef class Node(Vertex):
    pass

cdef class Edge:
    r"""
    A hypergraph contains a set of hyperedge :math:`e \in {\cal E}`.
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

    label : int
        Label associated with the hyperedge.
    """

    def __cinit__(self):
        ""
        pass

    # def __hash__(self):
    #     return self.edgeptr

    def __dealloc__(self):
        pass

    cdef Edge init(self, int ptr, Hypergraph graph, bool unfinished=False):
        self.edgeptr = ptr
        self.graph = graph
        self.unfinished = unfinished
        return self

    def __repr__(self):
        return "EDGE:%d" % (self.edgeptr)

    def __hash__(self):
        return self.id

    def __richcmp__(self, Edge other, int cmp_type):
        return int_cmp(self.id, other.id, cmp_type)

    property tail:
        def __get__(self):
            return [Vertex().init(self.graph.thisptr.tail_node(self.id, i), self.graph)
                    for i in range(self.graph.thisptr.tail_nodes(self.id))]


    property head:
        def __get__(self):
            return Vertex().init(self.graph.thisptr.head(self.id), self.graph)


    # property head_label:
    #     def __get__(self):
    #         return self.graph.labeling.node_labels[self.graph.thisptr.head(self.id).id()]

    property label:
        def __get__(self):
            return self.graph.thisptr.label(self.edgeptr)

    property id:
        def __get__(self):
            assert self.edgeptr != -1, "Bad edge id."
            # if self.unfinished:
            #     return self.graph.thisptr.new_id(self.edgeptr)
            # else:
            return self.edgeptr

    # def _removed(self):
    #     return (self.id == -1)

cdef convert_edges(vector[int] edges,
                   Hypergraph graph):
    return [Edge().init(edge, graph) for edge in edges]

# cdef convert_nodes(vector[int] nodes,
#                    Hypergraph graph):
#     return [Vertex().init(node, graph) for node in nodes]


cdef class Path:
    r"""
    A (hyper)path representing a possible traversal of the hypergraph.

    Formally a path is a member of the combinatorial set :math:`
    {\cal Y}` satisfying the consistency conditions.


    .. math::

          y(v_0) &= 1 \\

          y(v) &= \sum_{e \in {\cal E} : h(e) = v} y(e) \ \forall v \in {\cal V} \\

          y(v) &= \sum_{e \in {\cal E} : v \in t(e)} y(e) \ \forall v \in {\cal V}


    We represent a path as an ordered list of edges and vertices.

    Attributes
    -----------

    edges : iterator of :py:class:`Edge`
        The hyperedges in the path :math:`y_e = 1` in topological order.

    vertices : iterator of :py:class:`Vertex`
        The vertices in the path in topological order.

    v : sparse array
        Indicator of hyperedges in the hyperpath.
    """

    def __dealloc__(self):
        del self.thisptr

    def __cinit__(self, Hypergraph graph=None, edges=[]):
        """
        """
        cdef vector[int ] cedges
        cdef vector[int ] cnodes
        cdef set[int] nodes
        self.graph = graph

        edges.sort(key=lambda e: e.id)
        if graph and edges:
            nodes = set[int]()
            for edge in edges:
                cedges.push_back((<Edge>edge).id)
                nodes.add(edge.head.id)
                for node in edge.tail:
                    nodes.add(node.id)
            for node in nodes:
                cnodes.push_back(node)

            self.thisptr = new CHyperpath(graph.thisptr, cnodes, cedges)
            self.vector = None
            self._vertex_vector = None
            #self._make_vector()
            self._edge_indices = \
                np.array(self.thisptr.edges())

            self._vertex_indices = \
                np.array(self.thisptr.nodes())

    def _make_vector(self):
        data = []
        indices = []
        for edge in self.edges:
            indices.append(edge.id)
            data.append(1)
        self.vector = scipy.sparse.csc_matrix(
            (data, indices, [0, len(data)]),
            shape=(len(self.graph.edges),1),
            dtype=np.uint16)

        data = []
        indices = []
        for vertex in self.vertices:
            indices.append(vertex.id)
            data.append(1)
        self._vertex_vector = scipy.sparse.csc_matrix(
            (data, indices, [0, len(data)]),
            shape=(len(self.graph.vertices),1),
            dtype=np.uint16)

    cdef Path init(self, const CHyperpath *path, Hypergraph graph):
        self.thisptr = path
        self.graph = graph

        cdef int edge
        self._edge_indices = \
            np.array(self.thisptr.edges())
        self._vertex_indices = \
            np.array(self.thisptr.nodes())

        self.vector = None
        self._vertex_vector = None
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

    property edge_indices:
        def __get__(self):
            return self._edge_indices

    property vertex_indices:
        def __get__(self):
            return self._vertex_indices

    property edges:
        def __get__(self):
            return (Edge().init(self.thisptr.edges()[i], self.graph)
                    for i in range(self.thisptr.edges().size()))

    property vertices:
        def __get__(self):
            return (Vertex().init(self.thisptr.nodes()[i], self.graph)
                    for i in range(self.thisptr.nodes().size()))

    property nodes:
        def __get__(self):
            return self.vertices

    property v:
        def __get__(self):
            if self.vector is None:
                self._make_vector()
            return self.vector

    property vertex_vector:
        def __get__(self):
            if self._vertex_vector is None:
                self._make_vector()
            return self._vertex_vector

    property labeling:
        def __get__(self):
            cdef const int *labels = self.thisptr.labels()
            cdef int size = self.thisptr.edges().size()
            return np.asarray(<int[:size]> labels)


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
