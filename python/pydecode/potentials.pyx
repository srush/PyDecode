#cython: embedsignature=True

from cython.operator cimport dereference as deref
import scipy.sparse
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np

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

    cdef init(self, vector[int] edges):
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

    Hypergraph consisting of a set of M nodes :math:`{\cal V}`,
    N hyperedges :math:`{\cal E}`, and a root vertex :math:`v_0 \in {\cal V}`.

    Warning: Direct use of this interface is relatively slow and is mainly
    designed for prototyping and debugging.

    Attributes
    -----------

    vertices : list of :py:class:`Vertex` of length M
      List of vertices :math:`{\cal V}` in topological order.

    edges : list of :py:class:`Edge` of length N
      List of hyperedges :math:`{\cal E}` in topological order.

    root : :py:class:`Vertex`
      Root vertex in :math:`v_0 \in {\cal V}`.

    labeling : :py:class:`Labeling`
      The labels associated with vertices and edges. (For debugging.)

    """
    def __cinit__(Hypergraph self, bool unary=False):
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

    # def builder(self):
    #     self.thisptr = new CHypergraph(self.unary)
    #     #_hypergraph_registry[self.thisptr.id()] = self
    #     _hypergraph_registry_counts[self.thisptr.id()] = 1
    #     return GraphBuilder().init(self, self.thisptr)

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
            s += " %s %s \n" % (edge.id, edge.label)
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

    def __richcmp__(self, Vertex other, int cmp_type):
        return int_cmp(self.id, other.id, cmp_type)

    def __repr__(self):
        return "Vertex:%d" % (self.nodeptr.id())

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

    property head_label:
        def __get__(self):
            return self.graph.labeling.node_labels[self.graph.thisptr.head(self.id).id()]


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

    Attributes
    -----------

    edges : iterator of :py:class:`Edge`
        The hyperedges in the path :math:`y_e = 1` in topological order.

    vertices : iterator of :py:class:`Vertex`
        The vertices in the path in topological order.

    v : sparse Nx1 column vector
        Indicator of hyperedges in the hyperpath.

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
            self._make_vector()

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

    cdef Path init(self, const CHyperpath *path, Hypergraph graph):
        self.thisptr = path
        self.graph = graph
        self._make_vector()
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

    property v:
        def __get__(self):
            return self.vector

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


def count_constrained_viterbi(Hypergraph graph,
                              _LogViterbiPotentials potentials,
                              _CountingPotentials counts,
                              int limit):
    """
    DEPRECATED
    """

    cdef CHyperpath *path = \
        ccount_constrained_viterbi(graph.thisptr,
                                   deref(potentials.thisptr),
                                   deref(counts.thisptr),
                                   limit)

    return Path().init(path, graph)


def extend_hypergraph_by_count(Hypergraph graph,
                               _CountingPotentials potentials,
                               int lower_limit,
                               int upper_limit,
                               int goal):
    """
    DEPRECATED
    """

    cdef CHypergraphMap *projection = \
        cextend_hypergraph_by_count(graph.thisptr,
                                    deref(potentials.thisptr),
                                    lower_limit,
                                    upper_limit,
                                    goal)

    return convert_hypergraph_map(projection, None, graph)


def extend_hypergraph_by_dfa(Hypergraph graph,
                             _CountingPotentials potentials,
                             DFA dfa):
    """
    DEPRECATED
    """

    cdef vector[CDFALabel] labels
    cdef CHypergraphMap *projection = \
        cextend_hypergraph_by_dfa(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(dfa.thisptr),
                                  &labels)
    node_labels = []
    cdef const CHypernode *node
    cdef vector[const CHypernode*] new_nodes = \
        projection.domain_graph().nodes()

    for i in range(labels.size()):
        node = projection.map(new_nodes[i])
        node_labels.append(DFALabel().init(labels[i],
                                           graph.labeling.node_labels[node.id()]))

    # Build domain graph
    cdef Hypergraph range_graph = Hypergraph()
    assert(projection.domain_graph().nodes().size() == \
               len(node_labels))
    range_graph.init(projection.domain_graph(),
                     Labeling(range_graph, node_labels, None))
    return convert_hypergraph_map(projection, range_graph, graph)

def binarize(Hypergraph graph):
    """
    Binarize a hypergraph by making all k-ary edges right branching.

    Parameters
    ----------
    graph : Hypergraph

    Returns
    --------
    map : HypergraphMap
        A map from the original graph to the binary branching graph.
    """
    cdef CHypergraphMap *hypergraph_map = cbinarize(graph.thisptr)
    return convert_hypergraph_map(hypergraph_map, graph, None)

import numpy as np
cimport cython

class IdHasher:
    def hash_item(self, i):
        return i

cdef class IndexedEncoder:
    """
    Encoder mapping integer tuples to integers.

    Encodes the mapping N_1 x N_2 x ... N_k -> {0...|N_1||N_2|...|N_k|}.
    Attributes
    ----------
    max_size : int
        The range size |N_1||N_2|...|N_k|.

    """
    def __cinit__(self, shape):
        """
        Initialize the encoder from set sizes.

        Parameters
        -----------
        size : int tuple
           A tuple of (|N_1|, |N_2|, ..., |N_k|)

        """
        self._multipliers = np.zeros([len(shape), 1], dtype=np.int32)
        self._shape = shape
        self._multipliers[0] = 1

        for j in range(1, len(shape)):
            self._multipliers[j] = self._multipliers[j-1] * shape[j-1]
        self._max_size = np.product(shape)

    cpdef np.ndarray transform(self, np.ndarray element):
        """
        Transform from tuple to int.
        """
        return np.dot(self._multipliers.T, element.T)

    cpdef np.ndarray inverse_transform(self, np.ndarray indices):
        """
        Inverse transform from int to tuple.
        """
        v = indices
        m = np.zeros((len(self._multipliers), len(indices)), dtype=np.int32)
        for j in range(len(self._multipliers) - 1, -1, -1):
            m[j, :] =  v // self._multipliers[j]
            v = v % self._multipliers[j]
        return m

    def reshape(self, matrix):
        assert matrix.shape == self._shape
        return np.reshape(matrix.T, (self.max_size, 1))

#         self._multipliers[0] = 1

#         for j in range(1, len(size)):
#             self._multipliers[j] = self._multiplier[j-1] * size[j-1]

    property max_size:
        def __get__(self):
            return self._max_size

#     cpdef int [:] transform(self, int [:,:] element):
#         """
#         Transform from tuple to int.
#         """
#         return self._multipliers * element

#     cpdef int [:,:] inverse_transform(self, int [:]  index):
#         """
#         Inverse transform from int to tuple.
#         """
#         if self._hasher is None: return index
#         return self._hasher.unhash(index)


#     def iteritems(self):
#         cdef int i
#         if self._hasher is None:
#             for i in range(self._max_size):
#                 yield i, i
#         else:
#             for i in range(self._max_size):
#                 yield i, self._hasher.unhash(i)


    def item_vector(self, elements):
        data = []
        indices = []
        ind = [0]
        for element in elements:
            data.append(1)
            indices.append(self.transform(element))
        ind.append(len(data))
        return scipy.sparse.csc_matrix(
            (data, indices, ind),
            shape=(self._max_size, 1),
            dtype=np.uint8)


cdef class _ChartEdge:
    pass

cdef class ChartBuilder:
    """
    ChartBuilder is an interface for specifying dynamic programs.

    The chart acts like a dictionary between items I and "tokens". ::
       >> c[item] = c.init()
       >> c[item2] = [c.merge(item)]
       >> c[item3] = [c.merge(item, item2), c.merge(item)]
       >> c[item4] = [c.merge(item3, out=[output])]

    When a chart is complete, it creates a hypergraph (V, E).

       >> hypergraph = c.finish()

    The chart builder also maintains a mapping between the hypergraph
    and item set and output set.

    Define the set of items I to specify the cells in a dynamic
    programming chart.

    Define the set of outputs O to specify the output emitted by a
    decision in the dynamic program.

    Attributes
    ----------
    item_encoder : Encoder I -> {0...|I|}
       Encodes elements of the item set I as integers.

    output_encoder : Encoder O -> {0...|O|}
       Encodes elements of the output set O as integers.

    item_matrix : Sparse matrix |I| x |V|
       Maps vertices of the hypergraph to item indicators.

    output_matrix : Sparse matrix |O| x |E|
       Maps hyperedges of the hypergraph to sparse output counts.
    """

    def __init__(self,
                 item_encoder,
                 item_set_size=None,
                 output_encoder=None,
                 output_size=None,
                 unstrict=False,
                 expected_size=None):
        """
        Initialize the dynamic programming chart.

        Parameters
        ------------

        item_encoder, item_set_size : Encoder I -> {0...|I|}, Int
            Specifies the item set I for chart, and the size |I|
            The encoder must have a `transform` method.

        output_encoder : Encoder O -> {0...|O|}
            Specifies the item set O for chart.
            The encoder must have a `transform` method.

        unstrict : bool
            Allows the chart to merge NULL items.

        expected_size : (int, int)
            Set the expected number of edges |E| and
            the max-arity of edges. Useful for efficiency.
        """

        self._done = False
        self._last = -1
        self._strict = not unstrict
        self._hg_ptr = new CHypergraph(False)

        self._item_encoder = item_encoder
        cdef int size = item_set_size
        self._chart = new vector[const CHypernode *](size, NULL)
        self._item_size = size

        self._output_encoder = output_encoder
        self._output_size = output_size

        self._label_chart = []

        if expected_size:
            self._hg_ptr.set_expected_size(size,
                                           expected_size[0],
                                           expected_size[1])
            self._max_arity = expected_size[1]

        self._data = []
        self._indices = []
        self._ind = [0]

        self._ndata = []
        self._nindices = []
        self._nind = [0]

    # def init(self):
    #     """
    #     Returns the initial value for a chart item. Usage::

    #        >> c[item] = c.init()
    #     Returns
    #     --------
    #      : token
    #        Chart token used to initialize a cell.

    #     """
    #     return _ChartEdge()

    # cpdef _ChartEdge merge():
    #     cdef _ChartEdge chart_edge = _ChartEdge()

    def merge(self, *args, out=[]):
        """
        Merges the items given as arguments. Usage::

           >> c[item_head] = [c.merge(item_tail1, item_tail2, out=[2])]

        Parameters
        ----------
        *args : list of items in I
            The items to merge.

        out : list of outputs in O
            The outputs to associate with this merge.

        Returns
        --------
         : token
           Chart token used to represent the merge.

        """

        cdef int index
        if len(args) == 0:
            raise Exception("Merge takes at least one item.")
        # chart_edge.items = np.array(args)
        return self._merge(args, out)

    @cython.boundscheck(False)
    cdef _ChartEdge  _merge(self, args, outs):
        cdef _ChartEdge chart_edge = _ChartEdge()
        chart_edge.tail_ptrs.resize(len(args))
        cdef int i = 0
        for arg in args:
            chart_edge.tail_ptrs[i] = deref(self._chart)[
                    self._item_encoder[arg]]
            i += 1
        chart_edge.values = [self._output_encoder[out] for out in outs]
        return chart_edge

    def finish(self, reconstruct=True):
        """
        Finish chart construction.

        Returns
        -------
        hypergraph :
           The hypergraph representing the dynamic program.

        """
        if self._done:
            raise Exception("Hypergraph not constructed.")
        self._done = True
        self._hg_ptr.finish(reconstruct)

        final_node_labels = [None] * self._hg_ptr.nodes().size()
        for h, label in self._label_chart:
            if deref(self._chart)[h] != NULL and deref(self._chart)[h].id() >= 0:
                final_node_labels[deref(self._chart)[h].id()] = label

        hypergraph = Hypergraph(False)
        hypergraph.init(self._hg_ptr,
                        Labeling(hypergraph, final_node_labels))
        return hypergraph

    def __contains__(self, item):
        cdef int index = self._item_encoder.transform(np.array([item]))[0,0]
        return deref(self._chart)[index] != NULL

    # def build_up(self, chart_edges):
    #     cdef _ChartEdge chart_edge

    #     cdef np.ndarray[np.int_t, ndim=2] stack1 = np.zeros([len(chart_edges), 4],
    #                                                        dtype=np.int)
    #     cdef np.ndarray[np.int_t, ndim=2] stack2 = np.zeros([len(chart_edges), 4],
    #                                                        dtype=np.int)
    #     cdef np.ndarray[np.int_t, ndim=2] values = np.zeros([len(chart_edges), 2],
    #                                                        dtype=np.int)

    #     cdef int j = 0
    #     cdef int i
    #     cdef np.ndarray[np.int_t, ndim=2] items
    #     for i, chart_edge in enumerate(chart_edges):
    #         items = chart_edge.items
    #         stack1[i,:] = items[0]
    #         stack2[i,:] = items[1]
    #         for v in chart_edge.values:
    #             values[j,:] = v
    #             j+=1


        # ind1 = self._item_encoder.transform(stack1)[0]
        # ind2 = self._item_encoder.transform(stack2)[0]
        # trans_values = self._output_encoder.transform(values)[0]
        # return ind1, ind2, trans_values

    # def build_up2(self, edges1, edges2, values):
    #     cdef _ChartEdge chart_edge

        # cdef np.ndarray[np.int_t, ndim=2] stack1 = np.zeros([len(edges1), 4],
        #                                                    dtype=np.int)
        # cdef np.ndarray[np.int_t, ndim=2] stack2 = np.zeros([len(edges2), 4],
        #                                                    dtype=np.int)
        # cdef np.ndarray[np.int_t, ndim=2] values_a = np.zeros([len(values), 2],
        #                                                    dtype=np.int)

        # cdef int j = 0
        # cdef int i
        # cdef np.ndarray[np.int_t, ndim=2] items
        # for i, edge in enumerate(edges1):
        #     stack1[i,:] = edges1[i]
        #     stack2[i,:] = edges2[i]
            # for v in chart_edge.values:
            #     values[j,:] = v
            #     j+=1


        # ind1 = self._item_encoder.transform(edges1)[0]
        # ind2 = self._item_encoder.transform(edges2)[0]
        # trans_values = self._output_encoder.transform(values)[0]
        # return ind1, ind2, trans_values

    # def __setitem__(self, key, q):

    #     cdef int index = self._item_encoder.transform(np.array([key]))[0,0]
    #     if self._strict and deref(self._chart)[index] != NULL:
    #         raise Exception("Chart already has label %s"%(key,))
    #     if isinstance(q, _ChartEdge):
    #         deref(self._chart)[index] = self._hg_ptr.add_terminal_node()
    #         self._label_chart.append((index, key))

    #         self._ndata.append(1)
    #         self._nindices.append(index)
    #         self._nind.append(len(self._ndata))
    #         return
    #     arr = q
    #     values = np.array([[0,0]])
    #     deref(self._chart)[index] = self._hg_ptr.start_node()

    #     ind1, ind2, trans_values = self.build_up2(arr[:,:4], arr[:,4:], values)

    #     cdef int j = 0
    #     cdef int i
    #     cdef vector[const CHypernode *] tail_ptrs
    #     #cdef _ChartEdge chart_edge
    #     for i in range(len(ind1)):
    #         tail_ptrs.clear()
    #         tail_ptrs.push_back(deref(self._chart)[ind1[i]])
    #         tail_ptrs.push_back(deref(self._chart)[ind2[i]])

    #         edge_num = self._hg_ptr.add_edge(tail_ptrs)

    #         # for v in chart_edge.values:
    #         #     self._data.append(1)
    #         #     self._indices.append(trans_values[j])
    #         #     j += 1
    #         # self._ind.append(len(self._data))

    #     result = self._hg_ptr.end_node()
    #     if not result:
    #         if self._strict:
    #             raise Exception("Index failed. %s"%(key,))
    #         deref(self._chart)[index] = NULL

    #     self._last = index
    #     self._label_chart.append((index, key))

    #     self._ndata.append(1)
    #     self._nindices.append(index)
    #     self._nind.append(len(self._ndata))

    @cython.boundscheck(False)
    cpdef init(self, int index):
        deref(self._chart)[index] = self._hg_ptr.add_terminal_node()
        # self._label_chart.append((index, key))

        self._ndata.append(1)
        self._nindices.append(index)
        self._nind.append(len(self._ndata))


    @cython.boundscheck(False)
    cpdef set2(self, long key, long [:] edges1, long [:] edges2,
               long [:] labels):
        cdef long index = key
        deref(self._chart)[index] = self._hg_ptr.start_node()
        cdef vector[const CHypernode *] tails
        J = edges1.shape[0]
        cdef int i, j
        for i in range(2):
            tails.clear()
            for j in range(J):
                if i == 0:
                    tails.push_back(deref(self._chart)[edges1[j]])
                elif i == 1:
                    tails.push_back(deref(self._chart)[edges2[j]])
            edge_num = self._hg_ptr.add_edge(tails)

        # for i in range(labels.shape[0]):
        #     self._ind.append(labels[i])
        #     self._data.append(1)
        #     self._indices.append(len(self._data))


        result = self._hg_ptr.end_node()
        if not result:
            if self._strict:
                raise Exception("Index failed. %s"%(key,))
            deref(self._chart)[index] = NULL

        self._last = index
        self._label_chart.append((index, key))

        self._ndata.append(1)
        self._nindices.append(index)
        self._nind.append(len(self._ndata))

        # cdef int index = self._item_encoder[key]
        # if self._strict and deref(self._chart)[index] != NULL:
        #     raise Exception("Chart already has label %s"%(key,))
        # if isinstance(chart_edges, _ChartEdge):
        #     deref(self._chart)[index] = self._hg_ptr.add_terminal_node()
        #     self._label_chart.append((index, key))

        #     self._ndata.append(1)
        #     self._nindices.append(index)
        #     self._nind.append(len(self._ndata))
        #     return



        # ind1, ind2, trans_values = self.build_up(chart_edges)

        # cdef int j = 0
        # cdef int i
        # cdef _ChartEdge chart_edge
        # for i, chart_edge in enumerate(chart_edges):
        #     edge_num = self._hg_ptr.add_edge(
        #         chart_edge.tail_ptrs)


        #     self._data += [1] * len(chart_edge.values)
        #     self._indices += chart_edge.values
        #     self._ind.append(len(self._data))


    property output_encoder:
        def __get__(self):
            return self._output_encoder

    property item_encoder:
        def __get__(self):
            return self._item_encoder

    property output_matrix:
        def __get__(self):
            return scipy.sparse.csc_matrix(
                (self._data, self._indices, self._ind),
                shape=(self._output_size,
                       self._hg_ptr.edges().size()),
                dtype=np.uint8)

    property item_matrix:
        def __get__(self):
            return scipy.sparse.csc_matrix(
                (self._ndata, self._nindices, self._nind),
                shape=(self._item_size,
                       self._hg_ptr.nodes().size()),
                dtype=np.uint8)

# 
# cdef class IntTuple2:
#     def __cinit__(self, int a, int b, 
#                   blank=None):
#     
#         self.a = a
#     
#         self.b = b
#     

#     def unpack(self):
#         return ( self.a,  self.b, )

# cdef class IntTuple2Hasher:
#     def __cinit__(self, int a, int b, ):
#         # n = 2
#         # m = ( a, b,)
#         self._multipliers_a = 1
#         
#         self._multipliers_b = self._multipliers_a * a
#         
#         self._max_size =  a *  b *  1

#     def max_size(self):
#         return self._max_size

#     cpdef unhash(self, int val):
#         cdef t = []
#         cdef int v = val
#         
#         #for k in range(2-1, -1, -1):
#         t.insert(0, v / self._multipliers_b)
#         v = v % self._multipliers_b
#         
#         #for k in range(2-1, -1, -1):
#         t.insert(0, v / self._multipliers_a)
#         v = v % self._multipliers_a
#         
#         return tuple(t)



#     cpdef int hash_item(self, a, b, ):
#         return \
#            \
#             a * self._multipliers_a + \
#            \
#             b * self._multipliers_b + \
#            + 0
# 
# cdef class IntTuple3:
#     def __cinit__(self, int a, int b, int c, 
#                   blank=None):
#     
#         self.a = a
#     
#         self.b = b
#     
#         self.c = c
#     

#     def unpack(self):
#         return ( self.a,  self.b,  self.c, )

# cdef class IntTuple3Hasher:
#     def __cinit__(self, int a, int b, int c, ):
#         # n = 3
#         # m = ( a, b, c,)
#         self._multipliers_a = 1
#         
#         self._multipliers_b = self._multipliers_a * a
#         
#         self._multipliers_c = self._multipliers_b * b
#         
#         self._max_size =  a *  b *  c *  1

#     def max_size(self):
#         return self._max_size

#     cpdef unhash(self, int val):
#         cdef t = []
#         cdef int v = val
#         
#         #for k in range(3-1, -1, -1):
#         t.insert(0, v / self._multipliers_c)
#         v = v % self._multipliers_c
#         
#         #for k in range(3-1, -1, -1):
#         t.insert(0, v / self._multipliers_b)
#         v = v % self._multipliers_b
#         
#         #for k in range(3-1, -1, -1):
#         t.insert(0, v / self._multipliers_a)
#         v = v % self._multipliers_a
#         
#         return tuple(t)



#     cpdef int hash_item(self, a, b, c, ):
#         return \
#            \
#             a * self._multipliers_a + \
#            \
#             b * self._multipliers_b + \
#            \
#             c * self._multipliers_c + \
#            + 0
# 
# cdef class IntTuple4:
#     def __cinit__(self, int a, int b, int c, int d, 
#                   blank=None):
#     
#         self.a = a
#     
#         self.b = b
#     
#         self.c = c
#     
#         self.d = d
#     

#     def unpack(self):
#         return ( self.a,  self.b,  self.c,  self.d, )

# cdef class IntTuple4Hasher:
#     def __cinit__(self, int a, int b, int c, int d, ):
#         # n = 4
#         # m = ( a, b, c, d,)
#         self._multipliers_a = 1
#         
#         self._multipliers_b = self._multipliers_a * a
#         
#         self._multipliers_c = self._multipliers_b * b
#         
#         self._multipliers_d = self._multipliers_c * c
#         
#         self._max_size =  a *  b *  c *  d *  1

#     def max_size(self):
#         return self._max_size

#     cpdef unhash(self, int val):
#         cdef t = []
#         cdef int v = val
#         
#         #for k in range(4-1, -1, -1):
#         t.insert(0, v / self._multipliers_d)
#         v = v % self._multipliers_d
#         
#         #for k in range(4-1, -1, -1):
#         t.insert(0, v / self._multipliers_c)
#         v = v % self._multipliers_c
#         
#         #for k in range(4-1, -1, -1):
#         t.insert(0, v / self._multipliers_b)
#         v = v % self._multipliers_b
#         
#         #for k in range(4-1, -1, -1):
#         t.insert(0, v / self._multipliers_a)
#         v = v % self._multipliers_a
#         
#         return tuple(t)



#     cpdef int hash_item(self, a, b, c, d, ):
#         return \
#            \
#             a * self._multipliers_a + \
#            \
#             b * self._multipliers_b + \
#            \
#             c * self._multipliers_c + \
#            \
#             d * self._multipliers_d + \
#            + 0
# 
# cdef class IntTuple5:
#     def __cinit__(self, int a, int b, int c, int d, int e, 
#                   blank=None):
#     
#         self.a = a
#     
#         self.b = b
#     
#         self.c = c
#     
#         self.d = d
#     
#         self.e = e
#     

#     def unpack(self):
#         return ( self.a,  self.b,  self.c,  self.d,  self.e, )

# cdef class IntTuple5Hasher:
#     def __cinit__(self, int a, int b, int c, int d, int e, ):
#         # n = 5
#         # m = ( a, b, c, d, e,)
#         self._multipliers_a = 1
#         
#         self._multipliers_b = self._multipliers_a * a
#         
#         self._multipliers_c = self._multipliers_b * b
#         
#         self._multipliers_d = self._multipliers_c * c
#         
#         self._multipliers_e = self._multipliers_d * d
#         
#         self._max_size =  a *  b *  c *  d *  e *  1

#     def max_size(self):
#         return self._max_size

#     cpdef unhash(self, int val):
#         cdef t = []
#         cdef int v = val
#         
#         #for k in range(5-1, -1, -1):
#         t.insert(0, v / self._multipliers_e)
#         v = v % self._multipliers_e
#         
#         #for k in range(5-1, -1, -1):
#         t.insert(0, v / self._multipliers_d)
#         v = v % self._multipliers_d
#         
#         #for k in range(5-1, -1, -1):
#         t.insert(0, v / self._multipliers_c)
#         v = v % self._multipliers_c
#         
#         #for k in range(5-1, -1, -1):
#         t.insert(0, v / self._multipliers_b)
#         v = v % self._multipliers_b
#         
#         #for k in range(5-1, -1, -1):
#         t.insert(0, v / self._multipliers_a)
#         v = v % self._multipliers_a
#         
#         return tuple(t)



#     cpdef int hash_item(self, a, b, c, d, e, ):
#         return \
#            \
#             a * self._multipliers_a + \
#            \
#             b * self._multipliers_b + \
#            \
#             c * self._multipliers_c + \
#            \
#             d * self._multipliers_d + \
#            \
#             e * self._multipliers_e + \
#            + 0
# 
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector


from pydecode.potentials cimport *

cdef class Bitset:
    """
    Bitset


    """
    def __init__(self, v=-1):
        if v != -1:
            self.data[v] = 1

    cdef init(self, cbitset data):
        self.data = data
        return self

    def __getitem__(self, int position):
        return self.data[position]

    def __setitem__(self, int position, int val):
        self.data[position] = val


#cython: embedsignature=True

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
cimport libcpp.map as c_map
from libcpp.pair cimport pair
from libcpp cimport bool
cimport numpy as np
import numpy as np
from cython cimport view

cdef class _Potentials:
    property kind:
        def __get__(self):
            return self.kind



############# This is the templated semiring part. ##############



cdef class _BoolPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = Bool
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphBoolPotentials *ptr):
        self.thisptr = ptr
        return self

    
    def from_array(self, char [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_Bool(self.graph.thisptr,
                                               <char *> &X[0])
        return self

    def as_array(self):
        return _Boolvector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    


cdef class BoolValue:
    cdef BoolValue init(self, char val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(char val):
        created = BoolValue()
        created.thisval = _Bool_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Bool_from_cpp(Bool_zero())

    @staticmethod
    def one_raw():
        return _Bool_from_cpp(Bool_one())

    @staticmethod
    def zero():
        return BoolValue().init(Bool_zero())

    @staticmethod
    def one():
        return BoolValue().init(Bool_one())

    def __add__(BoolValue self, BoolValue other):
        return BoolValue().init(Bool_add(self.thisval,
                                                  other.thisval))

    def __mul__(BoolValue self, BoolValue other):
        return BoolValue().init(Bool_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Bool_from_cpp(self.thisval)


cdef char _Bool_to_cpp(char val):
    return val


cdef _Bool_from_cpp(char val):
    
    return val
    





cdef _Boolvector_to_numpy(const char *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof(char),
                   format="i",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef char [:] my_view = my_array
    return np.asarray(my_view)



class Bool:
    Potentials = _BoolPotentials

    
    @staticmethod
    def inside(Hypergraph graph,
               _BoolPotentials potentials,
               char [:] chart=None):
        cdef char [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CBoolChart *in_chart = new CBoolChart(
            graph.thisptr,
            &my_chart[0])

        inside_Bool(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _BoolPotentials potentials,
                char [:] inside_chart,
                char [:] chart=None):
        cdef char [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef CBoolChart *in_chart = new CBoolChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CBoolChart *out_chart = new CBoolChart(
            graph.thisptr,
            &my_chart[0])

        outside_Bool(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _BoolPotentials potentials,
                          char [:] inside_chart,
                          char [:] outside_chart):

        cdef char [:] node_margs = np.zeros(len(graph.nodes))
        cdef char [:] edge_margs = np.zeros(len(graph.edges))

        cdef CBoolChart *in_chart = new CBoolChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CBoolChart *out_chart = new CBoolChart(
            graph.thisptr,
            &outside_chart[0])

        cdef CBoolChart *node_chart = new CBoolChart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_Bool(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_Bool(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const CBoolMarginals *marginals = \


        # return (_Boolvector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _Boolvector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    
    

    @staticmethod
    def viterbi(Hypergraph graph,
                _BoolPotentials potentials,
                char [:] chart=None):
        cdef char [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CBoolChart *in_chart = new CBoolChart(
            graph.thisptr,
            &my_chart[0])

        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)

        viterbi_Bool(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back)
        cdef CHyperpath *path = used_back.construct_path()
        del in_chart
        return Path().init(path, graph)

    


cdef class _ViterbiPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = Viterbi
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphViterbiPotentials *ptr):
        self.thisptr = ptr
        return self

    
    def from_array(self, double [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_Viterbi(self.graph.thisptr,
                                               <double *> &X[0])
        return self

    def as_array(self):
        return _Viterbivector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    


cdef class ViterbiValue:
    cdef ViterbiValue init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = ViterbiValue()
        created.thisval = _Viterbi_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Viterbi_from_cpp(Viterbi_zero())

    @staticmethod
    def one_raw():
        return _Viterbi_from_cpp(Viterbi_one())

    @staticmethod
    def zero():
        return ViterbiValue().init(Viterbi_zero())

    @staticmethod
    def one():
        return ViterbiValue().init(Viterbi_one())

    def __add__(ViterbiValue self, ViterbiValue other):
        return ViterbiValue().init(Viterbi_add(self.thisval,
                                                  other.thisval))

    def __mul__(ViterbiValue self, ViterbiValue other):
        return ViterbiValue().init(Viterbi_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Viterbi_from_cpp(self.thisval)


cdef double _Viterbi_to_cpp(double val):
    return val


cdef _Viterbi_from_cpp(double val):
    
    return val
    





cdef _Viterbivector_to_numpy(const double *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef double [:] my_view = my_array
    return np.asarray(my_view)



class Viterbi:
    Potentials = _ViterbiPotentials

    
    @staticmethod
    def inside(Hypergraph graph,
               _ViterbiPotentials potentials,
               double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CViterbiChart *in_chart = new CViterbiChart(
            graph.thisptr,
            &my_chart[0])

        inside_Viterbi(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _ViterbiPotentials potentials,
                double [:] inside_chart,
                double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef CViterbiChart *in_chart = new CViterbiChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CViterbiChart *out_chart = new CViterbiChart(
            graph.thisptr,
            &my_chart[0])

        outside_Viterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _ViterbiPotentials potentials,
                          double [:] inside_chart,
                          double [:] outside_chart):

        cdef double [:] node_margs = np.zeros(len(graph.nodes))
        cdef double [:] edge_margs = np.zeros(len(graph.edges))

        cdef CViterbiChart *in_chart = new CViterbiChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CViterbiChart *out_chart = new CViterbiChart(
            graph.thisptr,
            &outside_chart[0])

        cdef CViterbiChart *node_chart = new CViterbiChart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_Viterbi(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_Viterbi(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const CViterbiMarginals *marginals = \


        # return (_Viterbivector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _Viterbivector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    
    

    @staticmethod
    def viterbi(Hypergraph graph,
                _ViterbiPotentials potentials,
                double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CViterbiChart *in_chart = new CViterbiChart(
            graph.thisptr,
            &my_chart[0])

        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)

        viterbi_Viterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back)
        cdef CHyperpath *path = used_back.construct_path()
        del in_chart
        return Path().init(path, graph)

    


cdef class _CountingPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = Counting
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphCountingPotentials *ptr):
        self.thisptr = ptr
        return self

    
    def from_array(self, int [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_Counting(self.graph.thisptr,
                                               <int *> &X[0])
        return self

    def as_array(self):
        return _Countingvector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    


cdef class CountingValue:
    cdef CountingValue init(self, int val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(int val):
        created = CountingValue()
        created.thisval = _Counting_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Counting_from_cpp(Counting_zero())

    @staticmethod
    def one_raw():
        return _Counting_from_cpp(Counting_one())

    @staticmethod
    def zero():
        return CountingValue().init(Counting_zero())

    @staticmethod
    def one():
        return CountingValue().init(Counting_one())

    def __add__(CountingValue self, CountingValue other):
        return CountingValue().init(Counting_add(self.thisval,
                                                  other.thisval))

    def __mul__(CountingValue self, CountingValue other):
        return CountingValue().init(Counting_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Counting_from_cpp(self.thisval)


cdef int _Counting_to_cpp(int val):
    return val


cdef _Counting_from_cpp(int val):
    
    return val
    





cdef _Countingvector_to_numpy(const int *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof(int),
                   format="i",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef int [:] my_view = my_array
    return np.asarray(my_view)



class Counting:
    Potentials = _CountingPotentials

    
    @staticmethod
    def inside(Hypergraph graph,
               _CountingPotentials potentials,
               int [:] chart=None):
        cdef int [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CCountingChart *in_chart = new CCountingChart(
            graph.thisptr,
            &my_chart[0])

        inside_Counting(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _CountingPotentials potentials,
                int [:] inside_chart,
                int [:] chart=None):
        cdef int [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef CCountingChart *in_chart = new CCountingChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CCountingChart *out_chart = new CCountingChart(
            graph.thisptr,
            &my_chart[0])

        outside_Counting(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _CountingPotentials potentials,
                          int [:] inside_chart,
                          int [:] outside_chart):

        cdef int [:] node_margs = np.zeros(len(graph.nodes))
        cdef int [:] edge_margs = np.zeros(len(graph.edges))

        cdef CCountingChart *in_chart = new CCountingChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CCountingChart *out_chart = new CCountingChart(
            graph.thisptr,
            &outside_chart[0])

        cdef CCountingChart *node_chart = new CCountingChart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_Counting(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_Counting(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const CCountingMarginals *marginals = \


        # return (_Countingvector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _Countingvector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    
    

    @staticmethod
    def viterbi(Hypergraph graph,
                _CountingPotentials potentials,
                int [:] chart=None):
        cdef int [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CCountingChart *in_chart = new CCountingChart(
            graph.thisptr,
            &my_chart[0])

        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)

        viterbi_Counting(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back)
        cdef CHyperpath *path = used_back.construct_path()
        del in_chart
        return Path().init(path, graph)

    


cdef class _LogViterbiPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = LogViterbi
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphLogViterbiPotentials *ptr):
        self.thisptr = ptr
        return self

    
    def from_array(self, double [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_LogViterbi(self.graph.thisptr,
                                               <double *> &X[0])
        return self

    def as_array(self):
        return _LogViterbivector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    


cdef class LogViterbiValue:
    cdef LogViterbiValue init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = LogViterbiValue()
        created.thisval = _LogViterbi_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _LogViterbi_from_cpp(LogViterbi_zero())

    @staticmethod
    def one_raw():
        return _LogViterbi_from_cpp(LogViterbi_one())

    @staticmethod
    def zero():
        return LogViterbiValue().init(LogViterbi_zero())

    @staticmethod
    def one():
        return LogViterbiValue().init(LogViterbi_one())

    def __add__(LogViterbiValue self, LogViterbiValue other):
        return LogViterbiValue().init(LogViterbi_add(self.thisval,
                                                  other.thisval))

    def __mul__(LogViterbiValue self, LogViterbiValue other):
        return LogViterbiValue().init(LogViterbi_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _LogViterbi_from_cpp(self.thisval)


cdef double _LogViterbi_to_cpp(double val):
    return val


cdef _LogViterbi_from_cpp(double val):
    
    return val
    





cdef _LogViterbivector_to_numpy(const double *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef double [:] my_view = my_array
    return np.asarray(my_view)



class LogViterbi:
    Potentials = _LogViterbiPotentials

    
    @staticmethod
    def inside(Hypergraph graph,
               _LogViterbiPotentials potentials,
               double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CLogViterbiChart *in_chart = new CLogViterbiChart(
            graph.thisptr,
            &my_chart[0])

        inside_LogViterbi(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _LogViterbiPotentials potentials,
                double [:] inside_chart,
                double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef CLogViterbiChart *in_chart = new CLogViterbiChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CLogViterbiChart *out_chart = new CLogViterbiChart(
            graph.thisptr,
            &my_chart[0])

        outside_LogViterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _LogViterbiPotentials potentials,
                          double [:] inside_chart,
                          double [:] outside_chart):

        cdef double [:] node_margs = np.zeros(len(graph.nodes))
        cdef double [:] edge_margs = np.zeros(len(graph.edges))

        cdef CLogViterbiChart *in_chart = new CLogViterbiChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CLogViterbiChart *out_chart = new CLogViterbiChart(
            graph.thisptr,
            &outside_chart[0])

        cdef CLogViterbiChart *node_chart = new CLogViterbiChart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_LogViterbi(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_LogViterbi(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const CLogViterbiMarginals *marginals = \


        # return (_LogViterbivector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _LogViterbivector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    
    

    @staticmethod
    def viterbi(Hypergraph graph,
                _LogViterbiPotentials potentials,
                double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CLogViterbiChart *in_chart = new CLogViterbiChart(
            graph.thisptr,
            &my_chart[0])

        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)

        viterbi_LogViterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back)
        cdef CHyperpath *path = used_back.construct_path()
        del in_chart
        return Path().init(path, graph)

    



cdef convert_to_sparse(vector[int] positions):
    data = []
    indices = []
    ind = [0]
    cdef int i
    for i in range(positions.size()):
        if positions[i] > -1:
            data.append(1)
            indices.append(positions[i])
        ind.append(len(data))
    return (data, indices, ind)

cdef convert_hypergraph_map(const CHypergraphMap *hyper_map,
                            graph1_arg, graph2_arg):
    cdef Hypergraph graph1 = graph1_arg
    if not graph1:
        graph1 = Hypergraph()
        graph1.init(hyper_map.domain_graph(), Labeling(graph1))

    cdef Hypergraph graph2 = graph2_arg
    if not graph2:
        graph2 = Hypergraph()
        graph2.init(hyper_map.range_graph(), Labeling(graph2))


    cdef vector[int] edges = hyper_map.edge_map()

    edge_matrix = scipy.sparse.csc_matrix(
        convert_to_sparse(hyper_map.edge_map()),
        shape=(len(graph2.edges),
               len(graph1.edges)),
        dtype=np.uint8)

    # cdef vector[int] nodes = hyper_map.node_map()

    # node_matrix = scipy.sparse.css_matrix(
    #     hyper_map.edge_map(),
    #     shape=(len(graph1.nodes),
    #            len(graph2.nodes)),
    #     dtype=np.int8)
    return graph1, edge_matrix, graph2

####### Methods that use specific potential ########

def _get_potentials(graph, potentials, kind=_LogViterbiPotentials):
    if isinstance(potentials, _Potentials):
        return potentials
    else:
        return kind(graph).from_array(potentials)

def inside(Hypergraph graph, potentials,
           kind=LogViterbi, chart=None):
    r"""
    Compute the inside values for potentials.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    potentials : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    kind : A semiring type.
      The semiring to use. Must agree with potentials.

    chart : Mx1 column vector.
      A chart buffer to reuse.
    Returns
    -------

    chart : Mx1 column vector (type depends on `kind`).
       The inside chart. Type depends on potentials type, i.e.
       for inside potentials this will be the probability paths
       reaching this vertex.
    """
    new_potentials = _get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.inside(graph, new_potentials, chart)


def outside(Hypergraph graph, potentials, inside_chart,
            kind=LogViterbi, chart=None):
    r"""
    Compute the outside values for potentials.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    potentials : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    inside_chart : :py:class:`Chart`
       The associated inside chart. Compute by calling
       :py:function:`inside`.  Must be the same type as potentials.

    kind : A semiring type.
      The semiring to use. Must agree with potentials.

    chart : Mx1 column vector.
      A chart buffer to reuse.

    Returns
    ---------

    chart : Mx1 column vector (type depends on `kind`).
       The outside chart. Type depends on potentials type, i.e. for
       inside potentials this will be the probability paths reaching
       this node.

    """
    new_potentials = _get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.outside(graph, new_potentials, inside_chart, chart)



def best_path(Hypergraph graph, potentials,
              kind=LogViterbi, chart=None):
    r"""
    Find the best path through a hypergraph for a given set of potentials.

    Formally gives
    :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    potentials : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    kind : A semiring type.
      The semiring to use. Must agree with potentials.

    chart : Mx1 column vector.
      A chart buffer to reuse.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """
    new_potentials = _get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.viterbi(graph, new_potentials, chart)

def marginals(Hypergraph graph, potentials,
              inside_chart=None,
              outside_chart=None,
              kind=LogViterbi):
    r"""
    Compute marginals for hypergraph and potentials.

    Parameters
    -----------
    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials of the hypergraph.

    Returns
    --------
    marginals : :py:class:`Marginals`
       The node and edge marginals associated with these potentials.
    """
    my_inside = inside_chart
    if my_inside is None:
        my_inside = inside(graph, potentials, kind=kind)

    my_outside = outside_chart
    if my_outside is None:
        my_outside = outside(graph, potentials, inside_chart=my_inside, kind=kind)

    new_potentials = _get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.compute_marginals(graph, new_potentials,
                                                 my_inside, my_outside)


def project(Hypergraph graph, hyperedge_filter):
    """
    Project a graph based on a set of boolean potentials.

    Edges with value 0 are pruned, edges with value
    1 are pruned if they are no longer in a path.

    Parameters
    -----------
    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    hyperedge_filter : Nx1 int8 column vector.
        The pruning filter to use.

    Returns
    --------
    original_graph : Hypergraph
       The original hypergraph.

    projection : N`xN sparse int8 matrix
       Matrix mapping original edges indices to projected indices.

    projected_graph : Hypergraph
       The new projected hypergraph with :math:`|{\cal E}| = N'`.


    """
    new_filt = <_BoolPotentials> _get_potentials(graph, hyperedge_filter,
                                                 Bool.Potentials)
    cdef const CHypergraphMap *projection = \
        cproject_hypergraph(graph.thisptr,
                            deref(new_filt.thisptr))
    return convert_hypergraph_map(projection, graph, None)
