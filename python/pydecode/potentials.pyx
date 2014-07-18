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

    cdef init(self, vector[int] nodes):
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

    property heads:
        def __get__(self):
            return np.array(self.thisptr.heads())

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
            return self.graph.labeling[self]

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
        return self.edgeptr

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

    # def _removed(self):
    #     return (self.id == -1)

cdef convert_edges(vector[int] edges,
                   Hypergraph graph):
    return [Edge().init(edge, graph) for edge in edges]

cdef convert_nodes(vector[int] nodes,
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
            return _LazyEdges(self.graph).init(self.thisptr.edges())

    property vertices:
        def __get__(self):
            return _LazyVertices(self.graph).init(self.thisptr.nodes())

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

cimport cython

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
    cdef int node
    cdef vector[int] new_nodes = \
        projection.domain_graph().nodes()

    for i in range(labels.size()):
        node = projection.map_node(new_nodes[i])
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

    # item_matrix : Sparse matrix |I| x |V|
    #    Maps vertices of the hypergraph to item indicators.

    # output_matrix : Sparse matrix |O| x |E|
    #    Maps hyperedges of the hypergraph to sparse output counts.


cdef class DynamicProgram:
    def __init__(self, hypergraph,
                 item_indices,
                 output_indices,
                 items, outputs):
        self._hypergraph = hypergraph
        self._items = items
        self._output_shape = outputs.shape
        self._outputs = outputs
        self._item_indices = np.array(item_indices)
        self._output_indices = np.array(output_indices)

        self._item_matrix = None
        self._output_matrix = None
        self._active_outputs = None
        self._active_output_indices = None
        self._active_dict = None

        self._active_output_elements = None

    def _make_output_matrix(self):
        # assert self._construct_output, \
        #     "Output size not specified."
        data = []
        outputs = []
        ind = [0]
        for index in self.output_indices:
            if index != -1:
                data.append(1)
                outputs.append(index)
            ind.append(len(data))
        return scipy.sparse.csc_matrix(
            (data, outputs, ind),
            shape=(np.max(self.outputs) + 1,
                   len(self.hypergraph.edges)),
            dtype=np.uint8)

    def _make_active_output_matrix(self):
        data = []
        outputs = []
        ind = [0]
        for index in self.active_output_indices:
            if index != -1:
                data.append(1)
                outputs.append(index)
            ind.append(len(data))
        return scipy.sparse.csc_matrix(
            (data, outputs, ind),
            shape=(max(self.active_output_indices)+1,
                   len(self.hypergraph.edges)),
            dtype=np.uint8)

    def _make_item_matrix(self):
        return scipy.sparse.csc_matrix(
            ([1] * len(self.item_indices),
             self.item_indices,
             range(len(self.item_indices) + 1)),
            shape=(np.max(self.items) + 1,
                   len(self.hypergraph.nodes)),
            dtype=np.uint8)

    def _make_active_outputs(self):
        d = {}
        counter = 0
        aoi = []
        ao = []
        cdef int index
        for index in self._output_indices:
            if index == -1:
                aoi.append(-1)
            elif index in d:
                aoi.append(d[index])
            else:
                aoi.append(counter)
                ao.append(index)
                d[index] = counter
                counter += 1
        self._active_output_indices = np.array(aoi)
        self._active_outputs = np.array(ao)

    property hypergraph:
        def __get__(self):
            return self._hypergraph

    property items:
        def __get__(self):
            return self._items

    property outputs:
        def __get__(self):
            return self._outputs

    property output_shape:
        def __get__(self):
            return self._output_shape

    property item_indices:
        def __get__(self):
            return self._item_indices

    property output_indices:
        def __get__(self):
            return self._output_indices

    property output_matrix:
        def __get__(self):
            if self._output_matrix is None:
                self._output_matrix = self._make_output_matrix()
            return self._output_matrix

    property item_matrix:
        def __get__(self):
            if self._item_matrix is None:
                self._item_matrix = self._make_item_matrix()
            return self._item_matrix

    property active_outputs:
        def __get__(self):
            if self._active_outputs is None:
                self._make_active_outputs()
            return self._active_outputs

    property active_output_matrix:
        def __get__(self):
            if self._active_output_matrix is None:
                self._active_output_matrix = self._make_active_output_matrix()
            return self._active_output_matrix

    property active_output_indices:
        def __get__(self):
            if self._active_output_indices is None:
                self._make_active_outputs()
            return self._active_output_indices

    property active_output_elements:
        def __get__(self):
            if self._active_output_indices is None:
                self._active_output_elements = \
                    np.array(np.unravel_index(self.active_outputs,
                                              self.outputs.shape)).T
            return self._active_output_elements



cdef class _ChartEdge:
    pass

NODE_NULL = -1
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
    items : Encoder I -> {0...|I|}
       Encodes elements of the item set I as integers.

    outputs : Encoder O -> {0...|O|}
       Encodes elements of the output set O as integers.
    """

    def __init__(self,
                 items,
                 outputs,
                 unstrict=False,
                 expected_size=None,
                 lattice=False):
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
        self._no_tail = set[long]()
        self._strict = not unstrict
        self._hg_ptr = new CHypergraph(lattice)

        self._size = np.max(items) + 1
        self.items = items
        self.outputs = outputs
        self._chart = new vector[int](self._size, NODE_NULL)
        self._nindices = []

        # Output structures.
        self._output_size = np.max(outputs) + 1
        self._construct_output = self._output_size is not None

        self._indices = []

        if expected_size:
            self._hg_ptr.set_expected_size(self._size,
                                           expected_size[0],
                                           expected_size[1])
            self._max_arity = expected_size[1]

        self._out = np.array([0], dtype=np.int64)

        self._lattice = lattice

    @cython.boundscheck(False)
    cpdef init(self, long [:] indices):
        cdef int i
        for i in range(indices.shape[0]):
            deref(self._chart)[indices[i]] = \
                self._hg_ptr.add_terminal_node()

            self._nindices.append(indices[i])
            self._no_tail.insert(indices[i])

    @cython.boundscheck(False)
    cpdef set(self,
              long index,
              long [:] tails1,
              long [:] tails2=None,
              long [:] tails3=None,
              long [:] out=None):

        deref(self._chart)[index] = self._hg_ptr.start_node()

        blank = (out is None)
        cdef vector[int] tails
        if tails2 is not None:
            assert tails1.shape[0] == tails2.shape[0], \
                "Tails 1 shape: %s Tails 2 shape: %s"% (tails1.shape[0], tails2.shape[0])

        if tails3 is not None:
            assert tails1.shape[0] == tails3.shape[0], \
                "Tails 1 shape: %s Tails 3 shape: %s"% (tails1.shape[0], tails3.shape[0])

        if out is not None:
            assert tails1.shape[0] == out.shape[0], \
                "Tails 1 shape: %s Out shape: %s"% (tails1.shape[0], out.shape[0])

        cdef int i, j

        # assert blank_edge or tails1.shape[0] == tails2.shape[0]
        for j in range(tails1.shape[0]):
            tails.clear()

            if tails1[j] == NODE_NULL: continue
            tails.push_back(deref(self._chart)[tails1[j]])
            if tails.back() == NODE_NULL:
                raise Exception(
                    "Item %s not found for tail 1."%(tails1[j],))

            if tails2 is not None:
                if tails2[j] == NODE_NULL: continue
                tails.push_back(deref(self._chart)[tails2[j]])

                if tails.back() == NODE_NULL:
                    raise Exception(
                        "Item %s not found for tail 2."%(tails2[j],))

            if tails3 is not None:
                if tails3[j] == NODE_NULL: continue
                tails.push_back(deref(self._chart)[tails3[j]])
                if tails.back() == NODE_NULL:
                    raise Exception(
                        "Item %s not found for tail 3."%(tails3[j],))
            self._hg_ptr.add_edge(tails)

            if self._no_tail.find(tails1[j]) != self._no_tail.end():
                self._no_tail.erase(tails1[j])
            if tails2 is not None and self._no_tail.find(tails2[j]) != self._no_tail.end():
                self._no_tail.erase(tails2[j])
            if tails3 is not None and self._no_tail.find(tails3[j]) != self._no_tail.end():
                self._no_tail.erase(tails3[j])

            if self._construct_output:
                if not blank and out[j] != -1:
                    self._indices.append(out[j])
                else:
                    self._indices.append(-1)

        result = self._hg_ptr.end_node()
        if not result:
            if self._strict:
                raise Exception("No tail items found for item %s."%(index,))
            deref(self._chart)[index] = NODE_NULL
        else:
            self._last = index
            self._no_tail.insert(index)
            self._nindices.append(index)

    def finish(self, reconstruct=False):
        """
        Finish chart construction.

        Returns
        -------
        hypergraph :
           The hypergraph representing the dynamic program.

        """
        if self._done:
            raise Exception("Hypergraph not constructed.")
        if self._no_tail.size() != 1:
            raise Exception("Hypergraph has multiple vertices that are not connected: %s."%(self._no_tail,))
        self._done = True
        self._hg_ptr.finish(reconstruct)

        hypergraph = Hypergraph(self._lattice)
        hypergraph.init(self._hg_ptr,
                        Labeling(hypergraph, None))
        return DynamicProgram(hypergraph,
                              self._nindices,
                              self._indices,
                              self.items,
                              self.outputs)
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


cdef class BeamChartBinaryVectorPotential:
    cdef init(self, CBeamChartBinaryVectorPotential *chart, Hypergraph graph):
        self.thisptr = chart
        self.graph = graph
        return self

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    def path(self, int result):
        if self.thisptr.get_path(result) == NULL:
            return None
        return Path().init(self.thisptr.get_path(result),
                           self.graph)

    def __getitem__(self, Vertex vertex):
        cdef vector[CBeamHypBinaryVectorPotential *] beam = \
                    self.thisptr.get_beam(vertex.nodeptr)
        data = []
        i = 0
        for p in beam:
            data.append((_BinaryVector_from_cpp(p.sig),
                         p.current_score,
                         p.future_score))
        return data


    property exact:
        def __get__(self):
            return self.thisptr.exact


# def beam_search_BinaryVector(Hypergraph graph,
#                 _LogViterbiPotentials potentials,
#                 BinaryVectorPotentials constraints,
#                 outside,
#                 double lower_bound,
#                 groups,
#                 group_limits,
#                 int num_groups=-1,
#                 bool recombine=True,
#                            bool cube_pruning = False):
#     r"""

#     Parameters
#     -----------
#     graph : Hypergraph

#     potentials : LogViterbiPotentials
#        The potentials on each hyperedge.

#     constraints : BinaryVectorPotentials
#        The constraints (bitset) at each hyperedge.

#     lower_bound : double

#     outside : LogViterbiChart
#         The outside scores.

#     groups : size of vertex list
#        The group for each vertex.

#     group_limits :
#        The size limit for each group.

#     num_groups :
#         The total number of groups.
#     """
#     if num_groups == -1:
#         ngroups = max(groups) + 1
#     else:
#         ngroups = num_groups
#     cdef vector[int] cgroups = groups
#     cdef vector[int] cgroup_limits = group_limits

#     cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
#                                                     cgroups,
#                                                     cgroup_limits,
#                                                     ngroups)
#     # cgroups.resize(graph.nodes_size())
#     # cdef vector[int] cgroup_limits
#     # cgroups.resize(graph.nodes_size())

#     # for i, group in enumerate(groups):
#     #     cgroups[i] = group


#     cdef CBeamChartBinaryVectorPotential *chart
#     if cube_pruning:
#         chart = ccube_pruningBinaryVectorPotential(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     else:
#         chart = cbeam_searchBinaryVectorPotential(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     return BeamChartBinaryVectorPotential().init(chart, graph)


cdef class BeamChartAlphabetPotential:
    cdef init(self, CBeamChartAlphabetPotential *chart, Hypergraph graph):
        self.thisptr = chart
        self.graph = graph
        return self

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    def path(self, int result):
        if self.thisptr.get_path(result) == NULL:
            return None
        return Path().init(self.thisptr.get_path(result),
                           self.graph)

    def __getitem__(self, Vertex vertex):
        cdef vector[CBeamHypAlphabetPotential *] beam = \
                    self.thisptr.get_beam(vertex.nodeptr)
        data = []
        i = 0
        for p in beam:
            data.append((_Alphabet_from_cpp(p.sig),
                         p.current_score,
                         p.future_score))
        return data


    property exact:
        def __get__(self):
            return self.thisptr.exact


# def beam_search_Alphabet(Hypergraph graph,
#                 _LogViterbiPotentials potentials,
#                 AlphabetPotentials constraints,
#                 outside,
#                 double lower_bound,
#                 groups,
#                 group_limits,
#                 int num_groups=-1,
#                 bool recombine=True,
#                            bool cube_pruning = False):
#     r"""

#     Parameters
#     -----------
#     graph : Hypergraph

#     potentials : LogViterbiPotentials
#        The potentials on each hyperedge.

#     constraints : BinaryVectorPotentials
#        The constraints (bitset) at each hyperedge.

#     lower_bound : double

#     outside : LogViterbiChart
#         The outside scores.

#     groups : size of vertex list
#        The group for each vertex.

#     group_limits :
#        The size limit for each group.

#     num_groups :
#         The total number of groups.
#     """
#     if num_groups == -1:
#         ngroups = max(groups) + 1
#     else:
#         ngroups = num_groups
#     cdef vector[int] cgroups = groups
#     cdef vector[int] cgroup_limits = group_limits

#     cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
#                                                     cgroups,
#                                                     cgroup_limits,
#                                                     ngroups)
#     # cgroups.resize(graph.nodes_size())
#     # cdef vector[int] cgroup_limits
#     # cgroups.resize(graph.nodes_size())

#     # for i, group in enumerate(groups):
#     #     cgroups[i] = group


#     cdef CBeamChartAlphabetPotential *chart
#     if cube_pruning:
#         chart = ccube_pruningAlphabetPotential(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     else:
#         chart = cbeam_searchAlphabetPotential(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     return BeamChartAlphabetPotential().init(chart, graph)


cdef class BeamChartLogViterbiPotential:
    cdef init(self, CBeamChartLogViterbiPotential *chart, Hypergraph graph):
        self.thisptr = chart
        self.graph = graph
        return self

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    def path(self, int result):
        if self.thisptr.get_path(result) == NULL:
            return None
        return Path().init(self.thisptr.get_path(result),
                           self.graph)

    def __getitem__(self, Vertex vertex):
        cdef vector[CBeamHypLogViterbiPotential *] beam = \
                    self.thisptr.get_beam(vertex.nodeptr)
        data = []
        i = 0
        for p in beam:
            data.append((_LogViterbi_from_cpp(p.sig),
                         p.current_score,
                         p.future_score))
        return data


    property exact:
        def __get__(self):
            return self.thisptr.exact


# def beam_search_LogViterbi(Hypergraph graph,
#                 _LogViterbiPotentials potentials,
#                 LogViterbiPotentials constraints,
#                 outside,
#                 double lower_bound,
#                 groups,
#                 group_limits,
#                 int num_groups=-1,
#                 bool recombine=True,
#                            bool cube_pruning = False):
#     r"""

#     Parameters
#     -----------
#     graph : Hypergraph

#     potentials : LogViterbiPotentials
#        The potentials on each hyperedge.

#     constraints : BinaryVectorPotentials
#        The constraints (bitset) at each hyperedge.

#     lower_bound : double

#     outside : LogViterbiChart
#         The outside scores.

#     groups : size of vertex list
#        The group for each vertex.

#     group_limits :
#        The size limit for each group.

#     num_groups :
#         The total number of groups.
#     """
#     if num_groups == -1:
#         ngroups = max(groups) + 1
#     else:
#         ngroups = num_groups
#     cdef vector[int] cgroups = groups
#     cdef vector[int] cgroup_limits = group_limits

#     cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
#                                                     cgroups,
#                                                     cgroup_limits,
#                                                     ngroups)
#     # cgroups.resize(graph.nodes_size())
#     # cdef vector[int] cgroup_limits
#     # cgroups.resize(graph.nodes_size())

#     # for i, group in enumerate(groups):
#     #     cgroups[i] = group


#     cdef CBeamChartLogViterbiPotential *chart
#     if cube_pruning:
#         chart = ccube_pruningLogViterbiPotential(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     else:
#         chart = cbeam_searchLogViterbiPotential(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     return BeamChartLogViterbiPotential().init(chart, graph)


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
    Value = BoolValue
    
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
                char [:] chart=None,
                int [:] back_pointers=None,
                bool [:] mask=None,
                bool get_path=True):
        cdef char [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CBoolChart *in_chart = new CBoolChart(
            graph.thisptr,
            &my_chart[0])

        cdef int [:] my_back_pointers = back_pointers
        if back_pointers is None:
            my_back_pointers = np.zeros(len(graph.nodes), dtype=np.int32)
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr,
                              &my_back_pointers[0])

        viterbi_Bool(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back,
                           (<bool *> NULL) if mask is None else (<bool *>&mask[0])
                           )
        cdef CHyperpath *path
        if get_path:
            path = used_back.construct_path()
            del in_chart, used_back
            return Path().init(path, graph)
        else:
            del in_chart, used_back

    


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
    Value = ViterbiValue
    
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
                double [:] chart=None,
                int [:] back_pointers=None,
                bool [:] mask=None,
                bool get_path=True):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CViterbiChart *in_chart = new CViterbiChart(
            graph.thisptr,
            &my_chart[0])

        cdef int [:] my_back_pointers = back_pointers
        if back_pointers is None:
            my_back_pointers = np.zeros(len(graph.nodes), dtype=np.int32)
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr,
                              &my_back_pointers[0])

        viterbi_Viterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back,
                           (<bool *> NULL) if mask is None else (<bool *>&mask[0])
                           )
        cdef CHyperpath *path
        if get_path:
            path = used_back.construct_path()
            del in_chart, used_back
            return Path().init(path, graph)
        else:
            del in_chart, used_back

    


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
    Value = CountingValue
    
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
                int [:] chart=None,
                int [:] back_pointers=None,
                bool [:] mask=None,
                bool get_path=True):
        cdef int [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CCountingChart *in_chart = new CCountingChart(
            graph.thisptr,
            &my_chart[0])

        cdef int [:] my_back_pointers = back_pointers
        if back_pointers is None:
            my_back_pointers = np.zeros(len(graph.nodes), dtype=np.int32)
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr,
                              &my_back_pointers[0])

        viterbi_Counting(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back,
                           (<bool *> NULL) if mask is None else (<bool *>&mask[0])
                           )
        cdef CHyperpath *path
        if get_path:
            path = used_back.construct_path()
            del in_chart, used_back
            return Path().init(path, graph)
        else:
            del in_chart, used_back

    


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
    Value = LogViterbiValue
    
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
                double [:] chart=None,
                int [:] back_pointers=None,
                bool [:] mask=None,
                bool get_path=True):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CLogViterbiChart *in_chart = new CLogViterbiChart(
            graph.thisptr,
            &my_chart[0])

        cdef int [:] my_back_pointers = back_pointers
        if back_pointers is None:
            my_back_pointers = np.zeros(len(graph.nodes), dtype=np.int32)
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr,
                              &my_back_pointers[0])

        viterbi_LogViterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back,
                           (<bool *> NULL) if mask is None else (<bool *>&mask[0])
                           )
        cdef CHyperpath *path
        if get_path:
            path = used_back.construct_path()
            del in_chart, used_back
            return Path().init(path, graph)
        else:
            del in_chart, used_back

    


cdef class _LogProbPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = LogProb
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphLogProbPotentials *ptr):
        self.thisptr = ptr
        return self

    
    def from_array(self, double [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_LogProb(self.graph.thisptr,
                                               <double *> &X[0])
        return self

    def as_array(self):
        return _LogProbvector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    


cdef class LogProbValue:
    cdef LogProbValue init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = LogProbValue()
        created.thisval = _LogProb_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _LogProb_from_cpp(LogProb_zero())

    @staticmethod
    def one_raw():
        return _LogProb_from_cpp(LogProb_one())

    @staticmethod
    def zero():
        return LogProbValue().init(LogProb_zero())

    @staticmethod
    def one():
        return LogProbValue().init(LogProb_one())

    def __add__(LogProbValue self, LogProbValue other):
        return LogProbValue().init(LogProb_add(self.thisval,
                                                  other.thisval))

    def __mul__(LogProbValue self, LogProbValue other):
        return LogProbValue().init(LogProb_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _LogProb_from_cpp(self.thisval)


cdef double _LogProb_to_cpp(double val):
    return val


cdef _LogProb_from_cpp(double val):
    
    return val
    





cdef _LogProbvector_to_numpy(const double *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef double [:] my_view = my_array
    return np.asarray(my_view)



class LogProb:
    Potentials = _LogProbPotentials
    Value = LogProbValue
    
    @staticmethod
    def inside(Hypergraph graph,
               _LogProbPotentials potentials,
               double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CLogProbChart *in_chart = new CLogProbChart(
            graph.thisptr,
            &my_chart[0])

        inside_LogProb(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _LogProbPotentials potentials,
                double [:] inside_chart,
                double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef CLogProbChart *in_chart = new CLogProbChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CLogProbChart *out_chart = new CLogProbChart(
            graph.thisptr,
            &my_chart[0])

        outside_LogProb(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _LogProbPotentials potentials,
                          double [:] inside_chart,
                          double [:] outside_chart):

        cdef double [:] node_margs = np.zeros(len(graph.nodes))
        cdef double [:] edge_margs = np.zeros(len(graph.edges))

        cdef CLogProbChart *in_chart = new CLogProbChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CLogProbChart *out_chart = new CLogProbChart(
            graph.thisptr,
            &outside_chart[0])

        cdef CLogProbChart *node_chart = new CLogProbChart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_LogProb(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_LogProb(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const CLogProbMarginals *marginals = \


        # return (_LogProbvector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _LogProbvector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    
    


cdef class _InsidePotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = Inside
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphInsidePotentials *ptr):
        self.thisptr = ptr
        return self

    
    def from_array(self, double [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_Inside(self.graph.thisptr,
                                               <double *> &X[0])
        return self

    def as_array(self):
        return _Insidevector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    


cdef class InsideValue:
    cdef InsideValue init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = InsideValue()
        created.thisval = _Inside_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Inside_from_cpp(Inside_zero())

    @staticmethod
    def one_raw():
        return _Inside_from_cpp(Inside_one())

    @staticmethod
    def zero():
        return InsideValue().init(Inside_zero())

    @staticmethod
    def one():
        return InsideValue().init(Inside_one())

    def __add__(InsideValue self, InsideValue other):
        return InsideValue().init(Inside_add(self.thisval,
                                                  other.thisval))

    def __mul__(InsideValue self, InsideValue other):
        return InsideValue().init(Inside_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Inside_from_cpp(self.thisval)


cdef double _Inside_to_cpp(double val):
    return val


cdef _Inside_from_cpp(double val):
    
    return val
    





cdef _Insidevector_to_numpy(const double *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef double [:] my_view = my_array
    return np.asarray(my_view)



class Inside:
    Potentials = _InsidePotentials
    Value = InsideValue
    
    @staticmethod
    def inside(Hypergraph graph,
               _InsidePotentials potentials,
               double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CInsideChart *in_chart = new CInsideChart(
            graph.thisptr,
            &my_chart[0])

        inside_Inside(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _InsidePotentials potentials,
                double [:] inside_chart,
                double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef CInsideChart *in_chart = new CInsideChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CInsideChart *out_chart = new CInsideChart(
            graph.thisptr,
            &my_chart[0])

        outside_Inside(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _InsidePotentials potentials,
                          double [:] inside_chart,
                          double [:] outside_chart):

        cdef double [:] node_margs = np.zeros(len(graph.nodes))
        cdef double [:] edge_margs = np.zeros(len(graph.edges))

        cdef CInsideChart *in_chart = new CInsideChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CInsideChart *out_chart = new CInsideChart(
            graph.thisptr,
            &outside_chart[0])

        cdef CInsideChart *node_chart = new CInsideChart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_Inside(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_Inside(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const CInsideMarginals *marginals = \


        # return (_Insidevector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _Insidevector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    
    

    @staticmethod
    def viterbi(Hypergraph graph,
                _InsidePotentials potentials,
                double [:] chart=None,
                int [:] back_pointers=None,
                bool [:] mask=None,
                bool get_path=True):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CInsideChart *in_chart = new CInsideChart(
            graph.thisptr,
            &my_chart[0])

        cdef int [:] my_back_pointers = back_pointers
        if back_pointers is None:
            my_back_pointers = np.zeros(len(graph.nodes), dtype=np.int32)
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr,
                              &my_back_pointers[0])

        viterbi_Inside(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back,
                           (<bool *> NULL) if mask is None else (<bool *>&mask[0])
                           )
        cdef CHyperpath *path
        if get_path:
            path = used_back.construct_path()
            del in_chart, used_back
            return Path().init(path, graph)
        else:
            del in_chart, used_back

    


cdef class _MinMaxPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = MinMax
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphMinMaxPotentials *ptr):
        self.thisptr = ptr
        return self

    
    def from_array(self, double [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_MinMax(self.graph.thisptr,
                                               <double *> &X[0])
        return self

    def as_array(self):
        return _MinMaxvector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    


cdef class MinMaxValue:
    cdef MinMaxValue init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = MinMaxValue()
        created.thisval = _MinMax_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _MinMax_from_cpp(MinMax_zero())

    @staticmethod
    def one_raw():
        return _MinMax_from_cpp(MinMax_one())

    @staticmethod
    def zero():
        return MinMaxValue().init(MinMax_zero())

    @staticmethod
    def one():
        return MinMaxValue().init(MinMax_one())

    def __add__(MinMaxValue self, MinMaxValue other):
        return MinMaxValue().init(MinMax_add(self.thisval,
                                                  other.thisval))

    def __mul__(MinMaxValue self, MinMaxValue other):
        return MinMaxValue().init(MinMax_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _MinMax_from_cpp(self.thisval)


cdef double _MinMax_to_cpp(double val):
    return val


cdef _MinMax_from_cpp(double val):
    
    return val
    





cdef _MinMaxvector_to_numpy(const double *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef double [:] my_view = my_array
    return np.asarray(my_view)



class MinMax:
    Potentials = _MinMaxPotentials
    Value = MinMaxValue
    
    @staticmethod
    def inside(Hypergraph graph,
               _MinMaxPotentials potentials,
               double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef CMinMaxChart *in_chart = new CMinMaxChart(
            graph.thisptr,
            &my_chart[0])

        inside_MinMax(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _MinMaxPotentials potentials,
                double [:] inside_chart,
                double [:] chart=None):
        cdef double [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef CMinMaxChart *in_chart = new CMinMaxChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CMinMaxChart *out_chart = new CMinMaxChart(
            graph.thisptr,
            &my_chart[0])

        outside_MinMax(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _MinMaxPotentials potentials,
                          double [:] inside_chart,
                          double [:] outside_chart):

        cdef double [:] node_margs = np.zeros(len(graph.nodes))
        cdef double [:] edge_margs = np.zeros(len(graph.edges))

        cdef CMinMaxChart *in_chart = new CMinMaxChart(
            graph.thisptr,
            &inside_chart[0])
        cdef CMinMaxChart *out_chart = new CMinMaxChart(
            graph.thisptr,
            &outside_chart[0])

        cdef CMinMaxChart *node_chart = new CMinMaxChart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_MinMax(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_MinMax(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const CMinMaxMarginals *marginals = \


        # return (_MinMaxvector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _MinMaxvector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    
    


cdef class _AlphabetPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = Alphabet
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphAlphabetPotentials *ptr):
        self.thisptr = ptr
        return self

    


cdef class AlphabetValue:
    cdef AlphabetValue init(self, vector[int] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[int] val):
        created = AlphabetValue()
        created.thisval = _Alphabet_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Alphabet_from_cpp(Alphabet_zero())

    @staticmethod
    def one_raw():
        return _Alphabet_from_cpp(Alphabet_one())

    @staticmethod
    def zero():
        return AlphabetValue().init(Alphabet_zero())

    @staticmethod
    def one():
        return AlphabetValue().init(Alphabet_one())

    def __add__(AlphabetValue self, AlphabetValue other):
        return AlphabetValue().init(Alphabet_add(self.thisval,
                                                  other.thisval))

    def __mul__(AlphabetValue self, AlphabetValue other):
        return AlphabetValue().init(Alphabet_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Alphabet_from_cpp(self.thisval)


cdef vector[int] _Alphabet_to_cpp(vector[int] val):
    return val


cdef _Alphabet_from_cpp(vector[int] val):
    
    return val
    







class Alphabet:
    Potentials = _AlphabetPotentials
    Value = AlphabetValue
    
    


cdef class _BinaryVectorPotentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = BinaryVector
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraphBinaryVectorPotentials *ptr):
        self.thisptr = ptr
        return self

    


cdef class BinaryVectorValue:
    cdef BinaryVectorValue init(self, cbitset val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(Bitset val):
        created = BinaryVectorValue()
        created.thisval = _BinaryVector_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _BinaryVector_from_cpp(BinaryVector_zero())

    @staticmethod
    def one_raw():
        return _BinaryVector_from_cpp(BinaryVector_one())

    @staticmethod
    def zero():
        return BinaryVectorValue().init(BinaryVector_zero())

    @staticmethod
    def one():
        return BinaryVectorValue().init(BinaryVector_one())

    def __add__(BinaryVectorValue self, BinaryVectorValue other):
        return BinaryVectorValue().init(BinaryVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(BinaryVectorValue self, BinaryVectorValue other):
        return BinaryVectorValue().init(BinaryVector_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _BinaryVector_from_cpp(self.thisval)


cdef cbitset _BinaryVector_to_cpp(Bitset val):
    return <cbitset>val.data


cdef _BinaryVector_from_cpp(cbitset val):
    
    return Bitset().init(val)
    







class BinaryVector:
    Potentials = _BinaryVectorPotentials
    Value = BinaryVectorValue
    
    



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

def get_potentials(graph, potentials, kind=_LogViterbiPotentials):
    # if potentials.size != len(graph.edges):
    #     raise ValueError("Potentials must match hypergraph hyperedges size: %s != %s"%(potentials.size, len(graph.edges)))
    return kind(graph).from_array(potentials)

@cython.boundscheck(False)
cpdef map_potentials(dp, out_potentials):
    cdef np.ndarray raveled = out_potentials.ravel()
    cdef np.ndarray potentials = raveled[dp.output_indices]
    return potentials

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
    new_filt = <_BoolPotentials> get_potentials(graph, hyperedge_filter,
                                                 Bool.Potentials)
    cdef const CHypergraphMap *projection = \
        cproject_hypergraph(graph.thisptr,
                            deref(new_filt.thisptr))
    return convert_hypergraph_map(projection, graph, None)
