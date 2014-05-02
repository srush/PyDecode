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

    cdef init(self, vector[const CHyperedge *] edges):
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

    def __str__(self):
        s = "Hypergraph: Edges: %s Vertices: %s" % (len(self.edges),
                                                 len(self.nodes)) + "\n"
        s += "Root %s" % (self.root.id) + "\n"
        for edge in self.edges:
            s += " %s %s \n" % (edge.id, self.label(edge))
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
        cdef const CHyperedge *edgeptr
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
                self.edge_labels.append((Edge().init(edgeptr, self.graph), t))
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
        return (self.nodeptr.id() == -1)

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
        return self.id

    def __dealloc__(self):
        pass

    cdef Edge init(self, const CHyperedge *ptr, Hypergraph graph):
        self.edgeptr = ptr
        self.graph = graph
        return self

    def __repr__(self):
        return "EDGE:%d" % (self.edgeptr.id())

    property tail:
        def __get__(self):
            return convert_nodes(self.edgeptr.tail_nodes(), self.graph)

    property head:
        def __get__(self):
            return Vertex().init(self.edgeptr.head_node(), self.graph)

    property label:
        def __get__(self):
            return self.graph.labeling[self]

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

        cdef vector[const CHyperedge *] cedges
        self.graph = graph
        edges.sort(key=lambda e: e.id)
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
        else:
            self.range_graph = range_graph
            assert self.range_graph.thisptr.id() == \
                self.thisptr.range_graph().id()
            self.domain_graph = self._build_domain_hypergraph()
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
        cdef const CHyperedge *edge
        cdef const CHypernode *node
        if isinstance(obj, Edge):
            edge = self.thisptr.map((<Edge>obj).edgeptr)
            # assert edge.id() >= 0
            # assert edge.id() == self.range_graph.edges[edge.id()].id
            if edge != NULL and edge.id() >= 0:
                return self.range_graph.edges[edge.id()]
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
        cdef vector[const CHyperedge*] old_edges = \
            projection.domain_graph().edges()

        cdef const CHypernode *node
        cdef const CHyperedge *edge

        for i in range(old_nodes.size()):
            node = self.thisptr.map(old_nodes[i])
            if node != NULL and node.id() >= 0:
                node_labels[node.id()] = \
                    self.domain_graph.labeling.node_labels[i]

        if self.domain_graph.labeling.edge_labels:
            for i in range(old_edges.size()):
                edge = self.thisptr.map(old_edges[i])
                if edge != NULL and edge.id() >= 0:
                    edge_labels[edge.id()] = \
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
        cdef const CHyperedge *edge

        for i in range(graph.nodes().size()):
            node = self.thisptr.map(graph.nodes()[i])
            if node != NULL and node.id() >= 0:
                node_labels[i] = \
                    self.range_graph.labeling.node_labels[node.id()]

        if self.range_graph.labeling.edge_labels:
            for i in range(graph.edges().size()):
                edge = self.thisptr.map(graph.edges()[i])
                if edge != NULL and edge.id() >= 0:
                    edge_labels[i] = \
                        self.range_graph.labeling.edge_labels[edge.id()]

        cdef Hypergraph h = Hypergraph()
        return h.init(graph, Labeling(h, node_labels, edge_labels))

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector


from pydecode.potentials cimport *

cdef class Bitset:
    """
    Bitset


    """

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


def beam_search_BinaryVector(Hypergraph graph,
                LogViterbiPotentials potentials,
                BinaryVectorPotentials constraints,
                LogViterbiChart outside,
                double lower_bound,
                groups,
                group_limits,
                int num_groups):
    r"""

    Parameters
    -----------
    graph : Hypergraph

    potentials : LogViterbiPotentials
       The potentials on each hyperedge.

    constraints : BinaryVectorPotentials
       The constraints (bitset) at each hyperedge.

    lower_bound : double

    groups : size of vetex list
       The group for each vertex.

    group_limits :
       The size limit for each group.

    num_groups :
        The total number of groups.
    """

    cdef vector[int] cgroups = groups
    cdef vector[int] cgroup_limits = group_limits
    cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
                                                    cgroups,
                                                    cgroup_limits,
                                                    num_groups)
    # cgroups.resize(graph.nodes_size())
    # cdef vector[int] cgroup_limits
    # cgroups.resize(graph.nodes_size())

    # for i, group in enumerate(groups):
    #     cgroups[i] = group


    cdef CBeamChartBinaryVectorPotential *chart = \
        cbeam_searchBinaryVectorPotential(graph.thisptr,
                     deref(potentials.thisptr),
                     deref(constraints.thisptr),
                     deref(outside.chart),
                     lower_bound,
                     deref(beam_groups))
    return BeamChartBinaryVectorPotential().init(chart, graph)


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


def beam_search_Alphabet(Hypergraph graph,
                LogViterbiPotentials potentials,
                AlphabetPotentials constraints,
                LogViterbiChart outside,
                double lower_bound,
                groups,
                group_limits,
                int num_groups):
    r"""

    Parameters
    -----------
    graph : Hypergraph

    potentials : LogViterbiPotentials
       The potentials on each hyperedge.

    constraints : BinaryVectorPotentials
       The constraints (bitset) at each hyperedge.

    lower_bound : double

    groups : size of vetex list
       The group for each vertex.

    group_limits :
       The size limit for each group.

    num_groups :
        The total number of groups.
    """

    cdef vector[int] cgroups = groups
    cdef vector[int] cgroup_limits = group_limits
    cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
                                                    cgroups,
                                                    cgroup_limits,
                                                    num_groups)
    # cgroups.resize(graph.nodes_size())
    # cdef vector[int] cgroup_limits
    # cgroups.resize(graph.nodes_size())

    # for i, group in enumerate(groups):
    #     cgroups[i] = group


    cdef CBeamChartAlphabetPotential *chart = \
        cbeam_searchAlphabetPotential(graph.thisptr,
                     deref(potentials.thisptr),
                     deref(constraints.thisptr),
                     deref(outside.chart),
                     lower_bound,
                     deref(beam_groups))
    return BeamChartAlphabetPotential().init(chart, graph)


#cython: embedsignature=True

##
## DO NOT MODIFY THIS GENERATED FILE.
##

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

#from libhypergraph cimport *
#import libhypergraph as py_hypergraph


cdef class Potentials:
    r"""
    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Attributes
    ------------

    bias : value

    """

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    def times(self, Potentials other):
        r"""
        """
        pass

    def clone(self):
        pass


    def project(self, Hypergraph graph, HypergraphMap projection):
        pass

    property bias:
        def __get__(self):
           return None

    def from_array(self, X, bias=None):
        pass


    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        pass


cdef class Chart:
    r"""
    A dynamic programming chart associated with a hypergraph.

    Chart :math:`S^{|{\cal V}|}` associated with a hypergraph (V, E)
    and semiring S.

    Acts as a vector ::

       >> print chart[node]
    """
    pass


cdef class Marginals:
    r"""
    Marginal values with a hypergraph and potentials.

    Marginal values :math:`S^{|{\cal E} \times {\cal V}|}` associated
    with a hypergraph ({\cal V}, {\cal E}) and semiring S.

    Acts as a dictionary ::

       >> print marginals[edge]
       >> print marginals[node]

    """
    pass


############# This is the templated semiring part. ##############



cdef class ViterbiPotentials(Potentials):
    r"""
    Real-valued max probability potentials.
Uses the operations :math:`(+, *) = (\max, *)`.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Viterbi
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, ViterbiPotentials other):
        cdef CHypergraphViterbiPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return ViterbiPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return ViterbiPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return ViterbiPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphViterbiPotentials *ptr = \
            cmake_projected_potentials_Viterbi(self.thisptr,
                                                  projection.thisptr)
        return ViterbiPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _Viterbi_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef double my_bias
    #     if bias is None:
    #         my_bias = Viterbi_one()
    #     else:
    #         my_bias = _Viterbi_to_cpp(bias)

    #     cdef vector[double] potentials = \
    #          vector[double](self.hypergraph.thisptr.edges().size(),
    #          Viterbi_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Viterbi_zero()
    #         potentials[i] = _Viterbi_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Viterbi(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _Viterbi_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        potentials,
                                        _Viterbi_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias = self._bias(bias)

        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _Viterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Viterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef double _bias(self, bias):
        if bias is None:
            return Viterbi_one()
        else:
            return _Viterbi_to_cpp(bias)

    
    def from_array(self, double [:] X,
                   bias=None):
        cdef double my_bias = self._bias(bias)
        cdef int s = self.hypergraph.thisptr.edges().size()

        cdef vector[double] *vec= \
            new vector[double]()
        vec.assign(&X[0], (&X[0]) + s)

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        vec, my_bias, False)
        return self

    def as_array(self):
        return _Viterbivector_to_numpy(self.thisptr.potentials())
    


    cdef init(self, CHypergraphViterbiPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Viterbi_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _Viterbi_from_cpp(self.thisptr.dot(deref(path.thisptr)))


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
    



cdef class ViterbiChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = Viterbi
        self.chart = NULL
        if graph is not None:
            self.chart = new CViterbiChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _Viterbi_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    
    def as_array(self):
        return _Viterbivector_to_numpy(self.chart.chart())
    


cdef _Viterbivector_to_numpy(const vector[double] &vec):
    cdef view.array my_array = \
        view.array(shape=(vec.size(),),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec.data()
    cdef double [:] my_view = my_array
    return np.asarray(my_view)


cdef class _ViterbiMarginals(Marginals):
    cdef const CViterbiMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CViterbiMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Viterbi_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _Viterbi_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Viterbi marginal values." +
                "Passed %s." % obj)

    
    def as_array(self):
        return _Viterbivector_to_numpy(self.thisptr.node_marginals())
    


    

    def threshold(self, double semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)
    


class Viterbi:
    Chart = ViterbiChart
    Marginals = _ViterbiMarginals
    #Semi = _Viterbi
    Potentials = ViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               ViterbiPotentials potentials):
        cdef ViterbiChart chart = ViterbiChart()
        chart.chart = inside_Viterbi(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                ViterbiPotentials potentials,
                ViterbiChart inside_chart):
        cdef ViterbiChart out_chart = ViterbiChart()
        out_chart.chart = outside_Viterbi(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def viterbi(Hypergraph graph,
                ViterbiPotentials potentials,
                ViterbiChart chart=None):
        cdef CViterbiChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CViterbiChart(graph.thisptr)
        viterbi_Viterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          ViterbiPotentials potentials):
        cdef const CViterbiMarginals *marginals = \
            Viterbi_compute(graph.thisptr, potentials.thisptr)
        return _ViterbiMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         ViterbiPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class LogViterbiPotentials(Potentials):
    r"""
    Real-valued max log-probability potentials.
Uses the operations :math:`(+, *) = (\max, *)`.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = LogViterbi
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, LogViterbiPotentials other):
        cdef CHypergraphLogViterbiPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return LogViterbiPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return LogViterbiPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphLogViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return LogViterbiPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphLogViterbiPotentials *ptr = \
            cmake_projected_potentials_LogViterbi(self.thisptr,
                                                  projection.thisptr)
        return LogViterbiPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _LogViterbi_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef double my_bias
    #     if bias is None:
    #         my_bias = LogViterbi_one()
    #     else:
    #         my_bias = _LogViterbi_to_cpp(bias)

    #     cdef vector[double] potentials = \
    #          vector[double](self.hypergraph.thisptr.edges().size(),
    #          LogViterbi_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = LogViterbi_zero()
    #         potentials[i] = _LogViterbi_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_LogViterbi(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _LogViterbi_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        potentials,
                                        _LogViterbi_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias = self._bias(bias)

        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _LogViterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _LogViterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef double _bias(self, bias):
        if bias is None:
            return LogViterbi_one()
        else:
            return _LogViterbi_to_cpp(bias)

    
    def from_array(self, double [:] X,
                   bias=None):
        cdef double my_bias = self._bias(bias)
        cdef int s = self.hypergraph.thisptr.edges().size()

        cdef vector[double] *vec= \
            new vector[double]()
        vec.assign(&X[0], (&X[0]) + s)

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        vec, my_bias, False)
        return self

    def as_array(self):
        return _LogViterbivector_to_numpy(self.thisptr.potentials())
    


    cdef init(self, CHypergraphLogViterbiPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _LogViterbi_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _LogViterbi_from_cpp(self.thisptr.dot(deref(path.thisptr)))


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
    



cdef class LogViterbiChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = LogViterbi
        self.chart = NULL
        if graph is not None:
            self.chart = new CLogViterbiChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _LogViterbi_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    
    def as_array(self):
        return _LogViterbivector_to_numpy(self.chart.chart())
    


cdef _LogViterbivector_to_numpy(const vector[double] &vec):
    cdef view.array my_array = \
        view.array(shape=(vec.size(),),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec.data()
    cdef double [:] my_view = my_array
    return np.asarray(my_view)


cdef class _LogViterbiMarginals(Marginals):
    cdef const CLogViterbiMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CLogViterbiMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _LogViterbi_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _LogViterbi_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have LogViterbi marginal values." +
                "Passed %s." % obj)

    
    def as_array(self):
        return _LogViterbivector_to_numpy(self.thisptr.node_marginals())
    


    

    def threshold(self, double semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)
    


class LogViterbi:
    Chart = LogViterbiChart
    Marginals = _LogViterbiMarginals
    #Semi = _LogViterbi
    Potentials = LogViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               LogViterbiPotentials potentials):
        cdef LogViterbiChart chart = LogViterbiChart()
        chart.chart = inside_LogViterbi(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                LogViterbiPotentials potentials,
                LogViterbiChart inside_chart):
        cdef LogViterbiChart out_chart = LogViterbiChart()
        out_chart.chart = outside_LogViterbi(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def viterbi(Hypergraph graph,
                LogViterbiPotentials potentials,
                LogViterbiChart chart=None):
        cdef CLogViterbiChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CLogViterbiChart(graph.thisptr)
        viterbi_LogViterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          LogViterbiPotentials potentials):
        cdef const CLogViterbiMarginals *marginals = \
            LogViterbi_compute(graph.thisptr, potentials.thisptr)
        return _LogViterbiMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         LogViterbiPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class InsidePotentials(Potentials):
    r"""
    Real-valued probability potentials.
Uses the operations :math:`(+, *) = (+, *)`.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Inside
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, InsidePotentials other):
        cdef CHypergraphInsidePotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return InsidePotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return InsidePotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphInsidePotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return InsidePotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphInsidePotentials *ptr = \
            cmake_projected_potentials_Inside(self.thisptr,
                                                  projection.thisptr)
        return InsidePotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _Inside_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef double my_bias
    #     if bias is None:
    #         my_bias = Inside_one()
    #     else:
    #         my_bias = _Inside_to_cpp(bias)

    #     cdef vector[double] potentials = \
    #          vector[double](self.hypergraph.thisptr.edges().size(),
    #          Inside_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Inside_zero()
    #         potentials[i] = _Inside_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Inside(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _Inside_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        potentials,
                                        _Inside_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias = self._bias(bias)

        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _Inside_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Inside_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef double _bias(self, bias):
        if bias is None:
            return Inside_one()
        else:
            return _Inside_to_cpp(bias)

    
    def from_array(self, double [:] X,
                   bias=None):
        cdef double my_bias = self._bias(bias)
        cdef int s = self.hypergraph.thisptr.edges().size()

        cdef vector[double] *vec= \
            new vector[double]()
        vec.assign(&X[0], (&X[0]) + s)

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        vec, my_bias, False)
        return self

    def as_array(self):
        return _Insidevector_to_numpy(self.thisptr.potentials())
    


    cdef init(self, CHypergraphInsidePotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Inside_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _Inside_from_cpp(self.thisptr.dot(deref(path.thisptr)))


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
    



cdef class InsideChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = Inside
        self.chart = NULL
        if graph is not None:
            self.chart = new CInsideChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _Inside_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    
    def as_array(self):
        return _Insidevector_to_numpy(self.chart.chart())
    


cdef _Insidevector_to_numpy(const vector[double] &vec):
    cdef view.array my_array = \
        view.array(shape=(vec.size(),),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec.data()
    cdef double [:] my_view = my_array
    return np.asarray(my_view)


cdef class _InsideMarginals(Marginals):
    cdef const CInsideMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CInsideMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Inside_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _Inside_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Inside marginal values." +
                "Passed %s." % obj)

    
    def as_array(self):
        return _Insidevector_to_numpy(self.thisptr.node_marginals())
    


    

    def threshold(self, double semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)
    


class Inside:
    Chart = InsideChart
    Marginals = _InsideMarginals
    #Semi = _Inside
    Potentials = InsidePotentials

    @staticmethod
    def inside(Hypergraph graph,
               InsidePotentials potentials):
        cdef InsideChart chart = InsideChart()
        chart.chart = inside_Inside(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                InsidePotentials potentials,
                InsideChart inside_chart):
        cdef InsideChart out_chart = InsideChart()
        out_chart.chart = outside_Inside(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def viterbi(Hypergraph graph,
                InsidePotentials potentials,
                InsideChart chart=None):
        cdef CInsideChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CInsideChart(graph.thisptr)
        viterbi_Inside(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          InsidePotentials potentials):
        cdef const CInsideMarginals *marginals = \
            Inside_compute(graph.thisptr, potentials.thisptr)
        return _InsideMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         InsidePotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class MinMaxPotentials(Potentials):
    r"""
    Real-valued min value potentials.
Uses the operations :math:`(+, *) = (\min, \max)`.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = MinMax
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, MinMaxPotentials other):
        cdef CHypergraphMinMaxPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return MinMaxPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return MinMaxPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMinMaxPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return MinMaxPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMinMaxPotentials *ptr = \
            cmake_projected_potentials_MinMax(self.thisptr,
                                                  projection.thisptr)
        return MinMaxPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _MinMax_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef double my_bias
    #     if bias is None:
    #         my_bias = MinMax_one()
    #     else:
    #         my_bias = _MinMax_to_cpp(bias)

    #     cdef vector[double] potentials = \
    #          vector[double](self.hypergraph.thisptr.edges().size(),
    #          MinMax_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = MinMax_zero()
    #         potentials[i] = _MinMax_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_MinMax(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _MinMax_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_MinMax(self.hypergraph.thisptr,
                                        potentials,
                                        _MinMax_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias = self._bias(bias)

        cdef vector[double] *potentials = \
            new vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _MinMax_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinMax(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MinMax_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinMax(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef double _bias(self, bias):
        if bias is None:
            return MinMax_one()
        else:
            return _MinMax_to_cpp(bias)

    
    def from_array(self, double [:] X,
                   bias=None):
        cdef double my_bias = self._bias(bias)
        cdef int s = self.hypergraph.thisptr.edges().size()

        cdef vector[double] *vec= \
            new vector[double]()
        vec.assign(&X[0], (&X[0]) + s)

        self.thisptr =  \
            cmake_potentials_MinMax(self.hypergraph.thisptr,
                                        vec, my_bias, False)
        return self

    def as_array(self):
        return _MinMaxvector_to_numpy(self.thisptr.potentials())
    


    cdef init(self, CHypergraphMinMaxPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _MinMax_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _MinMax_from_cpp(self.thisptr.dot(deref(path.thisptr)))


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
    



cdef class MinMaxChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = MinMax
        self.chart = NULL
        if graph is not None:
            self.chart = new CMinMaxChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _MinMax_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    
    def as_array(self):
        return _MinMaxvector_to_numpy(self.chart.chart())
    


cdef _MinMaxvector_to_numpy(const vector[double] &vec):
    cdef view.array my_array = \
        view.array(shape=(vec.size(),),
                   itemsize=sizeof(double),
                   format="d",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec.data()
    cdef double [:] my_view = my_array
    return np.asarray(my_view)


cdef class _MinMaxMarginals(Marginals):
    cdef const CMinMaxMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CMinMaxMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _MinMax_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _MinMax_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have MinMax marginal values." +
                "Passed %s." % obj)

    
    def as_array(self):
        return _MinMaxvector_to_numpy(self.thisptr.node_marginals())
    


    


class MinMax:
    Chart = MinMaxChart
    Marginals = _MinMaxMarginals
    #Semi = _MinMax
    Potentials = MinMaxPotentials

    @staticmethod
    def inside(Hypergraph graph,
               MinMaxPotentials potentials):
        cdef MinMaxChart chart = MinMaxChart()
        chart.chart = inside_MinMax(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                MinMaxPotentials potentials,
                MinMaxChart inside_chart):
        cdef MinMaxChart out_chart = MinMaxChart()
        out_chart.chart = outside_MinMax(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          MinMaxPotentials potentials):
        cdef const CMinMaxMarginals *marginals = \
            MinMax_compute(graph.thisptr, potentials.thisptr)
        return _MinMaxMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         MinMaxPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class SparseVectorPotentials(Potentials):
    r"""
    Sparse-vector valued weights.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = SparseVector
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, SparseVectorPotentials other):
        cdef CHypergraphSparseVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return SparseVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return SparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return SparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphSparseVectorPotentials *ptr = \
            cmake_projected_potentials_SparseVector(self.thisptr,
                                                  projection.thisptr)
        return SparseVectorPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _SparseVector_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef vector[pair[int, int]] my_bias
    #     if bias is None:
    #         my_bias = SparseVector_one()
    #     else:
    #         my_bias = _SparseVector_to_cpp(bias)

    #     cdef vector[vector[pair[int, int]]] potentials = \
    #          vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
    #          SparseVector_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = SparseVector_zero()
    #         potentials[i] = _SparseVector_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_SparseVector(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] *potentials = \
            new vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _SparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _SparseVector_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias = self._bias(bias)

        cdef vector[vector[pair[int, int]]] *potentials = \
            new vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _SparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _SparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef vector[pair[int, int]] _bias(self, bias):
        if bias is None:
            return SparseVector_one()
        else:
            return _SparseVector_to_cpp(bias)

    


    cdef init(self, CHypergraphSparseVectorPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _SparseVector_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _SparseVector_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class SparseVectorValue:
    cdef SparseVectorValue init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = SparseVectorValue()
        created.thisval = _SparseVector_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _SparseVector_from_cpp(SparseVector_zero())

    @staticmethod
    def one_raw():
        return _SparseVector_from_cpp(SparseVector_one())

    @staticmethod
    def zero():
        return SparseVectorValue().init(SparseVector_zero())

    @staticmethod
    def one():
        return SparseVectorValue().init(SparseVector_one())

    def __add__(SparseVectorValue self, SparseVectorValue other):
        return SparseVectorValue().init(SparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(SparseVectorValue self, SparseVectorValue other):
        return SparseVectorValue().init(SparseVector_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _SparseVector_from_cpp(self.thisval)


cdef vector[pair[int, int]] _SparseVector_to_cpp(vector[pair[int, int]] val):
    return val


cdef _SparseVector_from_cpp(vector[pair[int, int]] val):
    
    return val
    



cdef class SparseVectorChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = SparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CSparseVectorChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _SparseVector_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    



cdef class _SparseVectorMarginals(Marginals):
    cdef const CSparseVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CSparseVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _SparseVector_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _SparseVector_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have SparseVector marginal values." +
                "Passed %s." % obj)

    


    


class SparseVector:
    Chart = SparseVectorChart
    Marginals = _SparseVectorMarginals
    #Semi = _SparseVector
    Potentials = SparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               SparseVectorPotentials potentials):
        cdef SparseVectorChart chart = SparseVectorChart()
        chart.chart = inside_SparseVector(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                SparseVectorPotentials potentials,
                SparseVectorChart inside_chart):
        cdef SparseVectorChart out_chart = SparseVectorChart()
        out_chart.chart = outside_SparseVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          SparseVectorPotentials potentials):
        cdef const CSparseVectorMarginals *marginals = \
            SparseVector_compute(graph.thisptr, potentials.thisptr)
        return _SparseVectorMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         SparseVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class AlphabetPotentials(Potentials):
    r"""
    Alphabet valued weights.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Alphabet
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, AlphabetPotentials other):
        cdef CHypergraphAlphabetPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return AlphabetPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return AlphabetPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphAlphabetPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return AlphabetPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphAlphabetPotentials *ptr = \
            cmake_projected_potentials_Alphabet(self.thisptr,
                                                  projection.thisptr)
        return AlphabetPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _Alphabet_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef vector[int] my_bias
    #     if bias is None:
    #         my_bias = Alphabet_one()
    #     else:
    #         my_bias = _Alphabet_to_cpp(bias)

    #     cdef vector[vector[int]] potentials = \
    #          vector[vector[int]](self.hypergraph.thisptr.edges().size(),
    #          Alphabet_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Alphabet_zero()
    #         potentials[i] = _Alphabet_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Alphabet(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[int]] *potentials = \
            new vector[vector[int]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _Alphabet_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Alphabet(self.hypergraph.thisptr,
                                        potentials,
                                        _Alphabet_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[int] my_bias = self._bias(bias)

        cdef vector[vector[int]] *potentials = \
            new vector[vector[int]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _Alphabet_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Alphabet(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[int] my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[int]] potentials = \
            vector[vector[int]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Alphabet_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Alphabet(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef vector[int] _bias(self, bias):
        if bias is None:
            return Alphabet_one()
        else:
            return _Alphabet_to_cpp(bias)

    


    cdef init(self, CHypergraphAlphabetPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Alphabet_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _Alphabet_from_cpp(self.thisptr.dot(deref(path.thisptr)))


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
    



cdef class AlphabetChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = Alphabet
        self.chart = NULL
        if graph is not None:
            self.chart = new CAlphabetChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _Alphabet_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    



cdef class _AlphabetMarginals(Marginals):
    cdef const CAlphabetMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CAlphabetMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Alphabet_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _Alphabet_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Alphabet marginal values." +
                "Passed %s." % obj)

    


    


class Alphabet:
    Chart = AlphabetChart
    Marginals = _AlphabetMarginals
    #Semi = _Alphabet
    Potentials = AlphabetPotentials

    @staticmethod
    def inside(Hypergraph graph,
               AlphabetPotentials potentials):
        cdef AlphabetChart chart = AlphabetChart()
        chart.chart = inside_Alphabet(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                AlphabetPotentials potentials,
                AlphabetChart inside_chart):
        cdef AlphabetChart out_chart = AlphabetChart()
        out_chart.chart = outside_Alphabet(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          AlphabetPotentials potentials):
        cdef const CAlphabetMarginals *marginals = \
            Alphabet_compute(graph.thisptr, potentials.thisptr)
        return _AlphabetMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         AlphabetPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class MinSparseVectorPotentials(Potentials):
    r"""
    

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = MinSparseVector
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, MinSparseVectorPotentials other):
        cdef CHypergraphMinSparseVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return MinSparseVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return MinSparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMinSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return MinSparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMinSparseVectorPotentials *ptr = \
            cmake_projected_potentials_MinSparseVector(self.thisptr,
                                                  projection.thisptr)
        return MinSparseVectorPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _MinSparseVector_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef vector[pair[int, int]] my_bias
    #     if bias is None:
    #         my_bias = MinSparseVector_one()
    #     else:
    #         my_bias = _MinSparseVector_to_cpp(bias)

    #     cdef vector[vector[pair[int, int]]] potentials = \
    #          vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
    #          MinSparseVector_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = MinSparseVector_zero()
    #         potentials[i] = _MinSparseVector_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] *potentials = \
            new vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _MinSparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _MinSparseVector_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias = self._bias(bias)

        cdef vector[vector[pair[int, int]]] *potentials = \
            new vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _MinSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MinSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef vector[pair[int, int]] _bias(self, bias):
        if bias is None:
            return MinSparseVector_one()
        else:
            return _MinSparseVector_to_cpp(bias)

    


    cdef init(self, CHypergraphMinSparseVectorPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _MinSparseVector_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _MinSparseVector_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class MinSparseVectorValue:
    cdef MinSparseVectorValue init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = MinSparseVectorValue()
        created.thisval = _MinSparseVector_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _MinSparseVector_from_cpp(MinSparseVector_zero())

    @staticmethod
    def one_raw():
        return _MinSparseVector_from_cpp(MinSparseVector_one())

    @staticmethod
    def zero():
        return MinSparseVectorValue().init(MinSparseVector_zero())

    @staticmethod
    def one():
        return MinSparseVectorValue().init(MinSparseVector_one())

    def __add__(MinSparseVectorValue self, MinSparseVectorValue other):
        return MinSparseVectorValue().init(MinSparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(MinSparseVectorValue self, MinSparseVectorValue other):
        return MinSparseVectorValue().init(MinSparseVector_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _MinSparseVector_from_cpp(self.thisval)


cdef vector[pair[int, int]] _MinSparseVector_to_cpp(vector[pair[int, int]] val):
    return val


cdef _MinSparseVector_from_cpp(vector[pair[int, int]] val):
    
    return val
    



cdef class MinSparseVectorChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = MinSparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CMinSparseVectorChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _MinSparseVector_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    



cdef class _MinSparseVectorMarginals(Marginals):
    cdef const CMinSparseVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CMinSparseVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _MinSparseVector_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _MinSparseVector_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have MinSparseVector marginal values." +
                "Passed %s." % obj)

    


    


class MinSparseVector:
    Chart = MinSparseVectorChart
    Marginals = _MinSparseVectorMarginals
    #Semi = _MinSparseVector
    Potentials = MinSparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               MinSparseVectorPotentials potentials):
        cdef MinSparseVectorChart chart = MinSparseVectorChart()
        chart.chart = inside_MinSparseVector(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                MinSparseVectorPotentials potentials,
                MinSparseVectorChart inside_chart):
        cdef MinSparseVectorChart out_chart = MinSparseVectorChart()
        out_chart.chart = outside_MinSparseVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          MinSparseVectorPotentials potentials):
        cdef const CMinSparseVectorMarginals *marginals = \
            MinSparseVector_compute(graph.thisptr, potentials.thisptr)
        return _MinSparseVectorMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         MinSparseVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class MaxSparseVectorPotentials(Potentials):
    r"""
    

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = MaxSparseVector
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, MaxSparseVectorPotentials other):
        cdef CHypergraphMaxSparseVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return MaxSparseVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return MaxSparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMaxSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return MaxSparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMaxSparseVectorPotentials *ptr = \
            cmake_projected_potentials_MaxSparseVector(self.thisptr,
                                                  projection.thisptr)
        return MaxSparseVectorPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _MaxSparseVector_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef vector[pair[int, int]] my_bias
    #     if bias is None:
    #         my_bias = MaxSparseVector_one()
    #     else:
    #         my_bias = _MaxSparseVector_to_cpp(bias)

    #     cdef vector[vector[pair[int, int]]] potentials = \
    #          vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
    #          MaxSparseVector_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = MaxSparseVector_zero()
    #         potentials[i] = _MaxSparseVector_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] *potentials = \
            new vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _MaxSparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _MaxSparseVector_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias = self._bias(bias)

        cdef vector[vector[pair[int, int]]] *potentials = \
            new vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _MaxSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MaxSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef vector[pair[int, int]] _bias(self, bias):
        if bias is None:
            return MaxSparseVector_one()
        else:
            return _MaxSparseVector_to_cpp(bias)

    


    cdef init(self, CHypergraphMaxSparseVectorPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _MaxSparseVector_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _MaxSparseVector_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class MaxSparseVectorValue:
    cdef MaxSparseVectorValue init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = MaxSparseVectorValue()
        created.thisval = _MaxSparseVector_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _MaxSparseVector_from_cpp(MaxSparseVector_zero())

    @staticmethod
    def one_raw():
        return _MaxSparseVector_from_cpp(MaxSparseVector_one())

    @staticmethod
    def zero():
        return MaxSparseVectorValue().init(MaxSparseVector_zero())

    @staticmethod
    def one():
        return MaxSparseVectorValue().init(MaxSparseVector_one())

    def __add__(MaxSparseVectorValue self, MaxSparseVectorValue other):
        return MaxSparseVectorValue().init(MaxSparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(MaxSparseVectorValue self, MaxSparseVectorValue other):
        return MaxSparseVectorValue().init(MaxSparseVector_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _MaxSparseVector_from_cpp(self.thisval)


cdef vector[pair[int, int]] _MaxSparseVector_to_cpp(vector[pair[int, int]] val):
    return val


cdef _MaxSparseVector_from_cpp(vector[pair[int, int]] val):
    
    return val
    



cdef class MaxSparseVectorChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = MaxSparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CMaxSparseVectorChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _MaxSparseVector_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    



cdef class _MaxSparseVectorMarginals(Marginals):
    cdef const CMaxSparseVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CMaxSparseVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _MaxSparseVector_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _MaxSparseVector_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have MaxSparseVector marginal values." +
                "Passed %s." % obj)

    


    


class MaxSparseVector:
    Chart = MaxSparseVectorChart
    Marginals = _MaxSparseVectorMarginals
    #Semi = _MaxSparseVector
    Potentials = MaxSparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               MaxSparseVectorPotentials potentials):
        cdef MaxSparseVectorChart chart = MaxSparseVectorChart()
        chart.chart = inside_MaxSparseVector(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                MaxSparseVectorPotentials potentials,
                MaxSparseVectorChart inside_chart):
        cdef MaxSparseVectorChart out_chart = MaxSparseVectorChart()
        out_chart.chart = outside_MaxSparseVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          MaxSparseVectorPotentials potentials):
        cdef const CMaxSparseVectorMarginals *marginals = \
            MaxSparseVector_compute(graph.thisptr, potentials.thisptr)
        return _MaxSparseVectorMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         MaxSparseVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class CountingPotentials(Potentials):
    r"""
    Natural-valued counting potentials.
Uses the operations :math:`(+, *) = (+, *)`.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Counting
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, CountingPotentials other):
        cdef CHypergraphCountingPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return CountingPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return CountingPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphCountingPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return CountingPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphCountingPotentials *ptr = \
            cmake_projected_potentials_Counting(self.thisptr,
                                                  projection.thisptr)
        return CountingPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _Counting_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef int my_bias
    #     if bias is None:
    #         my_bias = Counting_one()
    #     else:
    #         my_bias = _Counting_to_cpp(bias)

    #     cdef vector[int] potentials = \
    #          vector[int](self.hypergraph.thisptr.edges().size(),
    #          Counting_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Counting_zero()
    #         potentials[i] = _Counting_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Counting(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[int] *potentials = \
            new vector[int](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _Counting_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        potentials,
                                        _Counting_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef int my_bias = self._bias(bias)

        cdef vector[int] *potentials = \
            new vector[int](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _Counting_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef int my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[int] potentials = \
            vector[int](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Counting_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef int _bias(self, bias):
        if bias is None:
            return Counting_one()
        else:
            return _Counting_to_cpp(bias)

    
    def from_array(self, int [:] X,
                   bias=None):
        cdef int my_bias = self._bias(bias)
        cdef int s = self.hypergraph.thisptr.edges().size()

        cdef vector[int] *vec= \
            new vector[int]()
        vec.assign(&X[0], (&X[0]) + s)

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        vec, my_bias, False)
        return self

    def as_array(self):
        return _Countingvector_to_numpy(self.thisptr.potentials())
    


    cdef init(self, CHypergraphCountingPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Counting_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _Counting_from_cpp(self.thisptr.dot(deref(path.thisptr)))


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
    



cdef class CountingChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = Counting
        self.chart = NULL
        if graph is not None:
            self.chart = new CCountingChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _Counting_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    
    def as_array(self):
        return _Countingvector_to_numpy(self.chart.chart())
    


cdef _Countingvector_to_numpy(const vector[int] &vec):
    cdef view.array my_array = \
        view.array(shape=(vec.size(),),
                   itemsize=sizeof(int),
                   format="i",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec.data()
    cdef int [:] my_view = my_array
    return np.asarray(my_view)


cdef class _CountingMarginals(Marginals):
    cdef const CCountingMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CCountingMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Counting_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _Counting_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Counting marginal values." +
                "Passed %s." % obj)

    
    def as_array(self):
        return _Countingvector_to_numpy(self.thisptr.node_marginals())
    


    

    def threshold(self, int semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)
    


class Counting:
    Chart = CountingChart
    Marginals = _CountingMarginals
    #Semi = _Counting
    Potentials = CountingPotentials

    @staticmethod
    def inside(Hypergraph graph,
               CountingPotentials potentials):
        cdef CountingChart chart = CountingChart()
        chart.chart = inside_Counting(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                CountingPotentials potentials,
                CountingChart inside_chart):
        cdef CountingChart out_chart = CountingChart()
        out_chart.chart = outside_Counting(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def viterbi(Hypergraph graph,
                CountingPotentials potentials,
                CountingChart chart=None):
        cdef CCountingChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CCountingChart(graph.thisptr)
        viterbi_Counting(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          CountingPotentials potentials):
        cdef const CCountingMarginals *marginals = \
            Counting_compute(graph.thisptr, potentials.thisptr)
        return _CountingMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         CountingPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class BoolPotentials(Potentials):
    r"""
    Boolean-valued logical potentials.
Uses the operations :math:`(+, *) = (\land, \lor)`.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Bool
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, BoolPotentials other):
        cdef CHypergraphBoolPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return BoolPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return BoolPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphBoolPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return BoolPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphBoolPotentials *ptr = \
            cmake_projected_potentials_Bool(self.thisptr,
                                                  projection.thisptr)
        return BoolPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _Bool_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef bool my_bias
    #     if bias is None:
    #         my_bias = Bool_one()
    #     else:
    #         my_bias = _Bool_to_cpp(bias)

    #     cdef vector[bool] potentials = \
    #          vector[bool](self.hypergraph.thisptr.edges().size(),
    #          Bool_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Bool_zero()
    #         potentials[i] = _Bool_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Bool(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[bool] *potentials = \
            new vector[bool](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _Bool_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        potentials,
                                        _Bool_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef bool my_bias = self._bias(bias)

        cdef vector[bool] *potentials = \
            new vector[bool](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _Bool_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef bool my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[bool] potentials = \
            vector[bool](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Bool_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef bool _bias(self, bias):
        if bias is None:
            return Bool_one()
        else:
            return _Bool_to_cpp(bias)

    


    cdef init(self, CHypergraphBoolPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Bool_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _Bool_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class BoolValue:
    cdef BoolValue init(self, bool val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(bool val):
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


cdef bool _Bool_to_cpp(bool val):
    return val


cdef _Bool_from_cpp(bool val):
    
    return val
    



cdef class BoolChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = Bool
        self.chart = NULL
        if graph is not None:
            self.chart = new CBoolChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _Bool_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    



cdef class _BoolMarginals(Marginals):
    cdef const CBoolMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CBoolMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Bool_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _Bool_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Bool marginal values." +
                "Passed %s." % obj)

    


    

    def threshold(self, bool semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)
    


class Bool:
    Chart = BoolChart
    Marginals = _BoolMarginals
    #Semi = _Bool
    Potentials = BoolPotentials

    @staticmethod
    def inside(Hypergraph graph,
               BoolPotentials potentials):
        cdef BoolChart chart = BoolChart()
        chart.chart = inside_Bool(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                BoolPotentials potentials,
                BoolChart inside_chart):
        cdef BoolChart out_chart = BoolChart()
        out_chart.chart = outside_Bool(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def viterbi(Hypergraph graph,
                BoolPotentials potentials,
                BoolChart chart=None):
        cdef CBoolChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CBoolChart(graph.thisptr)
        viterbi_Bool(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          BoolPotentials potentials):
        cdef const CBoolMarginals *marginals = \
            Bool_compute(graph.thisptr, potentials.thisptr)
        return _BoolMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         BoolPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class BinaryVectorPotentials(Potentials):
    r"""
    Binary vector potentials.

    Potentials associated with the edges of a hypergraph.

    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::

       >> print potentials[edge]

    """

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = BinaryVector
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def times(self, BinaryVectorPotentials other):
        cdef CHypergraphBinaryVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return BinaryVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return BinaryVectorPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphBinaryVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return BinaryVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphBinaryVectorPotentials *ptr = \
            cmake_projected_potentials_BinaryVector(self.thisptr,
                                                  projection.thisptr)
        return BinaryVectorPotentials(graph).init(ptr, projection)

    property bias:
        def __get__(self):
            return _BinaryVector_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef cbitset my_bias
    #     if bias is None:
    #         my_bias = BinaryVector_one()
    #     else:
    #         my_bias = _BinaryVector_to_cpp(bias)

    #     cdef vector[cbitset] potentials = \
    #          vector[cbitset](self.hypergraph.thisptr.edges().size(),
    #          BinaryVector_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = BinaryVector_zero()
    #         potentials[i] = _BinaryVector_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_BinaryVector(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[cbitset] *potentials = \
            new vector[cbitset](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            deref(potentials)[i] = _BinaryVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_BinaryVector(self.hypergraph.thisptr,
                                        potentials,
                                        _BinaryVector_to_cpp(other_potentials.bias),
                                        False)

        return self

    def from_vector(self, in_vec, bias=None):
        cdef cbitset my_bias = self._bias(bias)

        cdef vector[cbitset] *potentials = \
            new vector[cbitset](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            deref(potentials)[i] = _BinaryVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_BinaryVector(self.hypergraph.thisptr,
                                        potentials,
                                        my_bias,
                                        False)
        return self

    def from_map(self, in_map, bias=None):
        cdef cbitset my_bias = self._bias(bias)
        cdef c_map.map[int, int] map_potentials
        cdef vector[cbitset] potentials = \
            vector[cbitset](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _BinaryVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_BinaryVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials,
                                        my_bias)
        return self

    cdef cbitset _bias(self, bias):
        if bias is None:
            return BinaryVector_one()
        else:
            return _BinaryVector_to_cpp(bias)

    


    cdef init(self, CHypergraphBinaryVectorPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _BinaryVector_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _BinaryVector_from_cpp(self.thisptr.dot(deref(path.thisptr)))


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
    



cdef class BinaryVectorChart(Chart):
    def __init__(self, Hypergraph graph=None):
        self.kind = BinaryVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CBinaryVectorChart(graph.thisptr)

    def __getitem__(self, Vertex node):
        return _BinaryVector_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

    



cdef class _BinaryVectorMarginals(Marginals):
    cdef const CBinaryVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CBinaryVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _BinaryVector_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Vertex):
            return _BinaryVector_from_cpp(
                self.thisptr.marginal((<Vertex>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have BinaryVector marginal values." +
                "Passed %s." % obj)

    


    


class BinaryVector:
    Chart = BinaryVectorChart
    Marginals = _BinaryVectorMarginals
    #Semi = _BinaryVector
    Potentials = BinaryVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               BinaryVectorPotentials potentials):
        cdef BinaryVectorChart chart = BinaryVectorChart()
        chart.chart = inside_BinaryVector(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                BinaryVectorPotentials potentials,
                BinaryVectorChart inside_chart):
        cdef BinaryVectorChart out_chart = BinaryVectorChart()
        out_chart.chart = outside_BinaryVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          BinaryVectorPotentials potentials):
        cdef const CBinaryVectorMarginals *marginals = \
            BinaryVector_compute(graph.thisptr, potentials.thisptr)
        return _BinaryVectorMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         BinaryVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




####### Methods that use specific potential ########


cdef class BackPointers:
    """
    The back pointers generated by the Viterbi algorithm.

    Acts as a map ::

       >> print bp[node]

    Gives the best back edge for node.

    Attributes
    -----------

    path : Hyperpath
       The best hyperpath from the root.
    """

    cdef BackPointers init(self, CBackPointers *ptr,
                           Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    property path:
        def __get__(self):
            cdef CHyperpath *path = self.thisptr.construct_path()
            return Path().init(path, self.graph)

    def __getitem__(self, Vertex node):
        return Edge().init(self.thisptr.get(node.nodeptr), self.graph)

    # def __dealloc__(self):
    #     del self.thisptr
    #     self.thisptr = NULL


def inside(Hypergraph graph, Potentials potentials):
    r"""
    Compute the inside values for potentials.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    potentials : :py:class:`Potentials`
      The potentials :math:`\theta` to use for inside computations.

    Returns
    -------

    chart : :py:class:`Chart`
       The inside chart. Type depends on potentials type, i.e.
       for inside potentials this will be the probability paths
       reaching this node.
    """
    return potentials.kind.inside(graph, potentials)



def outside(Hypergraph graph, Potentials potentials, Chart inside_chart):
    r"""
    Compute the outside values for potentials.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials :math:`\theta` to use for outside computations.

    inside_chart : :py:class:`Chart`
       The associated inside chart. Compute by calling
       :py:function:`inside`.  Must be the same type as potentials.

    Returns
    ---------

    chart : :py:class:`Chart`
       The outside chart. Type depends on potentials type, i.e. for
       inside potentials this will be the probability paths reaching
       this node.

    """
    return potentials.kind.outside(graph, potentials, inside_chart)


def best_path(Hypergraph graph, Potentials potentials, Chart chart=None):
    r"""
    Find the best path through a hypergraph for a given set of potentials.

    Formally gives
    :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    potentials : :py:class:`Potentials`
      The potentials :math:`\theta` of the hypergraph.

    chart : :py:class:`Chart`
      A chart to be reused. For memory efficiency.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """
    bp = potentials.kind.viterbi(graph, potentials, chart)
    return bp.path


def prune_hypergraph(Hypergraph graph, Potentials potentials, thres):
    return prune(graph, potentials, thres)

def prune(Hypergraph graph, Potentials potentials, thres):
    r"""
    Prune hyperedges with low marginal score from the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials of the hypergraph.

    thres : Potential
       The potential threshold to use.

    Returns
    --------
    map : :py:class:`HypergraphMap`
       Map from original graph to new graph.
    """
    return potentials.kind.prune_hypergraph(graph, potentials, thres)


def compute_marginals(Hypergraph graph, Potentials potentials):
    return marginals(graph, potentials)

def marginals(Hypergraph graph, Potentials potentials):
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
    return potentials.kind.compute_marginals(graph, potentials)

inside_values = inside
outside_values = outside


def make_pruning_projections(Hypergraph graph, BoolPotentials filt):
    """
    DEPRECATED

    Use project.
    """
    cdef const CHypergraphMap *projection = \
        cproject_hypergraph(graph.thisptr,
                            deref(filt.thisptr))
    return HypergraphMap().init(projection, graph, None)


def project(Hypergraph graph, BoolPotentials filter):
    """
    Project a graph based on a set of boolean potentials.

    Edges with value 0 are pruned, edges with value
    1 are pruned if they are no longer in a path.

    Parameters
    -----------
    graph : Hypergraph

    filter : BoolPotentials
        The pruning filter to use.

    Returns
    --------
    map : HypergraphMap
        A map from the original graph to the new graph produced.
    """
    return make_pruning_projections(graph, filter)


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
    return HypergraphMap().init(hypergraph_map, graph, None)


def pairwise_dot(SparseVectorPotentials potentials,
                 vec,
                 LogViterbiPotentials weights):
    """
    DEPRECATED.

    1) Take the dot produce of each element of potentials and vector.
    2) Add this value to each element of weights.

    Parameters
    -----------
    potentials: SparseVectorPotentials
        A vector associated with each edge.

    vec: list-like
        A float vector

    weights: LogViterbiPotentials
        A mutable set of potentials.
    """
    cdef vector[double] rvec = vector[double]()
    for i in vec:
        rvec.push_back(<double>i)
    cpairwise_dot(deref(potentials.thisptr), rvec, weights.thisptr)

def extend_hypergraph_by_count(Hypergraph graph,
                               CountingPotentials potentials,
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

    return HypergraphMap().init(projection, None, graph)


# def valid_binary_vectors(Bitset lhs, Bitset rhs):
#     return cvalid_binary_vectors(lhs.data, rhs.data)




# cdef class NodeUpdates:
#     def __cinit__(self, Hypergraph graph,
#                   SparseVectorPotentials potentials):
#         self.graph = graph
#         self.children = \
#             children_sparse(graph.thisptr,
#                             deref(potentials.thisptr))

#     def update(self, set[int] updates):
#         cdef set[int] *up = \
#             updated_nodes(self.graph.thisptr,
#                           deref(self.children),
#                           updates)
#         return deref(up)
