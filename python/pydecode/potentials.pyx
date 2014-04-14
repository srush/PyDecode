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
    hyperedges :math:`{\cal E}`, and a root node.

    Attributes
    -----------

    edges : iterator of :py:class:`Edge`s
      The edge set :math:`{\cal E}`. In topological order.

    root : :py:class:`Vertex`
      A specialized node in :math:`{\cal V}`.

    vertices : iterator of :py:class:`Vertex`s
      The node set :math:`{\cal V}`. In topological order.
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
        s = "Hypergraph: Edges: %s Nodes: %s" % (len(self.edges),
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
        add_node(edges=[], label=None)

        Add a node to the hypergraph.

        Parameters
        ------------

        edges :
           An iterator where each of the items is of the form
           ([v_2, v_3..], label)  where v_2 ... are :py:class:`Vertex`s and
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
    A vertex in a hypergraph.

    Formally :math:`v \in {\cal V}` associated with a :py:class:`Hypergraph`.

    Attributes
    -------------

    edges : iterator of :py:class:`Edge`s

       The edges with :math:`v` as head node.

       :math:`\{e \in {\cal E} : h(e) = v \}`

    is_terminal : bool
       Is the node :math:`v` in terminal node.

    label : any
        A piece of data associated with the edge.

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

    property edges:
        def __get__(self):
            return convert_edges(self.nodeptr.edges(), self.graph)

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
    A hyperedge associated with hypergraph.

    Hyperedge :math:`e \in {\cal E}` associated with a :py:class:`Hypergraph`.

    Attributes
    -----------

    head : :py:class:`Vertex`
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
    A valid hyperpath through the hypergraph.

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

    property nodes:
        def __get__(self):
            return _LazyVertices(self.graph).init(self.thisptr.nodes())


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
    A map between two hypergraphs.


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
        TODO: fill in
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

#from libhypergraph cimport *
#import libhypergraph as py_hypergraph


cdef class Potentials:
    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

cdef class Chart:
    r"""
    A dynamic programming chart associated with a hypergraph.

    Chart :math:`S^{|{\cal V}|}` associated with a hypergraph (V, E)
    and semiring S.

    Acts as a vector::
       >> print chart[node]
    """
    pass


cdef class Marginals:
    r"""
    Marginal values with a hypergraph and potentials.

    Marginal values :math:`S^{|{\cal E} \times {\cal V}|}` associated
    with a hypergraph ({\cal V}, {\cal E}) and semiring S.

    Acts as a dictionary::
       >> print marginals[edge]
       >> print marginals[node]
    """
    pass


############# This is the templated semiring part. ##############



cdef class ViterbiPotentials(Potentials):
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
        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Viterbi_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        potentials,
                                        _Viterbi_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = _Viterbi_to_cpp(bias)

        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Viterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = _Viterbi_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Viterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _Viterbi:
    cdef _Viterbi init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = _Viterbi()
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
        return _Viterbi().init(Viterbi_zero())

    @staticmethod
    def one():
        return _Viterbi().init(Viterbi_one())

    def __add__(_Viterbi self, _Viterbi other):
        return _Viterbi().init(Viterbi_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Viterbi self, _Viterbi other):
        return _Viterbi().init(Viterbi_times(self.thisval,
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
        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _LogViterbi_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        potentials,
                                        _LogViterbi_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = _LogViterbi_to_cpp(bias)

        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _LogViterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = _LogViterbi_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _LogViterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _LogViterbi:
    cdef _LogViterbi init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = _LogViterbi()
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
        return _LogViterbi().init(LogViterbi_zero())

    @staticmethod
    def one():
        return _LogViterbi().init(LogViterbi_one())

    def __add__(_LogViterbi self, _LogViterbi other):
        return _LogViterbi().init(LogViterbi_add(self.thisval,
                                                  other.thisval))

    def __mul__(_LogViterbi self, _LogViterbi other):
        return _LogViterbi().init(LogViterbi_times(self.thisval,
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
        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Inside_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        potentials,
                                        _Inside_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = _Inside_to_cpp(bias)

        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Inside_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = _Inside_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Inside_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _Inside:
    cdef _Inside init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = _Inside()
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
        return _Inside().init(Inside_zero())

    @staticmethod
    def one():
        return _Inside().init(Inside_one())

    def __add__(_Inside self, _Inside other):
        return _Inside().init(Inside_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Inside self, _Inside other):
        return _Inside().init(Inside_times(self.thisval,
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




cdef class BoolPotentials(Potentials):
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
        cdef vector[bool] potentials = \
            vector[bool](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Bool_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        potentials,
                                        _Bool_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef bool my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = _Bool_to_cpp(bias)

        cdef vector[bool] potentials = \
            vector[bool](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Bool_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef bool my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = _Bool_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[bool] potentials = \
            vector[bool](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Bool_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _Bool:
    cdef _Bool init(self, bool val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(bool val):
        created = _Bool()
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
        return _Bool().init(Bool_zero())

    @staticmethod
    def one():
        return _Bool().init(Bool_one())

    def __add__(_Bool self, _Bool other):
        return _Bool().init(Bool_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Bool self, _Bool other):
        return _Bool().init(Bool_times(self.thisval,
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




cdef class SparseVectorPotentials(Potentials):
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
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _SparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _SparseVector_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = _SparseVector_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _SparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = _SparseVector_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _SparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _SparseVector:
    cdef _SparseVector init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = _SparseVector()
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
        return _SparseVector().init(SparseVector_zero())

    @staticmethod
    def one():
        return _SparseVector().init(SparseVector_one())

    def __add__(_SparseVector self, _SparseVector other):
        return _SparseVector().init(SparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(_SparseVector self, _SparseVector other):
        return _SparseVector().init(SparseVector_times(self.thisval,
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
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _MinSparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _MinSparseVector_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MinSparseVector_one()
        else:
            my_bias = _MinSparseVector_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _MinSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MinSparseVector_one()
        else:
            my_bias = _MinSparseVector_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MinSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _MinSparseVector:
    cdef _MinSparseVector init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = _MinSparseVector()
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
        return _MinSparseVector().init(MinSparseVector_zero())

    @staticmethod
    def one():
        return _MinSparseVector().init(MinSparseVector_one())

    def __add__(_MinSparseVector self, _MinSparseVector other):
        return _MinSparseVector().init(MinSparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(_MinSparseVector self, _MinSparseVector other):
        return _MinSparseVector().init(MinSparseVector_times(self.thisval,
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
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _MaxSparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _MaxSparseVector_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MaxSparseVector_one()
        else:
            my_bias = _MaxSparseVector_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _MaxSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MaxSparseVector_one()
        else:
            my_bias = _MaxSparseVector_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MaxSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _MaxSparseVector:
    cdef _MaxSparseVector init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = _MaxSparseVector()
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
        return _MaxSparseVector().init(MaxSparseVector_zero())

    @staticmethod
    def one():
        return _MaxSparseVector().init(MaxSparseVector_one())

    def __add__(_MaxSparseVector self, _MaxSparseVector other):
        return _MaxSparseVector().init(MaxSparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(_MaxSparseVector self, _MaxSparseVector other):
        return _MaxSparseVector().init(MaxSparseVector_times(self.thisval,
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
        cdef vector[int] potentials = \
            vector[int](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Counting_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        potentials,
                                        _Counting_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef int my_bias
        if bias is None:
            my_bias = Counting_one()
        else:
            my_bias = _Counting_to_cpp(bias)

        cdef vector[int] potentials = \
            vector[int](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Counting_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef int my_bias
        if bias is None:
            my_bias = Counting_one()
        else:
            my_bias = _Counting_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[int] potentials = \
            vector[int](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Counting_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

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


cdef class _Counting:
    cdef _Counting init(self, int val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(int val):
        created = _Counting()
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
        return _Counting().init(Counting_zero())

    @staticmethod
    def one():
        return _Counting().init(Counting_one())

    def __add__(_Counting self, _Counting other):
        return _Counting().init(Counting_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Counting self, _Counting other):
        return _Counting().init(Counting_times(self.thisval,
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




####### Methods that use specific potential ########


cdef class BackPointers:
    """
    The back pointers generated by the Viterbi algorithm.

    Acts as a map::
       >> print bp[node]

    Gives the best back edge for node.

    Attributes
    -----------

    path: Hyperpath
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
    Compute inside chart values for the given potentials.

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
    Compute the outside scores for the hypergraph.

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
    (hypergraph, potentials) : :py:class:`Hypergraph`, :py:class:`Potentials`
       The new hypergraphs and potentials.
    """
    return potentials.kind.prune_hypergraph(graph, potentials, thres)


def compute_marginals(Hypergraph graph, Potentials potentials):
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
    Prune a graph based on a set of boolean potentials.

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
