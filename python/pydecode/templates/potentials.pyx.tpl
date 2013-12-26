#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
cimport libcpp.map as c_map
from libcpp.pair cimport pair
from libcpp cimport bool

from wrap cimport *
from hypergraph cimport *
import hypergraph as py_hypergraph


cdef class Bitset:
    cdef init(self, cbitset data):
        self.data = data
        return self

    def __setitem__(self, int position, bool val):
        self.data.set(position, val)

    def __getitem__(self, int position):
        return self.data[position]

############# This is the templated semiring part. ##############

{% for S in semirings %}

cdef class {{S.type}}Potentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL


    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = {{S.type}}
        self.thisptr = NULL

    def times(self, {{S.type}}Potentials other):
        cdef CHypergraph{{S.type}}Potentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return {{S.type}}Potentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return {{S.type}}Potentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraph{{S.type}}Potentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return {{S.type}}Potentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraph{{S.type}}Potentials *ptr = \
            cmake_projected_potentials_{{S.type}}(self.thisptr, projection.thisptr)
        return {{S.type}}Potentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _{{S.ptype}}_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef {{S.vtype}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = _{{S.ptype}}_to_cpp(bias)

        cdef vector[{{S.vtype}}] potentials = \
             vector[{{S.vtype}}](self.hypergraph.thisptr.edges().size(),
             {{S.type}}_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = {{S.type}}_zero()
            potentials[i] = _{{S.ptype}}_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[{{S.vtype}}] potentials = \
             vector[{{S.vtype}}](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _{{S.ptype}}_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_{{S.type}}(
            self.hypergraph.thisptr,
            potentials,
            _{{S.ptype}}_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef {{S.vtype}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = _{{S.ptype}}_to_cpp(bias)

        cdef vector[{{S.vtype}}] potentials = \
             vector[{{S.vtype}}](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _{{S.ptype}}_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef {{S.vtype}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = _{{S.ptype}}_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[{{S.vtype}}] potentials = \
            vector[{{S.vtype}}](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _{{S.ptype}}_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraph{{S.type}}Potentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _{{S.ptype}}_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _{{S.ptype}}_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _{{S.ptype}}().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _{{S.ptype}}:
    @staticmethod
    def one():
        return _{{S.ptype}}_from_cpp({{S.type}}_one())

    @staticmethod
    def zero():
        return _{{S.ptype}}_from_cpp({{S.type}}_zero())


cdef {{S.vtype}} _{{S.ptype}}_to_cpp({{S.intype}} val):
    {% if 'to_cpp' in S %}
    return {{S.to_cpp}}
    {% else %}
    return val
    {% endif %}


cdef _{{S.ptype}}_from_cpp({{S.vtype}} val):
    {% if 'from_cpp' in S %}
    return {{S.from_cpp}}
    {% else %}
    return val
    {% endif %}


    # cdef {{S.vtype}} wrap

    # def __cmp__(_{{S.ptype}} self, _{{S.ptype}} other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, {{S.vtype}} wrap):
    #     self.wrap = wrap
    #     return self

    # {% if S.float %}
    # def __float__(self):
    #     return <float>self.wrap
    # {% endif %}

    # {% if S.bool %}
    # def __bool__(self):
    #     return <bool>self.wrap
    # {% endif %}

    # property value:
    #     def __get__(self):
    #         {% if S.float %}
    #         return <float>self.wrap
    #         {% elif S.bool %}
    #         return <bool>self.wrap
    #         {% else %}
    #         {{S.conversion}}
    #         {% endif %}

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_{{S.ptype}} self, _{{S.ptype}} other):
    #     return _{{S.ptype}}().init(
    #         {{S.type}}_add(self.wrap, other.wrap))

    # def __mul__(_{{S.ptype}} self, _{{S.ptype}} other):
    #     return _{{S.ptype}}().init(
    #         {{S.type}}_times(self.wrap, other.wrap))

cdef class {{S.type}}Chart:

    def __init__(self, Hypergraph graph=None):
        self.kind = {{S.type}}
        self.chart = NULL
        if graph is not None:
            self.chart = new C{{S.type}}Chart(graph.thisptr)

    def __getitem__(self, Node node):
        return _{{S.ptype}}_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _{{S.type}}Marginals:
    cdef const C{{S.type}}Marginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const C{{S.type}}Marginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _{{S.ptype}}_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _{{S.ptype}}_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have {{S.type}} marginal values." + \
                "Passed %s."%obj)

    {% if S.viterbi %}
    def threshold(self, {{S.vtype}} semi):
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi), None)
    {% endif %}

class {{S.type}}:
    Chart = {{S.type}}Chart
    Marginals = _{{S.type}}Marginals
    #Semi = _{{S.ptype}}
    Potentials = {{S.type}}Potentials

    @staticmethod
    def inside(Hypergraph graph,
               {{S.type}}Potentials potentials):
        cdef {{S.type}}Chart chart = {{S.type}}Chart()
        chart.chart = inside_{{S.type}}(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                {{S.type}}Potentials potentials,
                {{S.type}}Chart inside_chart):
        cdef {{S.type}}Chart out_chart = {{S.type}}Chart()
        out_chart.chart = outside_{{S.type}}(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart


    {% if S.viterbi %}
    @staticmethod
    def viterbi(Hypergraph graph,
                {{S.type}}Potentials potentials,
                {{S.type}}Chart chart=None):
        cdef C{{S.type}}Chart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new C{{S.type}}Chart(graph.thisptr)
        viterbi_{{S.type}}(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp

    @staticmethod
    def count_constrained_viterbi(Hypergraph graph,
                                  {{S.type}}Potentials potentials,
                                  CountingPotentials count_potentials,
                                  int limit):
        cdef CHyperpath *path = \
            count_constrained_viterbi_{{S.type}}(graph.thisptr,
                                                 deref(potentials.thisptr),
                                                 deref(count_potentials.thisptr),
                                                 limit)
        return Path().init(path, graph)

    {% endif %}

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          {{S.type}}Potentials potentials):
        cdef const C{{S.type}}Marginals *marginals = \
            {{S.type}}_compute(graph.thisptr, potentials.thisptr)
        return _{{S.type}}Marginals().init(marginals, graph)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         {{S.type}}Potentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)


{% endfor %}

####### Methods that use specific potential ########

cdef class LogViterbiDynamicViterbi:
    def __cinit__(self, Hypergraph graph):
        self.graph = graph

    def initialize(self, LogViterbiPotentials pots):
        self.thisptr = new CLogViterbiDynamicViterbi(self.graph.thisptr)
        self.thisptr.initialize(deref(pots.thisptr))
        return self.path

    def update(self, LogViterbiPotentials pots, set[int] updated):
        self.thisptr.update(deref(pots.thisptr), &updated)
        return self.path

    property path:
        def __get__(self):
            cdef BackPointers bp = BackPointers()
            bp.init(self.thisptr.back_pointers(), self.graph)
            return bp.path

def pairwise_dot(SparseVectorPotentials potentials, vec, LogViterbiPotentials weights):
    cdef vector[double] rvec = vector[double]()
    for i in vec:
        rvec.push_back(<double>i)
    cpairwise_dot(deref(potentials.thisptr), rvec, weights.thisptr)


cdef class Projection:
    def __cinit__(self):
        self.thisptr = NULL

    cdef Projection init(self, const CHypergraphProjection *thisptr,
                         Hypergraph small_graph):
        self.thisptr = thisptr
        assert thisptr.big_graph().id() >= 0
        assert thisptr.new_graph().id() >= 0
        if small_graph == None:
            self.small_graph = self.small_hypergraph()
        else:
            self.small_graph = small_graph
        self.big_graph = self.big_hypergraph()
        return self

    def compose(self, Projection other, bool reverse):
        cdef CHypergraphProjection *newptr = \
            ccompose_projections(other.thisptr, reverse, self.thisptr)
        return Projection().init(newptr, None)


    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    def __getitem__(self, obj):
        cdef const CHyperedge *edge
        cdef const CHypernode *node
        if isinstance(obj, Edge):
            edge = self.thisptr.project((<Edge>obj).edgeptr)
            assert edge.id() >= 0
            assert edge.id() == self.small_graph.edges[edge.id()].id
            return self.small_graph.edges[edge.id()]
        if isinstance(obj, Node):
            node = self.thisptr.project((<Node>obj).nodeptr)
            if node != NULL and node.id() >= 0:
                return self.small_graph.nodes[node.id()]
            else:
                return None

    def project(self, Hypergraph graph):
        cdef Hypergraph new_graph = Hypergraph()
        cdef const CHypergraphProjection *projection = self.thisptr

        # Map nodes.
        node_labels = [None] * projection.new_graph().nodes().size()
        cdef vector[const CHypernode*] old_nodes = graph.thisptr.nodes()
        cdef const CHypernode *node
        for i in range(old_nodes.size()):
            node = projection.project(old_nodes[i])
            if node != NULL and node.id() >= 0:
                node_labels[node.id()] = graph.node_labels[i]

        # Map edges.
        edge_labels = [None] * projection.new_graph().edges().size()
        cdef vector[const CHyperedge *] old_edges = graph.thisptr.edges()
        cdef const CHyperedge *edge
        for i in range(old_edges.size()):
            edge = projection.project(old_edges[i])
            # if edge != NULL and edge.id() >= 0:
            #     edge_labels[edge.id()] = graph.edge_labels[i]

        new_graph.init(projection.new_graph(), node_labels, edge_labels)
        return new_graph

    def small_hypergraph(self):
        cdef const CHypergraph *graph = self.thisptr.new_graph()
        return Hypergraph().init(graph, [], [])

    def big_hypergraph(self):
        cdef const CHypergraph *graph = self.thisptr.big_graph()
        assert graph.id() >= 0
        return Hypergraph().init(graph, [], [])


def make_pruning_projections(Hypergraph graph, BoolPotentials filt):
    cdef const CHypergraphProjection *projection = \
        cproject_hypergraph(graph.thisptr,
                            deref(filt.thisptr))
    return Projection().init(projection, None)

def valid_binary_vectors(Bitset lhs, Bitset rhs):
    return cvalid_binary_vectors(lhs.data, rhs.data)

class Potentials(LogViterbiPotentials):
    pass

################


cdef class BackPointers:
    cdef BackPointers init(self, CBackPointers *ptr,
                           Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    property path:
        def __get__(self):
            cdef CHyperpath *path = self.thisptr.construct_path()
            return Path().init(path, self.graph)

    def __getitem__(self, Node node):
        return Edge().init(self.thisptr.get(node.nodeptr), self.graph)

    # def __dealloc__(self):
    #     del self.thisptr
    #     self.thisptr = NULL

def inside(Hypergraph graph, potentials):
    r"""
    inside(Hypergraph graph, Potentials potentials):

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
       The inside chart. Type depends on potentials type, i.e. for inside potentials this
       will be the probability paths reaching this node.
    """
    return potentials.kind.inside(graph, potentials)

def outside(Hypergraph graph, potentials, inside_chart):
    r"""
    outside(Hypergraph graph, Potentials potentials, Chart inside_chart)

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

def best_path(Hypergraph graph, potentials, chart=None):
    r"""
    best_path(Hypergraph graph, Potentials potentials):

    Find the highest-scoring path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    potentials : :py:class:`Potentials`
      The potentials :math:`\theta` of the hypergraph.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """
    bp = potentials.kind.viterbi(graph, potentials, chart)
    return bp.path

def prune_hypergraph(Hypergraph graph, potentials, thres):
    r"""
    prune_hypergraph(Hypergraph graph, potentials, thres)

    Prune hyperedges with low max-marginal score from the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials of the hypergraph.

    Returns
    --------
    (hypergraph, potentials) : :py:class:`Hypergraph`, :py:class:`Potentials`
       The new hypergraphs and potentials.
    """
    return potentials.kind.prune_hypergraph(graph, potentials, thres)

def compute_marginals(Hypergraph graph, potentials):
    r"""
    compute_marginals(Hypergraph graph, Potentials potentials):

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



class Chart(LogViterbiChart):
    r"""
    Chart :math:`S^{|{\cal V}|}` associated with a hypergraph (V, E) and semiring S.

    Acts as a vector::
       >> print chart[node]
    """
    pass

class Marginals(_LogViterbiMarginals):
    r"""
    Marginal values :math:`S^{|{\cal E} \times {\cal V}|}` associated with a hypergraph ({\cal V}, {\cal E}) and semiring S.

    Acts as a dictionary::
       >> print marginals[edge]
       >> print marginals[node]
    """
    pass

inside_values = inside
outside_values = outside


def extend_hypergraph_by_count(Hypergraph graph,
                               CountingPotentials potentials,
                               int lower_limit,
                               int upper_limit,
                               int goal):
    cdef CHypergraphProjection *projection = \
        cextend_hypergraph_by_count(graph.thisptr,
                                    deref(potentials.thisptr),
                                    lower_limit,
                                    upper_limit,
                                    goal)
    return Projection().init(projection, graph)



cdef class NodeUpdates:
    def __cinit__(self, Hypergraph graph,
                  SparseVectorPotentials potentials):
        self.graph = graph
        self.children = \
            children_sparse(graph.thisptr,
                            deref(potentials.thisptr))

    def update(self, set[int] updates):
        cdef set[int] *up = \
            updated_nodes(self.graph.thisptr,
                          deref(self.children),
                          updates)
        return deref(up)
