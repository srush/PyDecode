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


############# This is the templated semiring part. ##############

{% for S in semirings %}

cdef class {{S.type}}Potentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

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
        return {{S.type}}Potentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraph{{S.type}}Potentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return {{S.type}}Potentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraph{{S.type}}Potentials *ptr = \
            cmake_projected_potentials_{{S.type}}(self.thisptr,
                                                  projection.thisptr)
        return {{S.type}}Potentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _{{S.type}}_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef {{S.cvalue}} my_bias
    #     if bias is None:
    #         my_bias = {{S.type}}_one()
    #     else:
    #         my_bias = _{{S.type}}_to_cpp(bias)

    #     cdef vector[{{S.cvalue}}] potentials = \
    #          vector[{{S.cvalue}}](self.hypergraph.thisptr.edges().size(),
    #          {{S.type}}_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = {{S.type}}_zero()
    #         potentials[i] = _{{S.type}}_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[{{S.cvalue}}] potentials = \
            vector[{{S.cvalue}}](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _{{S.type}}_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                        potentials,
                                        _{{S.type}}_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef {{S.cvalue}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = _{{S.type}}_to_cpp(bias)

        cdef vector[{{S.cvalue}}] potentials = \
            vector[{{S.cvalue}}](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _{{S.type}}_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef {{S.cvalue}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = _{{S.type}}_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[{{S.cvalue}}] potentials = \
            vector[{{S.cvalue}}](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _{{S.type}}_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraph{{S.type}}Potentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _{{S.type}}_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _{{S.type}}_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _{{S.type}}:
    cdef _{{S.type}} init(self, {{S.cvalue}} val):
        self.thisval = val
        return self

    @staticmethod
    def from_value({{S.cvalue}} val):
        created = _{{S.type}}()
        created.thisval = _{{S.type}}_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _{{S.type}}_from_cpp({{S.type}}_zero())

    @staticmethod
    def one_raw():
        return _{{S.type}}_from_cpp({{S.type}}_one())

    @staticmethod
    def zero():
        return _{{S.type}}().init({{S.type}}_zero())

    @staticmethod
    def one():
        return _{{S.type}}().init({{S.type}}_one())

    def __add__(_{{S.type}} self, _{{S.type}} other):
        return _{{S.type}}().init({{S.type}}_add(self.thisval,
                                                  other.thisval))

    def __mul__(_{{S.type}} self, _{{S.type}} other):
        return _{{S.type}}().init({{S.type}}_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _{{S.type}}_from_cpp(self.thisval)

cdef {{S.cvalue}} _{{S.type}}_to_cpp({{S.cvalue}} val):
    return val


cdef _{{S.type}}_from_cpp({{S.cvalue}} val):
    return val

cdef class {{S.type}}Chart:

    def __init__(self, Hypergraph graph=None):
        self.kind = {{S.type}}
        self.chart = NULL
        if graph is not None:
            self.chart = new C{{S.type}}Chart(graph.thisptr)

    def __getitem__(self, Node node):
        return _{{S.type}}_from_cpp(self.chart.get(node.nodeptr))

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
            return _{{S.type}}_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _{{S.type}}_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have {{S.type}} marginal values." +
                "Passed %s." % obj)

    {% if S.viterbi %}

    def threshold(self, {{S.cvalue}} semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)
    {% endif %}


class {{S.type}}:
    Chart = {{S.type}}Chart
    Marginals = _{{S.type}}Marginals
    #Semi = _{{S.type}}
    Potentials = {{S.type}}Potentials

    @staticmethod
    def inside(Hypergraph graph,
               {{S.type}}Potentials potentials):
        cdef {{S.type}}Chart chart = {{S.type}}Chart()
        chart.chart = inside_{{S.type}}(graph.thisptr,
                                        deref(potentials.thisptr))
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


class Potentials(LogViterbiPotentials):
    pass


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
       The inside chart. Type depends on potentials type, i.e.
       for inside potentials this will be the probability paths
       reaching this node.
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


def best_path(Hypergraph graph, Potentials potentials, chart=None):
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


class Chart(LogViterbiChart):
    r"""
    A dynamic programming chart associated with a hypergraph.

    Chart :math:`S^{|{\cal V}|}` associated with a hypergraph (V, E)
    and semiring S.

    Acts as a vector::
       >> print chart[node]
    """
    pass


class Marginals(_LogViterbiMarginals):
    r"""
    Marginal values with a hypergraph and potentials.

    Marginal values :math:`S^{|{\cal E} \times {\cal V}|}` associated
    with a hypergraph ({\cal V}, {\cal E}) and semiring S.

    Acts as a dictionary::
       >> print marginals[edge]
       >> print marginals[node]
    """
    pass

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
