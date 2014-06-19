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

{% for S in semirings %}

cdef class _{{S.type}}Potentials(_Potentials):
    def __cinit__(self, Hypergraph graph):
        self.graph = graph
        self.kind = {{S.type}}
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    cdef init(self, CHypergraph{{S.type}}Potentials *ptr):
        self.thisptr = ptr
        return self

    {% if S.npvalue %}
    def from_array(self, {{S.cvalue}} [:] X):
        self.thisptr =  \
            cmake_pointer_potentials_{{S.type}}(self.graph.thisptr,
                                               <{{S.cvalue}} *> &X[0])
        return self

    def as_array(self):
        return _{{S.type}}vector_to_numpy(self.thisptr.potentials(),
                                          len(self.graph.edges))
    {% endif %}


cdef class {{S.type}}Value:
    cdef {{S.type}}Value init(self, {{S.cvalue}} val):
        self.thisval = val
        return self

    @staticmethod
    def from_value({{S.pvalue if S.pvalue else S.cvalue}} val):
        created = {{S.type}}Value()
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
        return {{S.type}}Value().init({{S.type}}_zero())

    @staticmethod
    def one():
        return {{S.type}}Value().init({{S.type}}_one())

    def __add__({{S.type}}Value self, {{S.type}}Value other):
        return {{S.type}}Value().init({{S.type}}_add(self.thisval,
                                                  other.thisval))

    def __mul__({{S.type}}Value self, {{S.type}}Value other):
        return {{S.type}}Value().init({{S.type}}_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _{{S.type}}_from_cpp(self.thisval)

{% if S.to_cpp %}
cdef {{S.cvalue}} _{{S.type}}_to_cpp({{S.pvalue}} val):
    return <{{S.cvalue}}>{{S.to_cpp}}
{% else %}
cdef {{S.cvalue}} _{{S.type}}_to_cpp({{S.cvalue}} val):
    return val
{% endif %}

cdef _{{S.type}}_from_cpp({{S.cvalue}} val):
    {% if S.from_cpp %}
    return {{S.from_cpp}}
    {% else %}
    return val
    {% endif %}




{% if S.npvalue %}
cdef _{{S.type}}vector_to_numpy(const {{S.cvalue}} *vec,
                                int size):
    cdef view.array my_array = \
        view.array(shape=(size,),
                   itemsize=sizeof({{S.cvalue}}),
                   format="{{S.format}}",
                   mode="c", allocate_buffer=False)
    my_array.data = <char *> vec
    cdef {{S.cvalue}} [:] my_view = my_array
    return np.asarray(my_view)
{% endif %}


class {{S.type}}:
    Potentials = _{{S.type}}Potentials

    {% if S.npvalue %}
    @staticmethod
    def inside(Hypergraph graph,
               _{{S.type}}Potentials potentials,
               {{S.cvalue}} [:] chart=None):
        cdef {{S.cvalue}} [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef C{{S.type}}Chart *in_chart = new C{{S.type}}Chart(
            graph.thisptr,
            &my_chart[0])

        inside_{{S.type}}(graph.thisptr,
                          deref(potentials.thisptr),
                          in_chart)
        del in_chart
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                _{{S.type}}Potentials potentials,
                {{S.cvalue}} [:] inside_chart,
                {{S.cvalue}} [:] chart=None):
        cdef {{S.cvalue}} [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef C{{S.type}}Chart *in_chart = new C{{S.type}}Chart(
            graph.thisptr,
            &inside_chart[0])
        cdef C{{S.type}}Chart *out_chart = new C{{S.type}}Chart(
            graph.thisptr,
            &my_chart[0])

        outside_{{S.type}}(graph.thisptr,
                           deref(potentials.thisptr),
                           deref(in_chart),
                           out_chart)
        del in_chart, out_chart
        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _{{S.type}}Potentials potentials,
                          {{S.cvalue}} [:] inside_chart,
                          {{S.cvalue}} [:] outside_chart):

        cdef {{S.cvalue}} [:] node_margs = np.zeros(len(graph.nodes))
        cdef {{S.cvalue}} [:] edge_margs = np.zeros(len(graph.edges))

        cdef C{{S.type}}Chart *in_chart = new C{{S.type}}Chart(
            graph.thisptr,
            &inside_chart[0])
        cdef C{{S.type}}Chart *out_chart = new C{{S.type}}Chart(
            graph.thisptr,
            &outside_chart[0])

        cdef C{{S.type}}Chart *node_chart = new C{{S.type}}Chart(
            graph.thisptr,
            &node_margs[0])

        node_marginals_{{S.type}}(graph.thisptr,
                                  deref(in_chart),
                                  deref(out_chart),
                                  node_chart)

        edge_marginals_{{S.type}}(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(in_chart),
                                  deref(out_chart),
                                  &edge_margs[0])
        del in_chart, out_chart, node_chart
        return np.asarray(node_margs), np.asarray(edge_margs)
        # #cdef const C{{S.type}}Marginals *marginals = \


        # return (_{{S.type}}vector_to_numpy(marginals.node_marginals(),
        #                                    len(graph.nodes)),
        #         _{{S.type}}vector_to_numpy(marginals.edge_marginals(),
        #                                    len(graph.edges)))

    {% endif %}
    {% if S.viterbi %}

    @staticmethod
    def viterbi(Hypergraph graph,
                _{{S.type}}Potentials potentials,
                {{S.cvalue}} [:] chart=None):
        cdef {{S.cvalue}} [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))
        cdef C{{S.type}}Chart *in_chart = new C{{S.type}}Chart(
            graph.thisptr,
            &my_chart[0])

        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)

        viterbi_{{S.type}}(graph.thisptr,
                           deref(potentials.thisptr),
                           in_chart,
                           used_back)
        cdef CHyperpath *path = used_back.construct_path()
        del in_chart
        return Path().init(path, graph)

    {% endif %}
{% endfor %}


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
