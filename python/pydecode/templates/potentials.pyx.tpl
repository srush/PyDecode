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

############# This is the templated semiring part. ##############

{% for S in semirings %}

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

class {{S.type}}:
    Value = {{S.type}}Value
    {% if S.npvalue %}
    @staticmethod
    def inside(Hypergraph graph,
               {{S.cvalue}} [:] weights,
               {{S.cvalue}} [:] chart=None):
        cdef {{S.cvalue}} [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        inside_{{S.type}}(graph.thisptr,
                          &weights[0],
                          &my_chart[0])
        return np.asarray(my_chart)

    @staticmethod
    def outside(Hypergraph graph,
                {{S.cvalue}} [:] weights,
                {{S.cvalue}} [:] inside_chart,
                {{S.cvalue}} [:] chart=None):
        cdef {{S.cvalue}} [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        outside_{{S.type}}(graph.thisptr,
                           &weights[0],
                           &inside_chart[0],
                           &my_chart[0])

        return np.asarray(my_chart)

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          {{S.cvalue}} [:] weights,
                          {{S.cvalue}} [:] inside_chart,
                          {{S.cvalue}} [:] outside_chart):
        cdef {{S.cvalue}} [:] node_margs = np.zeros(len(graph.nodes))
        cdef {{S.cvalue}} [:] edge_margs = np.zeros(len(graph.edges))


        node_marginals_{{S.type}}(graph.thisptr,
                                  &inside_chart[0],
                                  &outside_chart[0],
                                  &node_margs[0])

        edge_marginals_{{S.type}}(graph.thisptr,
                                  &weights[0],
                                  &inside_chart[0],
                                  &outside_chart[0],
                                  &edge_margs[0])
        return np.asarray(node_margs), np.asarray(edge_margs)

    {% endif %}
    {% if S.viterbi %}

    @staticmethod
    def viterbi(Hypergraph graph,
                {{S.cvalue}} [:] weights,
                {{S.cvalue}} [:] chart=None,
                int [:] back_pointers=None,
                bool [:] mask=None,
                bool get_path=True):
        cdef {{S.cvalue}} [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes))

        cdef int [:] my_back_pointers = back_pointers
        if back_pointers is None:
            my_back_pointers = np.zeros(len(graph.nodes),
                                        dtype=np.int32)

        viterbi_{{S.type}}(graph.thisptr,
                           &weights[0],
                           &my_chart[0],
                           &my_back_pointers[0],
                           (<bool *> NULL) if mask is None
                           else (<bool *>&mask[0]))

        cdef CHyperpath *path
        if get_path:
            path = construct_path(
                graph.thisptr,
                &my_back_pointers[0])
            return Path().init(path, graph)
    {% endif %}
{% endfor %}

# For mapping between hypergraphs.
# cdef convert_to_sparse(vector[int] positions):
#     data = []
#     indices = []
#     ind = [0]
#     cdef int i
#     for i in range(positions.size()):
#         if positions[i] > -1:
#             data.append(1)
#             indices.append(positions[i])
#         ind.append(len(data))
#     return (data, indices, ind)

# cdef convert_hypergraph_map(const CHypergraphMap *hyper_map,
#                             graph1_arg, graph2_arg):
#     cdef Hypergraph graph1 = graph1_arg
#     if not graph1:
#         graph1 = Hypergraph()
#         graph1.init(hyper_map.domain_graph(), Labeling(graph1))

#     cdef Hypergraph graph2 = graph2_arg
#     if not graph2:
#         graph2 = Hypergraph()
#         graph2.init(hyper_map.range_graph(), Labeling(graph2))


#     cdef vector[int] edges = hyper_map.edge_map()

#     edge_matrix = scipy.sparse.csc_matrix(
#         convert_to_sparse(hyper_map.edge_map()),
#         shape=(len(graph2.edges),
#                len(graph1.edges)),
#         dtype=np.uint8)

#     # cdef vector[int] nodes = hyper_map.node_map()

#     # node_matrix = scipy.sparse.css_matrix(
#     #     hyper_map.edge_map(),
#     #     shape=(len(graph1.nodes),
#     #            len(graph2.nodes)),
#     #     dtype=np.int8)
#     return graph1, edge_matrix, graph2

####### Methods that use specific potential ########

# def get_potentials(graph, potentials, kind=_LogViterbiPotentials):
#     # if potentials.size != len(graph.edges):
#     #     raise ValueError("Potentials must match hypergraph hyperedges size: %s != %s"%(potentials.size, len(graph.edges)))
#     return kind(graph).from_array(potentials)

# @cython.boundscheck(False)
# cpdef map_potentials(dp, out_potentials):
#     cdef np.ndarray raveled = out_potentials.ravel()
#     cdef np.ndarray potentials = raveled[dp.output_indices]
#     return potentials

def filter(Hypergraph graph, bool [:] mask):
    """
    Filter a hypergraph based on an edge mask.

    Edges with value 0 are pruned, edges with value
    1 are pruned if they are no longer in a path.

    Parameters
    -----------
    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    mask : Nx1 bool column vector.
        The pruning filter to use.

    Returns
    --------
    filtered_graph : Hypergraph
       The new projected hypergraph with :math:`|{\cal E}| = N'`.


    """
    cdef CHypergraph *new_graph = cfilter(graph.thisptr, &mask[0])
    return Hypergraph().init(new_graph, None)

def binarize(Hypergraph graph):
    cdef CHypergraph *new_graph = cbinarize(graph.thisptr)
    return Hypergraph().init(new_graph, None)
