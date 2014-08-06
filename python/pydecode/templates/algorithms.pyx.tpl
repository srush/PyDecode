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
    def __init__(self, val=None):
        if val is not None:
            self.thisval = val

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
            my_chart = np.zeros(len(graph.nodes),
                                dtype={{S.npvalue}})

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
            my_chart = np.zeros(len(graph.nodes),
                                dtype={{S.npvalue}})

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
        # cdef {{S.cvalue}} [:] node_margs = np.zeros(len(graph.nodes),
        #                                             dtype={{S.npvalue}})
        cdef {{S.cvalue}} [:] edge_margs = np.zeros(len(graph.edges),
                                                    dtype={{S.npvalue}})


        # node_marginals_{{S.type}}(graph.thisptr,
        #                           &inside_chart[0],
        #                           &outside_chart[0],
        #                           &node_margs[0])

        edge_marginals_{{S.type}}(graph.thisptr,
                                  &weights[0],
                                  &inside_chart[0],
                                  &outside_chart[0],
                                  &edge_margs[0])
        return np.asarray(edge_margs)
    #np.asarray(node_margs),

    {% endif %}
    {% if S.viterbi %}

    @staticmethod
    def kbest(Hypergraph graph,
              {{S.cvalue}} [:] weights,
              int K):
        cdef vector[CHyperpath *] paths
        ckbest_{{S.type}}(graph.thisptr, &weights[0], K, &paths)
        ret_paths = []
        for p in range(paths.size()):
            ret_paths.append(Path().init(paths[p], graph))
        return ret_paths

    @staticmethod
    def viterbi(Hypergraph graph,
                {{S.cvalue}} [:] weights,
                {{S.cvalue}} [:] chart=None,
                int [:] back_pointers=None,
                bool [:] mask=None,
                bool get_path=True):
        cdef {{S.cvalue}} [:] my_chart = chart
        if chart is None:
            my_chart = np.zeros(len(graph.nodes),
                                dtype={{S.npvalue}})

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

    @staticmethod
    def transform_to_labels(Hypergraph graph,
                            {{S.cvalue}} [:] weights,
                            int [:] labeling,
                            int label_size):
        cdef {{S.cvalue}} [:] label_weights = np.zeros(label_size,
                                 dtype={{S.npvalue}})
        ctransform_{{S.type}}(graph.thisptr,
                              &weights[0],
                              &labeling[0],
                              &label_weights[0],
                              label_size)
        return np.asarray(label_weights)

{% endfor %}


def filter_internal(Hypergraph graph, bool [:] mask):
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

def binarize_internal(Hypergraph graph):
    cdef CHypergraph *new_graph = cbinarize(graph.thisptr)
    return Hypergraph().init(new_graph, None)
