# cython: profile=True
#cython: embedsignature=True

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.set cimport set
cimport libcpp.map as c_map
from libcpp.pair cimport pair
from libcpp cimport bool

cdef class _Potentials:
     cdef Hypergraph graph
     cdef kind

############# This is the templated semiring part. ##############

{% for S in semirings %}

# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_{{S.type}} "general_inside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const {{S.cvalue}} *weights,
        {{S.cvalue}} *chart) except +

    void outside_{{S.type}} "general_outside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const {{S.cvalue}} *weights,
        const {{S.cvalue}} *inside_chart,
        {{S.cvalue}} *chart) except +

    void viterbi_{{S.type}}"general_viterbi<{{S.ctype}}>"(
        const CHypergraph *graph,
        const {{S.cvalue}} *weights,
        {{S.cvalue}} *chart,
        int *back,
        bool *mask) except +

    void node_marginals_{{S.type}} "node_marginals<{{S.ctype}}>"(
        const CHypergraph *graph,
        const {{S.cvalue}} *in_chart,
        const {{S.cvalue}} *out_chart,
        {{S.cvalue}} *chart)

    void edge_marginals_{{S.type}} "edge_marginals<{{S.ctype}}>"(
        const CHypergraph *graph,
        const {{S.cvalue}} *weights,
        const {{S.cvalue}} *in_chart,
        const {{S.cvalue}} *out_chart,
        {{S.cvalue}} *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "{{S.ctype}}":
    {{S.cvalue}} {{S.type}}_one "{{S.ctype}}::one" ()
    {{S.cvalue}} {{S.type}}_zero "{{S.ctype}}::zero" ()
    {{S.cvalue}} {{S.type}}_add "{{S.ctype}}::add" ({{S.cvalue}},
                                                    const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_times "{{S.ctype}}::times" ({{S.cvalue}},
                                                        const {{S.cvalue}}&)


# cdef class _{{S.type}}Potentials(_Potentials):
#     cdef CHypergraph{{S.type}}Potentials *thisptr
#     # cdef HypergraphMap projection

#     cdef init(self, CHypergraph{{S.type}}Potentials *ptr)
#               # HypergraphMap projection)

#     # cdef {{S.cvalue}} _bias(self, bias)


cdef class {{S.type}}Value:
    cdef {{S.cvalue}} thisval
    cdef {{S.type}}Value init(self, {{S.cvalue}} val)

{% endfor %}

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh":
#     CHyperpath *ccount_constrained_viterbi "count_constrained_viterbi<LogViterbiPotential>" (
#         const CHypergraph *graph,
#         const CHypergraphLogViterbiPotentials theta,
#         const CHypergraphCountingPotentials count,
#         int limit) except +


# cdef extern from "Hypergraph/Potentials.hh":
#     void cpairwise_dot "pairwise_dot"(
#         const CHypergraphSparseVectorPotentials sparse_potentials,
#         const vector[double] vec,
#         CHypergraphLogViterbiPotentials *)

# cdef extern from "Hypergraph/Semirings.hh":
#     bool cvalid_binary_vectors "valid_binary_vectors" (cbitset lhs,
#                                                        cbitset rhs)


cdef extern from "Hypergraph/Algorithms.hh":
    CHypergraphMap *cproject_hypergraph "project_hypergraph"(
        const CHypergraph *hypergraph,
        const bool *edge_mask) except +

#     CHypergraphMap *cextend_hypergraph_by_count "extend_hypergraph_by_count" (
#         CHypergraph *graph,
#         CHypergraphCountingPotentials potentials,
#         int lower_limit,
#         int upper_limit,
#         int goal) except +

#     CHypergraphMap *cextend_hypergraph_by_dfa "extend_with_dfa" (
#         CHypergraph *graph,
#         CHypergraphCountingPotentials potentials,
#         CDFA dfa,
#         vector[CDFALabel] *labels) except +


#     CHypergraphMap *cbinarize "binarize"(
#         const CHypergraph *hypergraph)






    # vector[set[int] ] *children_sparse(
    #     const CHypergraph *graph,
    #     const CHypergraphSparseVectorPotentials &potentials)

    # set[int] *updated_nodes(
    #     const CHypergraph *graph,
    #     const vector[set[int] ] &children,
    #     const set[int] &updated)

# cdef class NodeUpdates:
#     cdef Hypergraph graph
#     cdef vector[set[int] ] *children

cpdef map_potentials(dp, out_potentials)
