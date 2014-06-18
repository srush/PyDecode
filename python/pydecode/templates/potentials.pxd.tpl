#cython: embedsignature=True

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.set cimport set
cimport libcpp.map as c_map
from libcpp.pair cimport pair
from libcpp cimport bool


#from libhypergraph cimport *
#import libhypergraph as py_hypergraph

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    cdef cppclass CBackPointers "BackPointers":
        CBackPointers(CHypergraph *graph)
        int get(const CHypernode *node)
        CHyperpath *construct_path()

# cdef class BackPointers:
#      cdef const CBackPointers *thisptr
#      cdef Hypergraph graph
#      cdef BackPointers init(self, const CBackPointers *ptr,
#                             Hypergraph graph)

cdef class _Potentials:
     cdef Hypergraph graph
     cdef kind

# cdef class Chart:
#      pass

############# This is the templated semiring part. ##############

{% for S in semirings %}

# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_{{S.type}} "general_inside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        C{{S.type}}Chart * chart) except +

    void outside_{{S.type}} "general_outside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        C{{S.type}}Chart inside_chart,
        C{{S.type}}Chart * chart) except +

    void viterbi_{{S.type}}"general_viterbi<{{S.ctype}}>"(
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        C{{S.type}}Chart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_{{S.type}} "node_marginals<{{S.ctype}}>"(
        const CHypergraph *graph,
        const C{{S.type}}Chart &in_chart,
        const C{{S.type}}Chart &out_chart,
        C{{S.type}}Chart * chart)

    void edge_marginals_{{S.type}} "edge_marginals<{{S.ctype}}>"(
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        const C{{S.type}}Chart &in_chart,
        const C{{S.type}}Chart &out_chart,
        {{S.cvalue}} *vals)

    # cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
    #     {{S.cvalue}} marginal(int edge)
    #     {{S.cvalue}} marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const {{S.cvalue}} &threshold)
    #     const CHypergraph *hypergraph()
    #     {{S.cvalue}} *node_marginals()
    #     {{S.cvalue}} *edge_marginals()


    cdef cppclass C{{S.type}}Chart "Chart<{{S.ctype}}>":
        C{{S.type}}Chart(const CHypergraph *graph)
        C{{S.type}}Chart(const CHypergraph *graph, {{S.cvalue}} *)
        {{S.cvalue}} get(const CHypernode *node)
        {{S.cvalue}} *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<{{S.ctype}}>":
#     C{{S.type}}Marginals *{{S.type}}_compute "Marginals<{{S.ctype}}>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraph{{S.type}}Potentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass {{S.ctype}}:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraph{{S.type}}Potentials "HypergraphPotentials<{{S.ctype}}>":
        {{S.cvalue}} dot(const CHyperpath &path) except +
        {{S.cvalue}} score(int edge)
        # CHypergraph{{S.type}}Potentials *times(
        #     const CHypergraph{{S.type}}Potentials &potentials)
        CHypergraph{{S.type}}Potentials *project_potentials(
            const CHypergraphMap)
        CHypergraph{{S.type}}Potentials(
            const CHypergraph *hypergraph,
            const vector[{{S.cvalue}}] potentials) except +
        # {{S.cvalue}} bias()
        {{S.cvalue}} *potentials()
        CHypergraph{{S.type}}Potentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_potentials_{{S.type}} "HypergraphSparsePotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[{{S.cvalue}}] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_potentials_{{S.type}} "HypergraphVectorPotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[{{S.cvalue}}] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_pointer_potentials_{{S.type}} "HypergraphPointerPotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        {{S.cvalue}} *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_projected_potentials_{{S.type}} "HypergraphMappedPotentials<{{S.ctype}}>::make_potentials" (
        CHypergraph{{S.type}}Potentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "{{S.ctype}}":
    {{S.cvalue}} {{S.type}}_one "{{S.ctype}}::one" ()
    {{S.cvalue}} {{S.type}}_zero "{{S.ctype}}::zero" ()
    {{S.cvalue}} {{S.type}}_add "{{S.ctype}}::add" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_times "{{S.ctype}}::times" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_safeadd "{{S.ctype}}::safe_add" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_safetimes "{{S.ctype}}::safe_times" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_normalize "{{S.ctype}}::normalize" ({{S.cvalue}}&)


cdef class _{{S.type}}Potentials(_Potentials):
    cdef CHypergraph{{S.type}}Potentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraph{{S.type}}Potentials *ptr)
              # HypergraphMap projection)

    # cdef {{S.cvalue}} _bias(self, bias)

# cdef class {{S.type}}Chart(Chart):
#     cdef C{{S.type}}Chart *chart
#     cdef kind

cdef class {{S.type}}Value:
    cdef {{S.cvalue}} thisval
    cdef {{S.type}}Value init(self, {{S.cvalue}} val)

{% endfor %}


cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    CHyperpath *ccount_constrained_viterbi "count_constrained_viterbi<LogViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        const CHypergraphCountingPotentials count,
        int limit) except +


# cdef extern from "Hypergraph/Potentials.hh":
#     void cpairwise_dot "pairwise_dot"(
#         const CHypergraphSparseVectorPotentials sparse_potentials,
#         const vector[double] vec,
#         CHypergraphLogViterbiPotentials *)

# cdef extern from "Hypergraph/Semirings.hh":
#     bool cvalid_binary_vectors "valid_binary_vectors" (cbitset lhs,
#                                                        cbitset rhs)


cdef extern from "Hypergraph/Algorithms.hh":
    CHypergraphMap *cextend_hypergraph_by_count "extend_hypergraph_by_count" (
        CHypergraph *graph,
        CHypergraphCountingPotentials potentials,
        int lower_limit,
        int upper_limit,
        int goal) except +

    CHypergraphMap *cextend_hypergraph_by_dfa "extend_with_dfa" (
        CHypergraph *graph,
        CHypergraphCountingPotentials potentials,
        CDFA dfa,
        vector[CDFALabel] *labels) except +

    CHypergraphMap *cproject_hypergraph "project_hypergraph"(
        const CHypergraph *hypergraph,
        const CHypergraphBoolPotentials &edge_mask) except +

    CHypergraphMap *cbinarize "binarize"(
        const CHypergraph *hypergraph)

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
