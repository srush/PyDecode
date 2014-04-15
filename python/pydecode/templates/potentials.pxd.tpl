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

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    cdef cppclass CBackPointers "BackPointers":
        CBackPointers(CHypergraph *graph)
        const CHyperedge *get(const CHypernode *node)
        CHyperpath *construct_path()

cdef class BackPointers:
     cdef const CBackPointers *thisptr
     cdef Hypergraph graph
     cdef BackPointers init(self, const CBackPointers *ptr,
                            Hypergraph graph)

cdef class Potentials:
     cdef Hypergraph hypergraph
     cdef kind

cdef class Chart:
     pass

############# This is the templated semiring part. ##############

{% for S in semirings %}

# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    C{{S.type}}Chart *inside_{{S.type}} "general_inside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta) except +

    C{{S.type}}Chart *outside_{{S.type}} "general_outside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        C{{S.type}}Chart inside_chart) except +

    void viterbi_{{S.type}}"general_viterbi<{{S.ctype}}>"(
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        C{{S.type}}Chart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
        {{S.cvalue}} marginal(const CHyperedge *edge)
        {{S.cvalue}} marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const {{S.cvalue}} &threshold)
        const CHypergraph *hypergraph()
        vector[{{S.cvalue}}] node_marginals()

    cdef cppclass C{{S.type}}Chart "Chart<{{S.ctype}}>":
        C{{S.type}}Chart(const CHypergraph *graph)
        {{S.cvalue}} get(const CHypernode *node)
        void insert(const CHypernode& node, const {{S.cvalue}}& val)
        vector[{{S.cvalue}}] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<{{S.ctype}}>":
    C{{S.type}}Marginals *{{S.type}}_compute "Marginals<{{S.ctype}}>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraph{{S.type}}Potentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass {{S.ctype}}:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraph{{S.type}}Potentials "HypergraphPotentials<{{S.ctype}}>":
        {{S.cvalue}} dot(const CHyperpath &path) except +
        {{S.cvalue}} score(const CHyperedge *edge)
        CHypergraph{{S.type}}Potentials *times(
            const CHypergraph{{S.type}}Potentials &potentials)
        CHypergraph{{S.type}}Potentials *project_potentials(
            const CHypergraphMap)
        CHypergraph{{S.type}}Potentials(
            const CHypergraph *hypergraph,
            const vector[{{S.cvalue}}] potentials,
            {{S.cvalue}} bias) except +
        {{S.cvalue}} bias()
        vector[{{S.cvalue}}] &potentials()
        CHypergraph{{S.type}}Potentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_potentials_{{S.type}} "HypergraphSparsePotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[{{S.cvalue}}] potentials,
        {{S.cvalue}} bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_potentials_{{S.type}} "HypergraphVectorPotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[{{S.cvalue}}] *potentials,
        {{S.cvalue}} bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_projected_potentials_{{S.type}} "HypergraphMappedPotentials<{{S.ctype}}>::make_potentials" (
        CHypergraph{{S.type}}Potentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "{{S.ctype}}":
    {{S.cvalue}} {{S.type}}_one "{{S.ctype}}::one" ()
    {{S.cvalue}} {{S.type}}_zero "{{S.ctype}}::zero" ()
    {{S.cvalue}} {{S.type}}_add "{{S.ctype}}::add" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_times "{{S.ctype}}::times" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_safeadd "{{S.ctype}}::safe_add" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_safetimes "{{S.ctype}}::safe_times" ({{S.cvalue}}, const {{S.cvalue}}&)
    {{S.cvalue}} {{S.type}}_normalize "{{S.ctype}}::normalize" ({{S.cvalue}}&)


cdef class {{S.type}}Potentials(Potentials):
    cdef CHypergraph{{S.type}}Potentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraph{{S.type}}Potentials *ptr,
              HypergraphMap projection)

cdef class {{S.type}}Chart(Chart):
    cdef C{{S.type}}Chart *chart
    cdef kind

cdef class {{S.type}}Value:
    cdef {{S.cvalue}} thisval
    cdef {{S.type}}Value init(self, {{S.cvalue}} val)

{% endfor %}

cdef extern from "Hypergraph/Potentials.h":
    void cpairwise_dot "pairwise_dot"(
        const CHypergraphSparseVectorPotentials sparse_potentials,
        const vector[double] vec,
        CHypergraphLogViterbiPotentials *)

# cdef extern from "Hypergraph/Semirings.h":
#     bool cvalid_binary_vectors "valid_binary_vectors" (cbitset lhs,
#                                                        cbitset rhs)


cdef extern from "Hypergraph/Algorithms.h":
    CHypergraphMap *cextend_hypergraph_by_count "extend_hypergraph_by_count" (
        CHypergraph *graph,
        CHypergraphCountingPotentials potentials,
        int lower_limit,
        int upper_limit,
        int goal)

    CHypergraphMap *cproject_hypergraph "project_hypergraph"(
        const CHypergraph *hypergraph,
        const CHypergraphBoolPotentials &edge_mask)

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
