#cython: embedsignature=True

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.set cimport set
cimport libcpp.map as c_map
from libcpp.pair cimport pair
from libcpp cimport bool


from wrap cimport *
from hypergraph cimport *
import hypergraph as py_hypergraph

cdef extern from "<bitset>" namespace "std":
    cdef cppclass cbitset "bitset<500>":
        void set(int, int)
        bool& operator[](int)

cdef class Bitset:
    cdef cbitset data
    cdef init(self, cbitset data)

cdef extern from "Hypergraph/Algorithms.h":
    cdef cppclass CBackPointers "BackPointers":
        CBackPointers(CHypergraph *graph)
        const CHyperedge *get(const CHypernode *node)
        CHyperpath *construct_path()

cdef class BackPointers:
     cdef const CBackPointers *thisptr
     cdef Hypergraph graph
     cdef BackPointers init(self, const CBackPointers *ptr,
                            Hypergraph graph)

############# This is the templated semiring part. ##############

{% for S in semirings %}

# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_{{S.type}} "count_constrained_viterbi<{{S.ctype}}>"(
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
        {{S.vtype}} marginal(const CHyperedge *edge)
        {{S.vtype}} marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const {{S.vtype}} &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass C{{S.type}}Chart "Chart<{{S.ctype}}>":
        C{{S.type}}Chart(const CHypergraph *graph)
        {{S.vtype}} get(const CHypernode *node)
        void insert(const CHypernode& node, const {{S.vtype}}& val)

    cdef cppclass C{{S.type}}DynamicViterbi "DynamicViterbi<{{S.ctype}}>":
        C{{S.type}}DynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraph{{S.type}}Potentials theta)
        void update(const CHypergraph{{S.type}}Potentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<{{S.ctype}}>":
    C{{S.type}}Marginals *{{S.type}}_compute "Marginals<{{S.ctype}}>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraph{{S.type}}Potentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass {{S.ctype}}:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraph{{S.type}}Potentials "HypergraphPotentials<{{S.ctype}}>":
        {{S.vtype}} dot(const CHyperpath &path) except +
        {{S.vtype}} score(const CHyperedge *edge)
        CHypergraph{{S.type}}Potentials *times(
            const CHypergraph{{S.type}}Potentials &potentials)
        CHypergraph{{S.type}}Potentials *project_potentials(
            const CHypergraphProjection)
        CHypergraph{{S.type}}Potentials(
            const CHypergraph *hypergraph,
            const vector[{{S.vtype}}] potentials,
            {{S.vtype}} bias) except +
        {{S.vtype}} bias()
        CHypergraph{{S.type}}Potentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_potentials_{{S.type}} "HypergraphMapPotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[{{S.vtype}}] potentials,
        {{S.vtype}} bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_potentials_{{S.type}} "HypergraphVectorPotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[{{S.vtype}}] potentials,
        {{S.vtype}} bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_projected_potentials_{{S.type}} "HypergraphProjectedPotentials<{{S.ctype}}>::make_potentials" (
        CHypergraph{{S.type}}Potentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "{{S.ctype}}":
    {{S.vtype}} {{S.type}}_one "{{S.ctype}}::one" ()
    {{S.vtype}} {{S.type}}_zero "{{S.ctype}}::zero" ()
    {{S.vtype}} {{S.type}}_add "{{S.ctype}}::add" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_times "{{S.ctype}}::times" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_safeadd "{{S.ctype}}::safe_add" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_safetimes "{{S.ctype}}::safe_times" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_normalize "{{S.ctype}}::normalize" ({{S.vtype}}&)


cdef class {{S.type}}Potentials:
    cdef Hypergraph hypergraph
    cdef CHypergraph{{S.type}}Potentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraph{{S.type}}Potentials *ptr,
              Projection projection)

cdef class {{S.type}}Chart:
    cdef C{{S.type}}Chart *chart
    cdef kind



{% endfor %}

cdef class LogViterbiDynamicViterbi:
    cdef CLogViterbiDynamicViterbi *thisptr
    cdef Hypergraph graph

#

cdef class Projection:
    cdef const CHypergraphProjection *thisptr
    cdef Hypergraph small_graph
    cdef Hypergraph big_graph

    cdef Projection init(self, const CHypergraphProjection *thisptr,
                         Hypergraph small_graph, Hypergraph big_graph)




cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphProjection "HypergraphProjection":
        const CHyperedge *project(const CHyperedge *edge)
        const CHypernode *project(const CHypernode *node)
        const CHypergraph *big_graph()
        const CHypergraph *new_graph()


    void cpairwise_dot "pairwise_dot"(
        const CHypergraphSparseVectorPotentials sparse_potentials,
        const vector[double] vec,
        CHypergraphLogViterbiPotentials *)

cdef extern from "Hypergraph/Semirings.h":
    bool cvalid_binary_vectors "valid_binary_vectors" (cbitset lhs,
                                                       cbitset rhs)


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjection":
    CHypergraphProjection *cproject_hypergraph "HypergraphProjection::project_hypergraph"(
        const CHypergraph *hypergraph,
        const CHypergraphBoolPotentials edge_mask)


    CHypergraphProjection *ccompose_projections "HypergraphProjection::compose_projections" (const CHypergraphProjection *projection1,
                                                                                             bool reverse1,
                                                                                             const CHypergraphProjection *projection2)


cdef extern from "Hypergraph/Algorithms.h":
    CHypergraphProjection *cextend_hypergraph_by_count "extend_hypergraph_by_count" (
        CHypergraph *graph,
        CHypergraphCountingPotentials potentials,
        int lower_limit,
        int upper_limit,
        int goal)

    vector[set[int] ] *children_sparse(
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials &potentials)

    set[int] *updated_nodes(
        const CHypergraph *graph,
        const vector[set[int] ] &children,
        const set[int] &updated)

cdef class NodeUpdates:
    cdef Hypergraph graph
    cdef vector[set[int] ] *children
