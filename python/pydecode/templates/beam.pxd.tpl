# Cython template hack.
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector

from pydecode.potentials cimport *

cdef extern from "<bitset>" namespace "std":
    cdef cppclass cbitset "bitset<500>":
        void set(int, int)
        bool& operator[](int)

cdef class Bitset:
    cdef cbitset data
    cdef init(self, cbitset data)

cdef extern from "Hypergraph/BeamSearch.h":

    cdef cppclass CBeamGroups "BeamGroups":
        CBeamGroups(const CHypergraph *graph,
                    const vector[int] groups,
                    const vector[int] group_limit,
                    int num_groups)

{% for S in semirings %}

cdef extern from "Hypergraph/BeamSearch.h" namespace "BeamChart<{{S.type}}>":
    cdef cppclass CBeamHyp{{S.type}} "BeamChart<{{S.type}}>::BeamHyp":
        {{S.cvalue}} sig
        double current_score
        double future_score

    CBeamChart{{S.type}} *cbeam_search{{S.type}} "BeamChart<{{S.type}}>::beam_search" (
            const CHypergraph *graph,
            const CHypergraphLogViterbiPotentials &potentials,
            const CHypergraph{{S.type}}s &constraints,
            const CLogViterbiChart &outside,
            double lower_bound,
            const CBeamGroups &groups)

cdef extern from "Hypergraph/BeamSearch.h":
    cdef cppclass CBeamChart{{S.type}} "BeamChart<{{S.type}}>":
        CHyperpath *get_path(int result)
        vector[CBeamHyp{{S.type}} *] get_beam(const CHypernode *node)

cdef class BeamChart{{S.type}}:
    cdef CBeamChart{{S.type}} *thisptr
    cdef Hypergraph graph
    cdef init(self, CBeamChart{{S.type}} *chart, Hypergraph graph)

{% endfor %}
