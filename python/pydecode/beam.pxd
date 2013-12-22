# Cython template hack.
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector

from pydecode.hyper cimport *

cdef extern from "<bitset>" namespace "std":
    cdef cppclass cbitset "bitset<500>":
        void set(int, int)
        bool& operator[](int)

cdef extern from "Hypergraph/BeamSearch.h":
    cdef cppclass CBeamHyp "BeamHyp":
        cbitset sig
        double current_score
        double future_score

    cdef cppclass CBeamChart "BeamChart":
        CHyperpath *get_path(int result)
        vector[CBeamHyp *] get_beam(const CHypernode *node)

    cdef cppclass CBeamGroups "BeamGroups":
        CBeamGroups(const CHypergraph *graph,
                    const vector[int] groups,
                    const vector[int] group_limit,
                    int num_groups)

    CBeamChart *cbeam_search "beam_search" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials &potentials,
        const CHypergraphBinaryVectorPotentials &constraints,
        const CLogViterbiChart &outside,
        double lower_bound,
        const CBeamGroups &groups)

cdef class BeamChart:
    cdef CBeamChart *thisptr
    cdef Hypergraph graph
    cdef init(self, CBeamChart *chart, Hypergraph graph)
