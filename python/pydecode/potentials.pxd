from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool

cdef extern from "Hypergraph/Hypergraph.hh":
    cdef cppclass CHypernode "Hypernode":
        int id()
        vector[int ] edges()

    cdef cppclass CHypergraph "Hypergraph":
        CHypergraph(bool)
        void set_expected_size(int, int, int)
        const CHypernode *root()
        int tail_nodes(int)
        const CHypernode *tail_node(int, int)
        const CHypernode *head(int)
        const CHypernode *start_node()
        const CHypernode *add_terminal_node()
        bool end_node()
        int id()
        int new_id(int)
        int add_edge(vector[const CHypernode *]) except +
        void finish(bool reconstruct) except +
        vector[const CHypernode *] nodes()
        vector[int ] edges()

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[int] edges) except +
        vector[int] edges()
        vector[const CHypernode *] nodes()
        int has_edge(int)
        bool equal(const CHyperpath path)

cdef class Labeling:
    cdef edge_labels
    cdef node_labels

cdef class _LazyEdges:
    cdef Hypergraph _graph
    cdef vector[int] _edges
    cdef init(self, vector[int])

cdef class _LazyVertices:
    cdef Hypergraph _graph
    cdef vector[const CHypernode *] _nodes
    cdef init(self, vector[const CHypernode *])

cdef class Hypergraph:
    cdef CHypergraph *thisptr
    cdef Labeling labeling
    cdef _cached_edges
    cdef bool unary
    cdef Hypergraph init(self, const CHypergraph *ptr, Labeling labeling)

cdef class Vertex:
    cdef const CHypernode *nodeptr
    cdef CHypergraph *graphptr
    cdef Hypergraph graph

    cdef Vertex init(self, const CHypernode *nodeptr,
                   Hypergraph graph)

cdef class Edge:
    cdef int edgeptr
    cdef Hypergraph graph

    cdef Edge init(self, int edge, Hypergraph graph, bool unfinished=?)
    cdef bool unfinished

cdef class Path:
    cdef const CHyperpath *thisptr
    cdef Hypergraph graph
    cdef vector
    cdef _vertex_vector

    cdef Path init(self, const CHyperpath *path, Hypergraph graph)
    cdef public equal(Path self, Path other)

cdef extern from "Hypergraph/Map.hh":
    cdef cppclass CHypergraphMap "HypergraphMap":
        int map(int)
        const CHypernode *map(const CHypernode *node)
        const CHypergraph *domain_graph()
        const CHypergraph *range_graph()
        const vector[int] edge_map()

cdef extern from "Hypergraph/Automaton.hh":
    cdef cppclass CDFA "DFA":
        CDFA(int num_states, int num_symbols,
            const vector[map[int, int] ] &transition,
            const set[int] &final)
        bool final(int state)
        int transition(int state, int symbol)
        int valid_transition(int state, int symbol)


cdef extern from "Hypergraph/Algorithms.hh":
    CHypergraph *cmake_lattice "make_lattice"(
        int width, int height,
        const vector[vector[int] ] transitions,
        vector[CLatticeLabel ] *transitions) except +

    cdef cppclass CLatticeLabel "LatticeLabel":
        int i
        int j

    cdef cppclass CDFALabel "DFANode":
        int left_state
        int right_state


cdef class DFALabel:
    cdef CDFALabel label
    cdef _core
    cdef init(DFALabel self, CDFALabel label, core)

cdef class DFA:
    cdef const CDFA *thisptr


cdef class LatticeLabel:
    cdef CLatticeLabel label
    cdef init(LatticeLabel self, CLatticeLabel label)

cimport numpy as np
cimport cython

# cdef class IndexedEncoder:
#     cdef _hasher
#     cdef int _max_size
#     cdef _shape
#     cdef np.ndarray _multipliers
#     cpdef np.ndarray transform(self, np.ndarray element)
#     cpdef np.ndarray inverse_transform(self, np.ndarray index)

# cdef class _ChartEdge:
#     cdef vector[const CHypernode *] tail_ptrs
#     cdef values
#     cdef items

cdef class ChartBuilder:
    cpdef init(self, long [:] index)
    cpdef set(self,
              long index,
              long [:] tails1,
              long [:] tails2=*,
              long [:] tails3=*,
              long [:] out=*)

    cdef CHypergraph *_hg_ptr
    cdef vector[const CHypernode *] *_chart

    cdef bool _done
    cdef int _last
    cdef set[long] _no_tail
    cdef bool _strict
    cdef int _max_arity

    cdef int _size
    cdef items
    cdef outputs

    cdef _ndata
    cdef _nindices
    cdef _nind


    cdef int _output_size
    cdef bool _construct_output
    cdef _data
    cdef _indices
    cdef _ind


    cdef np.ndarray _edges1
    cdef np.ndarray _edges2
    cdef np.ndarray _out



# 
# cdef class IntTuple2:
#     cdef int a, b, c, d

# cdef class IntTuple2Hasher:
#     
#     cdef int _multipliers_a
#     
#     cdef int _multipliers_b
#     
#     cdef int _max_size
#     cpdef int hash_item(self, int a, int b, )
#     cpdef unhash(self, int val)
# 
# cdef class IntTuple3:
#     cdef int a, b, c, d

# cdef class IntTuple3Hasher:
#     
#     cdef int _multipliers_a
#     
#     cdef int _multipliers_b
#     
#     cdef int _multipliers_c
#     
#     cdef int _max_size
#     cpdef int hash_item(self, int a, int b, int c, )
#     cpdef unhash(self, int val)
# 
# cdef class IntTuple4:
#     cdef int a, b, c, d

# cdef class IntTuple4Hasher:
#     
#     cdef int _multipliers_a
#     
#     cdef int _multipliers_b
#     
#     cdef int _multipliers_c
#     
#     cdef int _multipliers_d
#     
#     cdef int _max_size
#     cpdef int hash_item(self, int a, int b, int c, int d, )
#     cpdef unhash(self, int val)
# 
# cdef class IntTuple5:
#     cdef int a, b, c, d

# cdef class IntTuple5Hasher:
#     
#     cdef int _multipliers_a
#     
#     cdef int _multipliers_b
#     
#     cdef int _multipliers_c
#     
#     cdef int _multipliers_d
#     
#     cdef int _multipliers_e
#     
#     cdef int _max_size
#     cpdef int hash_item(self, int a, int b, int c, int d, int e, )
#     cpdef unhash(self, int val)
# 
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

cdef extern from "Hypergraph/BeamSearch.hh":

    cdef cppclass CBeamGroups "BeamGroups":
        CBeamGroups(const CHypergraph *graph,
                    const vector[int] groups,
                    const vector[int] group_limit,
                    int num_groups)



cdef extern from "Hypergraph/BeamSearch.hh" namespace "BeamChart<BinaryVectorPotential>":
    cdef cppclass CBeamHypBinaryVectorPotential "BeamChart<BinaryVectorPotential>::BeamHyp":
        cbitset sig
        double current_score
        double future_score

    CBeamChartBinaryVectorPotential *cbeam_searchBinaryVectorPotential "BeamChart<BinaryVectorPotential>::beam_search" (
            const CHypergraph *graph,
            const CHypergraphLogViterbiPotentials &potentials,
            const CHypergraphBinaryVectorPotentials &constraints,
            const CLogViterbiChart &outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +

    CBeamChartBinaryVectorPotential *ccube_pruningBinaryVectorPotential "BeamChart<BinaryVectorPotential>::cube_pruning" (
            const CHypergraph *graph,
            const CHypergraphLogViterbiPotentials &potentials,
            const CHypergraphBinaryVectorPotentials &constraints,
            const CLogViterbiChart &outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +


cdef extern from "Hypergraph/BeamSearch.hh":
    cdef cppclass CBeamChartBinaryVectorPotential "BeamChart<BinaryVectorPotential>":
        CHyperpath *get_path(int result)
        vector[CBeamHypBinaryVectorPotential *] get_beam(const CHypernode *node)
        bool exact

cdef class BeamChartBinaryVectorPotential:
    cdef CBeamChartBinaryVectorPotential *thisptr
    cdef Hypergraph graph

    cdef init(self, CBeamChartBinaryVectorPotential *chart, Hypergraph graph)



cdef extern from "Hypergraph/BeamSearch.hh" namespace "BeamChart<AlphabetPotential>":
    cdef cppclass CBeamHypAlphabetPotential "BeamChart<AlphabetPotential>::BeamHyp":
        vector[int] sig
        double current_score
        double future_score

    CBeamChartAlphabetPotential *cbeam_searchAlphabetPotential "BeamChart<AlphabetPotential>::beam_search" (
            const CHypergraph *graph,
            const CHypergraphLogViterbiPotentials &potentials,
            const CHypergraphAlphabetPotentials &constraints,
            const CLogViterbiChart &outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +

    CBeamChartAlphabetPotential *ccube_pruningAlphabetPotential "BeamChart<AlphabetPotential>::cube_pruning" (
            const CHypergraph *graph,
            const CHypergraphLogViterbiPotentials &potentials,
            const CHypergraphAlphabetPotentials &constraints,
            const CLogViterbiChart &outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +


cdef extern from "Hypergraph/BeamSearch.hh":
    cdef cppclass CBeamChartAlphabetPotential "BeamChart<AlphabetPotential>":
        CHyperpath *get_path(int result)
        vector[CBeamHypAlphabetPotential *] get_beam(const CHypernode *node)
        bool exact

cdef class BeamChartAlphabetPotential:
    cdef CBeamChartAlphabetPotential *thisptr
    cdef Hypergraph graph

    cdef init(self, CBeamChartAlphabetPotential *chart, Hypergraph graph)



cdef extern from "Hypergraph/BeamSearch.hh" namespace "BeamChart<LogViterbiPotential>":
    cdef cppclass CBeamHypLogViterbiPotential "BeamChart<LogViterbiPotential>::BeamHyp":
        float sig
        double current_score
        double future_score

    CBeamChartLogViterbiPotential *cbeam_searchLogViterbiPotential "BeamChart<LogViterbiPotential>::beam_search" (
            const CHypergraph *graph,
            const CHypergraphLogViterbiPotentials &potentials,
            const CHypergraphLogViterbiPotentials &constraints,
            const CLogViterbiChart &outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +

    CBeamChartLogViterbiPotential *ccube_pruningLogViterbiPotential "BeamChart<LogViterbiPotential>::cube_pruning" (
            const CHypergraph *graph,
            const CHypergraphLogViterbiPotentials &potentials,
            const CHypergraphLogViterbiPotentials &constraints,
            const CLogViterbiChart &outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +


cdef extern from "Hypergraph/BeamSearch.hh":
    cdef cppclass CBeamChartLogViterbiPotential "BeamChart<LogViterbiPotential>":
        CHyperpath *get_path(int result)
        vector[CBeamHypLogViterbiPotential *] get_beam(const CHypernode *node)
        bool exact

cdef class BeamChartLogViterbiPotential:
    cdef CBeamChartLogViterbiPotential *thisptr
    cdef Hypergraph graph

    cdef init(self, CBeamChartLogViterbiPotential *chart, Hypergraph graph)


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



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Bool "general_inside<BoolPotential>" (
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        CBoolChart * chart) except +

    void outside_Bool "general_outside<BoolPotential>" (
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        CBoolChart inside_chart,
        CBoolChart * chart) except +

    void viterbi_Bool"general_viterbi<BoolPotential>"(
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        CBoolChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_Bool "node_marginals<BoolPotential>"(
        const CHypergraph *graph,
        const CBoolChart &in_chart,
        const CBoolChart &out_chart,
        CBoolChart * chart)

    void edge_marginals_Bool "edge_marginals<BoolPotential>"(
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        const CBoolChart &in_chart,
        const CBoolChart &out_chart,
        char *vals)

    # cdef cppclass CBoolMarginals "Marginals<BoolPotential>":
    #     char marginal(int edge)
    #     char marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const char &threshold)
    #     const CHypergraph *hypergraph()
    #     char *node_marginals()
    #     char *edge_marginals()


    cdef cppclass CBoolChart "Chart<BoolPotential>":
        CBoolChart(const CHypergraph *graph)
        CBoolChart(const CHypergraph *graph, char *)
        char get(const CHypernode *node)
        char *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<BoolPotential>":
#     CBoolMarginals *Bool_compute "Marginals<BoolPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphBoolPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass BoolPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphBoolPotentials "HypergraphPotentials<BoolPotential>":
        char dot(const CHyperpath &path) except +
        char score(int edge)
        # CHypergraphBoolPotentials *times(
        #     const CHypergraphBoolPotentials &potentials)
        CHypergraphBoolPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphBoolPotentials(
            const CHypergraph *hypergraph,
            const vector[char] potentials) except +
        # char bias()
        char *potentials()
        CHypergraphBoolPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_potentials_Bool "HypergraphSparsePotentials<BoolPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[char] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_potentials_Bool "HypergraphVectorPotentials<BoolPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[char] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_pointer_potentials_Bool "HypergraphPointerPotentials<BoolPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        char *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_projected_potentials_Bool "HypergraphMappedPotentials<BoolPotential>::make_potentials" (
        CHypergraphBoolPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "BoolPotential":
    char Bool_one "BoolPotential::one" ()
    char Bool_zero "BoolPotential::zero" ()
    char Bool_add "BoolPotential::add" (char, const char&)
    char Bool_times "BoolPotential::times" (char, const char&)
    char Bool_safeadd "BoolPotential::safe_add" (char, const char&)
    char Bool_safetimes "BoolPotential::safe_times" (char, const char&)
    char Bool_normalize "BoolPotential::normalize" (char&)


cdef class _BoolPotentials(_Potentials):
    cdef CHypergraphBoolPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphBoolPotentials *ptr)
              # HypergraphMap projection)

    # cdef char _bias(self, bias)

# cdef class BoolChart(Chart):
#     cdef CBoolChart *chart
#     cdef kind

cdef class BoolValue:
    cdef char thisval
    cdef BoolValue init(self, char val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Viterbi "general_inside<ViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        CViterbiChart * chart) except +

    void outside_Viterbi "general_outside<ViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        CViterbiChart inside_chart,
        CViterbiChart * chart) except +

    void viterbi_Viterbi"general_viterbi<ViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        CViterbiChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_Viterbi "node_marginals<ViterbiPotential>"(
        const CHypergraph *graph,
        const CViterbiChart &in_chart,
        const CViterbiChart &out_chart,
        CViterbiChart * chart)

    void edge_marginals_Viterbi "edge_marginals<ViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        const CViterbiChart &in_chart,
        const CViterbiChart &out_chart,
        double *vals)

    # cdef cppclass CViterbiMarginals "Marginals<ViterbiPotential>":
    #     double marginal(int edge)
    #     double marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const double &threshold)
    #     const CHypergraph *hypergraph()
    #     double *node_marginals()
    #     double *edge_marginals()


    cdef cppclass CViterbiChart "Chart<ViterbiPotential>":
        CViterbiChart(const CHypergraph *graph)
        CViterbiChart(const CHypergraph *graph, double *)
        double get(const CHypernode *node)
        double *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<ViterbiPotential>":
#     CViterbiMarginals *Viterbi_compute "Marginals<ViterbiPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphViterbiPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass ViterbiPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphViterbiPotentials "HypergraphPotentials<ViterbiPotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        # CHypergraphViterbiPotentials *times(
        #     const CHypergraphViterbiPotentials &potentials)
        CHypergraphViterbiPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials) except +
        # double bias()
        double *potentials()
        CHypergraphViterbiPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_potentials_Viterbi "HypergraphSparsePotentials<ViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_potentials_Viterbi "HypergraphVectorPotentials<ViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_pointer_potentials_Viterbi "HypergraphPointerPotentials<ViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        double *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_projected_potentials_Viterbi "HypergraphMappedPotentials<ViterbiPotential>::make_potentials" (
        CHypergraphViterbiPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "ViterbiPotential":
    double Viterbi_one "ViterbiPotential::one" ()
    double Viterbi_zero "ViterbiPotential::zero" ()
    double Viterbi_add "ViterbiPotential::add" (double, const double&)
    double Viterbi_times "ViterbiPotential::times" (double, const double&)
    double Viterbi_safeadd "ViterbiPotential::safe_add" (double, const double&)
    double Viterbi_safetimes "ViterbiPotential::safe_times" (double, const double&)
    double Viterbi_normalize "ViterbiPotential::normalize" (double&)


cdef class _ViterbiPotentials(_Potentials):
    cdef CHypergraphViterbiPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphViterbiPotentials *ptr)
              # HypergraphMap projection)

    # cdef double _bias(self, bias)

# cdef class ViterbiChart(Chart):
#     cdef CViterbiChart *chart
#     cdef kind

cdef class ViterbiValue:
    cdef double thisval
    cdef ViterbiValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Counting "general_inside<CountingPotential>" (
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta,
        CCountingChart * chart) except +

    void outside_Counting "general_outside<CountingPotential>" (
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta,
        CCountingChart inside_chart,
        CCountingChart * chart) except +

    void viterbi_Counting"general_viterbi<CountingPotential>"(
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta,
        CCountingChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_Counting "node_marginals<CountingPotential>"(
        const CHypergraph *graph,
        const CCountingChart &in_chart,
        const CCountingChart &out_chart,
        CCountingChart * chart)

    void edge_marginals_Counting "edge_marginals<CountingPotential>"(
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta,
        const CCountingChart &in_chart,
        const CCountingChart &out_chart,
        int *vals)

    # cdef cppclass CCountingMarginals "Marginals<CountingPotential>":
    #     int marginal(int edge)
    #     int marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const int &threshold)
    #     const CHypergraph *hypergraph()
    #     int *node_marginals()
    #     int *edge_marginals()


    cdef cppclass CCountingChart "Chart<CountingPotential>":
        CCountingChart(const CHypergraph *graph)
        CCountingChart(const CHypergraph *graph, int *)
        int get(const CHypernode *node)
        int *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<CountingPotential>":
#     CCountingMarginals *Counting_compute "Marginals<CountingPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphCountingPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass CountingPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphCountingPotentials "HypergraphPotentials<CountingPotential>":
        int dot(const CHyperpath &path) except +
        int score(int edge)
        # CHypergraphCountingPotentials *times(
        #     const CHypergraphCountingPotentials &potentials)
        CHypergraphCountingPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphCountingPotentials(
            const CHypergraph *hypergraph,
            const vector[int] potentials) except +
        # int bias()
        int *potentials()
        CHypergraphCountingPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_potentials_Counting "HypergraphSparsePotentials<CountingPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[int] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_potentials_Counting "HypergraphVectorPotentials<CountingPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[int] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_pointer_potentials_Counting "HypergraphPointerPotentials<CountingPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        int *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_projected_potentials_Counting "HypergraphMappedPotentials<CountingPotential>::make_potentials" (
        CHypergraphCountingPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "CountingPotential":
    int Counting_one "CountingPotential::one" ()
    int Counting_zero "CountingPotential::zero" ()
    int Counting_add "CountingPotential::add" (int, const int&)
    int Counting_times "CountingPotential::times" (int, const int&)
    int Counting_safeadd "CountingPotential::safe_add" (int, const int&)
    int Counting_safetimes "CountingPotential::safe_times" (int, const int&)
    int Counting_normalize "CountingPotential::normalize" (int&)


cdef class _CountingPotentials(_Potentials):
    cdef CHypergraphCountingPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphCountingPotentials *ptr)
              # HypergraphMap projection)

    # cdef int _bias(self, bias)

# cdef class CountingChart(Chart):
#     cdef CCountingChart *chart
#     cdef kind

cdef class CountingValue:
    cdef int thisval
    cdef CountingValue init(self, int val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_LogViterbi "general_inside<LogViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        CLogViterbiChart * chart) except +

    void outside_LogViterbi "general_outside<LogViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        CLogViterbiChart inside_chart,
        CLogViterbiChart * chart) except +

    void viterbi_LogViterbi"general_viterbi<LogViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        CLogViterbiChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_LogViterbi "node_marginals<LogViterbiPotential>"(
        const CHypergraph *graph,
        const CLogViterbiChart &in_chart,
        const CLogViterbiChart &out_chart,
        CLogViterbiChart * chart)

    void edge_marginals_LogViterbi "edge_marginals<LogViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        const CLogViterbiChart &in_chart,
        const CLogViterbiChart &out_chart,
        double *vals)

    # cdef cppclass CLogViterbiMarginals "Marginals<LogViterbiPotential>":
    #     double marginal(int edge)
    #     double marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const double &threshold)
    #     const CHypergraph *hypergraph()
    #     double *node_marginals()
    #     double *edge_marginals()


    cdef cppclass CLogViterbiChart "Chart<LogViterbiPotential>":
        CLogViterbiChart(const CHypergraph *graph)
        CLogViterbiChart(const CHypergraph *graph, double *)
        double get(const CHypernode *node)
        double *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<LogViterbiPotential>":
#     CLogViterbiMarginals *LogViterbi_compute "Marginals<LogViterbiPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphLogViterbiPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass LogViterbiPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphLogViterbiPotentials "HypergraphPotentials<LogViterbiPotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        # CHypergraphLogViterbiPotentials *times(
        #     const CHypergraphLogViterbiPotentials &potentials)
        CHypergraphLogViterbiPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphLogViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials) except +
        # double bias()
        double *potentials()
        CHypergraphLogViterbiPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_potentials_LogViterbi "HypergraphSparsePotentials<LogViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_potentials_LogViterbi "HypergraphVectorPotentials<LogViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_pointer_potentials_LogViterbi "HypergraphPointerPotentials<LogViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        double *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_projected_potentials_LogViterbi "HypergraphMappedPotentials<LogViterbiPotential>::make_potentials" (
        CHypergraphLogViterbiPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "LogViterbiPotential":
    double LogViterbi_one "LogViterbiPotential::one" ()
    double LogViterbi_zero "LogViterbiPotential::zero" ()
    double LogViterbi_add "LogViterbiPotential::add" (double, const double&)
    double LogViterbi_times "LogViterbiPotential::times" (double, const double&)
    double LogViterbi_safeadd "LogViterbiPotential::safe_add" (double, const double&)
    double LogViterbi_safetimes "LogViterbiPotential::safe_times" (double, const double&)
    double LogViterbi_normalize "LogViterbiPotential::normalize" (double&)


cdef class _LogViterbiPotentials(_Potentials):
    cdef CHypergraphLogViterbiPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphLogViterbiPotentials *ptr)
              # HypergraphMap projection)

    # cdef double _bias(self, bias)

# cdef class LogViterbiChart(Chart):
#     cdef CLogViterbiChart *chart
#     cdef kind

cdef class LogViterbiValue:
    cdef double thisval
    cdef LogViterbiValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_LogProb "general_inside<LogProbPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogProbPotentials theta,
        CLogProbChart * chart) except +

    void outside_LogProb "general_outside<LogProbPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogProbPotentials theta,
        CLogProbChart inside_chart,
        CLogProbChart * chart) except +

    void viterbi_LogProb"general_viterbi<LogProbPotential>"(
        const CHypergraph *graph,
        const CHypergraphLogProbPotentials theta,
        CLogProbChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_LogProb "node_marginals<LogProbPotential>"(
        const CHypergraph *graph,
        const CLogProbChart &in_chart,
        const CLogProbChart &out_chart,
        CLogProbChart * chart)

    void edge_marginals_LogProb "edge_marginals<LogProbPotential>"(
        const CHypergraph *graph,
        const CHypergraphLogProbPotentials theta,
        const CLogProbChart &in_chart,
        const CLogProbChart &out_chart,
        double *vals)

    # cdef cppclass CLogProbMarginals "Marginals<LogProbPotential>":
    #     double marginal(int edge)
    #     double marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const double &threshold)
    #     const CHypergraph *hypergraph()
    #     double *node_marginals()
    #     double *edge_marginals()


    cdef cppclass CLogProbChart "Chart<LogProbPotential>":
        CLogProbChart(const CHypergraph *graph)
        CLogProbChart(const CHypergraph *graph, double *)
        double get(const CHypernode *node)
        double *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<LogProbPotential>":
#     CLogProbMarginals *LogProb_compute "Marginals<LogProbPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphLogProbPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass LogProbPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphLogProbPotentials "HypergraphPotentials<LogProbPotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        # CHypergraphLogProbPotentials *times(
        #     const CHypergraphLogProbPotentials &potentials)
        CHypergraphLogProbPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphLogProbPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials) except +
        # double bias()
        double *potentials()
        CHypergraphLogProbPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<LogProbPotential>":
    CHypergraphLogProbPotentials *cmake_potentials_LogProb "HypergraphSparsePotentials<LogProbPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<LogProbPotential>":
    CHypergraphLogProbPotentials *cmake_potentials_LogProb "HypergraphVectorPotentials<LogProbPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<LogProbPotential>":
    CHypergraphLogProbPotentials *cmake_pointer_potentials_LogProb "HypergraphPointerPotentials<LogProbPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        double *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<LogProbPotential>":
    CHypergraphLogProbPotentials *cmake_projected_potentials_LogProb "HypergraphMappedPotentials<LogProbPotential>::make_potentials" (
        CHypergraphLogProbPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "LogProbPotential":
    double LogProb_one "LogProbPotential::one" ()
    double LogProb_zero "LogProbPotential::zero" ()
    double LogProb_add "LogProbPotential::add" (double, const double&)
    double LogProb_times "LogProbPotential::times" (double, const double&)
    double LogProb_safeadd "LogProbPotential::safe_add" (double, const double&)
    double LogProb_safetimes "LogProbPotential::safe_times" (double, const double&)
    double LogProb_normalize "LogProbPotential::normalize" (double&)


cdef class _LogProbPotentials(_Potentials):
    cdef CHypergraphLogProbPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphLogProbPotentials *ptr)
              # HypergraphMap projection)

    # cdef double _bias(self, bias)

# cdef class LogProbChart(Chart):
#     cdef CLogProbChart *chart
#     cdef kind

cdef class LogProbValue:
    cdef double thisval
    cdef LogProbValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Inside "general_inside<InsidePotential>" (
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        CInsideChart * chart) except +

    void outside_Inside "general_outside<InsidePotential>" (
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        CInsideChart inside_chart,
        CInsideChart * chart) except +

    void viterbi_Inside"general_viterbi<InsidePotential>"(
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        CInsideChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_Inside "node_marginals<InsidePotential>"(
        const CHypergraph *graph,
        const CInsideChart &in_chart,
        const CInsideChart &out_chart,
        CInsideChart * chart)

    void edge_marginals_Inside "edge_marginals<InsidePotential>"(
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        const CInsideChart &in_chart,
        const CInsideChart &out_chart,
        double *vals)

    # cdef cppclass CInsideMarginals "Marginals<InsidePotential>":
    #     double marginal(int edge)
    #     double marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const double &threshold)
    #     const CHypergraph *hypergraph()
    #     double *node_marginals()
    #     double *edge_marginals()


    cdef cppclass CInsideChart "Chart<InsidePotential>":
        CInsideChart(const CHypergraph *graph)
        CInsideChart(const CHypergraph *graph, double *)
        double get(const CHypernode *node)
        double *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<InsidePotential>":
#     CInsideMarginals *Inside_compute "Marginals<InsidePotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphInsidePotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass InsidePotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphInsidePotentials "HypergraphPotentials<InsidePotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        # CHypergraphInsidePotentials *times(
        #     const CHypergraphInsidePotentials &potentials)
        CHypergraphInsidePotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphInsidePotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials) except +
        # double bias()
        double *potentials()
        CHypergraphInsidePotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_potentials_Inside "HypergraphSparsePotentials<InsidePotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_potentials_Inside "HypergraphVectorPotentials<InsidePotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_pointer_potentials_Inside "HypergraphPointerPotentials<InsidePotential>::make_potentials" (
        const CHypergraph *hypergraph,
        double *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_projected_potentials_Inside "HypergraphMappedPotentials<InsidePotential>::make_potentials" (
        CHypergraphInsidePotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "InsidePotential":
    double Inside_one "InsidePotential::one" ()
    double Inside_zero "InsidePotential::zero" ()
    double Inside_add "InsidePotential::add" (double, const double&)
    double Inside_times "InsidePotential::times" (double, const double&)
    double Inside_safeadd "InsidePotential::safe_add" (double, const double&)
    double Inside_safetimes "InsidePotential::safe_times" (double, const double&)
    double Inside_normalize "InsidePotential::normalize" (double&)


cdef class _InsidePotentials(_Potentials):
    cdef CHypergraphInsidePotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphInsidePotentials *ptr)
              # HypergraphMap projection)

    # cdef double _bias(self, bias)

# cdef class InsideChart(Chart):
#     cdef CInsideChart *chart
#     cdef kind

cdef class InsideValue:
    cdef double thisval
    cdef InsideValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_MinMax "general_inside<MinMaxPotential>" (
        const CHypergraph *graph,
        const CHypergraphMinMaxPotentials theta,
        CMinMaxChart * chart) except +

    void outside_MinMax "general_outside<MinMaxPotential>" (
        const CHypergraph *graph,
        const CHypergraphMinMaxPotentials theta,
        CMinMaxChart inside_chart,
        CMinMaxChart * chart) except +

    void viterbi_MinMax"general_viterbi<MinMaxPotential>"(
        const CHypergraph *graph,
        const CHypergraphMinMaxPotentials theta,
        CMinMaxChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_MinMax "node_marginals<MinMaxPotential>"(
        const CHypergraph *graph,
        const CMinMaxChart &in_chart,
        const CMinMaxChart &out_chart,
        CMinMaxChart * chart)

    void edge_marginals_MinMax "edge_marginals<MinMaxPotential>"(
        const CHypergraph *graph,
        const CHypergraphMinMaxPotentials theta,
        const CMinMaxChart &in_chart,
        const CMinMaxChart &out_chart,
        double *vals)

    # cdef cppclass CMinMaxMarginals "Marginals<MinMaxPotential>":
    #     double marginal(int edge)
    #     double marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const double &threshold)
    #     const CHypergraph *hypergraph()
    #     double *node_marginals()
    #     double *edge_marginals()


    cdef cppclass CMinMaxChart "Chart<MinMaxPotential>":
        CMinMaxChart(const CHypergraph *graph)
        CMinMaxChart(const CHypergraph *graph, double *)
        double get(const CHypernode *node)
        double *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<MinMaxPotential>":
#     CMinMaxMarginals *MinMax_compute "Marginals<MinMaxPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphMinMaxPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass MinMaxPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphMinMaxPotentials "HypergraphPotentials<MinMaxPotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        # CHypergraphMinMaxPotentials *times(
        #     const CHypergraphMinMaxPotentials &potentials)
        CHypergraphMinMaxPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphMinMaxPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials) except +
        # double bias()
        double *potentials()
        CHypergraphMinMaxPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<MinMaxPotential>":
    CHypergraphMinMaxPotentials *cmake_potentials_MinMax "HypergraphSparsePotentials<MinMaxPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<MinMaxPotential>":
    CHypergraphMinMaxPotentials *cmake_potentials_MinMax "HypergraphVectorPotentials<MinMaxPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<MinMaxPotential>":
    CHypergraphMinMaxPotentials *cmake_pointer_potentials_MinMax "HypergraphPointerPotentials<MinMaxPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        double *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<MinMaxPotential>":
    CHypergraphMinMaxPotentials *cmake_projected_potentials_MinMax "HypergraphMappedPotentials<MinMaxPotential>::make_potentials" (
        CHypergraphMinMaxPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "MinMaxPotential":
    double MinMax_one "MinMaxPotential::one" ()
    double MinMax_zero "MinMaxPotential::zero" ()
    double MinMax_add "MinMaxPotential::add" (double, const double&)
    double MinMax_times "MinMaxPotential::times" (double, const double&)
    double MinMax_safeadd "MinMaxPotential::safe_add" (double, const double&)
    double MinMax_safetimes "MinMaxPotential::safe_times" (double, const double&)
    double MinMax_normalize "MinMaxPotential::normalize" (double&)


cdef class _MinMaxPotentials(_Potentials):
    cdef CHypergraphMinMaxPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphMinMaxPotentials *ptr)
              # HypergraphMap projection)

    # cdef double _bias(self, bias)

# cdef class MinMaxChart(Chart):
#     cdef CMinMaxChart *chart
#     cdef kind

cdef class MinMaxValue:
    cdef double thisval
    cdef MinMaxValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Alphabet "general_inside<AlphabetPotential>" (
        const CHypergraph *graph,
        const CHypergraphAlphabetPotentials theta,
        CAlphabetChart * chart) except +

    void outside_Alphabet "general_outside<AlphabetPotential>" (
        const CHypergraph *graph,
        const CHypergraphAlphabetPotentials theta,
        CAlphabetChart inside_chart,
        CAlphabetChart * chart) except +

    void viterbi_Alphabet"general_viterbi<AlphabetPotential>"(
        const CHypergraph *graph,
        const CHypergraphAlphabetPotentials theta,
        CAlphabetChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_Alphabet "node_marginals<AlphabetPotential>"(
        const CHypergraph *graph,
        const CAlphabetChart &in_chart,
        const CAlphabetChart &out_chart,
        CAlphabetChart * chart)

    void edge_marginals_Alphabet "edge_marginals<AlphabetPotential>"(
        const CHypergraph *graph,
        const CHypergraphAlphabetPotentials theta,
        const CAlphabetChart &in_chart,
        const CAlphabetChart &out_chart,
        vector[int] *vals)

    # cdef cppclass CAlphabetMarginals "Marginals<AlphabetPotential>":
    #     vector[int] marginal(int edge)
    #     vector[int] marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const vector[int] &threshold)
    #     const CHypergraph *hypergraph()
    #     vector[int] *node_marginals()
    #     vector[int] *edge_marginals()


    cdef cppclass CAlphabetChart "Chart<AlphabetPotential>":
        CAlphabetChart(const CHypergraph *graph)
        CAlphabetChart(const CHypergraph *graph, vector[int] *)
        vector[int] get(const CHypernode *node)
        vector[int] *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<AlphabetPotential>":
#     CAlphabetMarginals *Alphabet_compute "Marginals<AlphabetPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphAlphabetPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass AlphabetPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphAlphabetPotentials "HypergraphPotentials<AlphabetPotential>":
        vector[int] dot(const CHyperpath &path) except +
        vector[int] score(int edge)
        # CHypergraphAlphabetPotentials *times(
        #     const CHypergraphAlphabetPotentials &potentials)
        CHypergraphAlphabetPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphAlphabetPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[int]] potentials) except +
        # vector[int] bias()
        vector[int] *potentials()
        CHypergraphAlphabetPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<AlphabetPotential>":
    CHypergraphAlphabetPotentials *cmake_potentials_Alphabet "HypergraphSparsePotentials<AlphabetPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[int]] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<AlphabetPotential>":
    CHypergraphAlphabetPotentials *cmake_potentials_Alphabet "HypergraphVectorPotentials<AlphabetPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[int]] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<AlphabetPotential>":
    CHypergraphAlphabetPotentials *cmake_pointer_potentials_Alphabet "HypergraphPointerPotentials<AlphabetPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        vector[int] *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<AlphabetPotential>":
    CHypergraphAlphabetPotentials *cmake_projected_potentials_Alphabet "HypergraphMappedPotentials<AlphabetPotential>::make_potentials" (
        CHypergraphAlphabetPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "AlphabetPotential":
    vector[int] Alphabet_one "AlphabetPotential::one" ()
    vector[int] Alphabet_zero "AlphabetPotential::zero" ()
    vector[int] Alphabet_add "AlphabetPotential::add" (vector[int], const vector[int]&)
    vector[int] Alphabet_times "AlphabetPotential::times" (vector[int], const vector[int]&)
    vector[int] Alphabet_safeadd "AlphabetPotential::safe_add" (vector[int], const vector[int]&)
    vector[int] Alphabet_safetimes "AlphabetPotential::safe_times" (vector[int], const vector[int]&)
    vector[int] Alphabet_normalize "AlphabetPotential::normalize" (vector[int]&)


cdef class _AlphabetPotentials(_Potentials):
    cdef CHypergraphAlphabetPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphAlphabetPotentials *ptr)
              # HypergraphMap projection)

    # cdef vector[int] _bias(self, bias)

# cdef class AlphabetChart(Chart):
#     cdef CAlphabetChart *chart
#     cdef kind

cdef class AlphabetValue:
    cdef vector[int] thisval
    cdef AlphabetValue init(self, vector[int] val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_BinaryVector "general_inside<BinaryVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta,
        CBinaryVectorChart * chart) except +

    void outside_BinaryVector "general_outside<BinaryVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta,
        CBinaryVectorChart inside_chart,
        CBinaryVectorChart * chart) except +

    void viterbi_BinaryVector"general_viterbi<BinaryVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta,
        CBinaryVectorChart * chart,
        CBackPointers *back
        ) except +

    void node_marginals_BinaryVector "node_marginals<BinaryVectorPotential>"(
        const CHypergraph *graph,
        const CBinaryVectorChart &in_chart,
        const CBinaryVectorChart &out_chart,
        CBinaryVectorChart * chart)

    void edge_marginals_BinaryVector "edge_marginals<BinaryVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta,
        const CBinaryVectorChart &in_chart,
        const CBinaryVectorChart &out_chart,
        cbitset *vals)

    # cdef cppclass CBinaryVectorMarginals "Marginals<BinaryVectorPotential>":
    #     cbitset marginal(int edge)
    #     cbitset marginal(const CHypernode *node)
    #     # CHypergraphBoolPotentials *threshold(
    #     #     const cbitset &threshold)
    #     const CHypergraph *hypergraph()
    #     cbitset *node_marginals()
    #     cbitset *edge_marginals()


    cdef cppclass CBinaryVectorChart "Chart<BinaryVectorPotential>":
        CBinaryVectorChart(const CHypergraph *graph)
        CBinaryVectorChart(const CHypergraph *graph, cbitset *)
        cbitset get(const CHypernode *node)
        cbitset *chart()

cdef convert_to_sparse(vector[int] positions)
cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

# cdef extern from "Hypergraph/SemiringAlgorithms.hh" namespace "Marginals<BinaryVectorPotential>":
#     CBinaryVectorMarginals *BinaryVector_compute "Marginals<BinaryVectorPotential>::compute" (
#                            const CHypergraph *hypergraph,
#                            const CHypergraphBinaryVectorPotentials *potentials)

# cdef extern from "Hypergraph/Semirings.hh":
#     cdef cppclass BinaryVectorPotential:
#         pass


cdef extern from "Hypergraph/Potentials.hh":
    cdef cppclass CHypergraphBinaryVectorPotentials "HypergraphPotentials<BinaryVectorPotential>":
        cbitset dot(const CHyperpath &path) except +
        cbitset score(int edge)
        # CHypergraphBinaryVectorPotentials *times(
        #     const CHypergraphBinaryVectorPotentials &potentials)
        CHypergraphBinaryVectorPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphBinaryVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[cbitset] potentials) except +
        # cbitset bias()
        cbitset *potentials()
        CHypergraphBinaryVectorPotentials *clone() const


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphSparsePotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_potentials_BinaryVector "HypergraphSparsePotentials<BinaryVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[cbitset] potentials) except +


cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphVectorPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_potentials_BinaryVector "HypergraphVectorPotentials<BinaryVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[cbitset] *potentials,
        bool copy) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphPointerPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_pointer_potentials_BinaryVector "HypergraphPointerPotentials<BinaryVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        cbitset *potentials) except +

cdef extern from "Hypergraph/Potentials.hh" namespace "HypergraphMappedPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_projected_potentials_BinaryVector "HypergraphMappedPotentials<BinaryVectorPotential>::make_potentials" (
        CHypergraphBinaryVectorPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.hh" namespace "BinaryVectorPotential":
    cbitset BinaryVector_one "BinaryVectorPotential::one" ()
    cbitset BinaryVector_zero "BinaryVectorPotential::zero" ()
    cbitset BinaryVector_add "BinaryVectorPotential::add" (cbitset, const cbitset&)
    cbitset BinaryVector_times "BinaryVectorPotential::times" (cbitset, const cbitset&)
    cbitset BinaryVector_safeadd "BinaryVectorPotential::safe_add" (cbitset, const cbitset&)
    cbitset BinaryVector_safetimes "BinaryVectorPotential::safe_times" (cbitset, const cbitset&)
    cbitset BinaryVector_normalize "BinaryVectorPotential::normalize" (cbitset&)


cdef class _BinaryVectorPotentials(_Potentials):
    cdef CHypergraphBinaryVectorPotentials *thisptr
    # cdef HypergraphMap projection

    cdef init(self, CHypergraphBinaryVectorPotentials *ptr)
              # HypergraphMap projection)

    # cdef cbitset _bias(self, bias)

# cdef class BinaryVectorChart(Chart):
#     cdef CBinaryVectorChart *chart
#     cdef kind

cdef class BinaryVectorValue:
    cdef cbitset thisval
    cdef BinaryVectorValue init(self, cbitset val)




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
