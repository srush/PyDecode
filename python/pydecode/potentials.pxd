cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool

cdef extern from "Hypergraph/Hypergraph.hh":
    # cdef cppclass CHypernode "Hypernode":
    #     int id()
    #     vector[int ] edges()
    cdef cppclass CHypergraph "Hypergraph":
        CHypergraph(bool)
        void set_expected_size(int, int, int)
        int root()
        vector[int] edges(int)
        bool terminal(int)
        int tail_nodes(int)
        int tail_node(int, int)
        int head(int)
        int start_node()
        int add_terminal_node()
        bool end_node()
        int id()
        int new_id(int)
        int add_edge(vector[int]) except +
        void finish(bool reconstruct) except +
        vector[int] nodes()
        vector[int] edges()
        vector[int] heads()

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[int] edges) except +
        vector[int] edges()
        vector[int] nodes()
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
    cdef vector[int] _nodes
    cdef init(self, vector[int])

cdef class Hypergraph:
    cdef CHypergraph *thisptr
    cdef Labeling labeling
    cdef _cached_edges
    cdef bool unary
    cdef Hypergraph init(self, const CHypergraph *ptr, Labeling labeling)

cdef class Vertex:
    cdef int nodeptr
    cdef CHypergraph *graphptr
    cdef Hypergraph graph

    cdef Vertex init(self, int nodeptr,
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
    cdef np.ndarray _edge_indices
    cdef np.ndarray _vertex_indices

    cdef Path init(self, const CHyperpath *path, Hypergraph graph)
    cdef public equal(Path self, Path other)

cdef extern from "Hypergraph/Map.hh":
    cdef cppclass CHypergraphMap "HypergraphMap":
        int map_node(int)
        int map_edge(int)
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
#     cdef vector[int] tail_ptrs
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
    cdef vector[int] *_chart

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

    cdef _lattice

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
        int get(int node)
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
        CBackPointers *back,
        bool *mask
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
        char get(int node)
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
        CBackPointers *back,
        bool *mask
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
        double get(int node)
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
        CBackPointers *back,
        bool *mask
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
        int get(int node)
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
        CBackPointers *back,
        bool *mask
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
        double get(int node)
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

cpdef map_potentials(dp, out_potentials)
