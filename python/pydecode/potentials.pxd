from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool

cdef extern from "Hypergraph/Hypergraph.h":
    # cdef cppclass CHyperedge "Hyperedge":
    #     int id()
    #     const CHypernode *head_node()
    #     vector[const CHypernode *] tail_nodes()


    cdef cppclass CHypernode "Hypernode":
        int id()
        vector[int ] edges()

    cdef cppclass CHypergraph "Hypergraph":
        CHypergraph(bool)
        const CHypernode *root()
        int tail_nodes(int)
        const CHypernode *tail_node(int, int)
        const CHypernode *head(int)
        const CHypernode *start_node()
        const CHypernode *add_terminal_node()
        void end_node()
        int id()
        int new_id(int)
        int add_edge(vector[const CHypernode *]) except +
        void finish() except +
        vector[const CHypernode *] nodes()
        vector[int ] edges()

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[int] edges) except +
        vector[int] edges()
        vector[const CHypernode *] nodes()
        int has_edge(int)
        bool equal(const CHyperpath path)

cdef extern from "Hypergraph/Automaton.h":
    cdef cppclass CDFA "DFA":
        CDFA(int num_states, int num_symbols,
            const vector[map[int, int] ] &transition,
            const set[int] &final)
        bool final(int state)
        int transition(int state, int symbol)
        int valid_transition(int state, int symbol)


cdef extern from "Hypergraph/Algorithms.h":
    CHypergraph *cmake_lattice "make_lattice"(int width, int height,
                                              const vector[vector[int] ] transitions,
                                              vector[CLatticeLabel ] *transitions
                                              )


    cdef cppclass CLatticeLabel "LatticeLabel":
        int i
        int j

    cdef cppclass CDFALabel "DFANode":
        int left_state
        int right_state





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


cdef class GraphBuilder:
    cdef CHypergraph *thisptr
    cdef Hypergraph graph
    cdef edge_labels
    cdef node_labels
    cdef started

    cdef GraphBuilder init(self, Hypergraph hyper, CHypergraph *ptr)

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

    cdef Path init(self, const CHyperpath *path, Hypergraph graph)
    cdef public equal(Path self, Path other)

cdef extern from "Hypergraph/Map.h":
    cdef cppclass CHypergraphMap "HypergraphMap":
        int map(int)
        const CHypernode *map(const CHypernode *node)
        const CHypergraph *domain_graph()
        const CHypergraph *range_graph()
        CHypergraphMap *invert()
        CHypergraphMap *compose(const CHypergraphMap &)

cdef class HypergraphMap:
    cdef const CHypergraphMap *thisptr
    cdef Hypergraph range_graph
    cdef Hypergraph domain_graph

    cdef HypergraphMap init(self, const CHypergraphMap *thisptr,
                            Hypergraph range_graph, Hypergraph domain_graph)

cdef class DFALabel:
    cdef CDFALabel label
    cdef _core
    cdef init(DFALabel self, CDFALabel label, core)

cdef class DFA:
    cdef const CDFA *thisptr


cdef class LatticeLabel:
    cdef CLatticeLabel label
    cdef init(LatticeLabel self, CLatticeLabel label)

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



cdef extern from "Hypergraph/BeamSearch.h" namespace "BeamChart<BinaryVectorPotential>":
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
            const CBeamGroups &groups) except +


cdef extern from "Hypergraph/BeamSearch.h":
    cdef cppclass CBeamChartBinaryVectorPotential "BeamChart<BinaryVectorPotential>":
        CHyperpath *get_path(int result)
        vector[CBeamHypBinaryVectorPotential *] get_beam(const CHypernode *node)
        bool exact

cdef class BeamChartBinaryVectorPotential:
    cdef CBeamChartBinaryVectorPotential *thisptr
    cdef Hypergraph graph

    cdef init(self, CBeamChartBinaryVectorPotential *chart, Hypergraph graph)



cdef extern from "Hypergraph/BeamSearch.h" namespace "BeamChart<AlphabetPotential>":
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
            const CBeamGroups &groups) except +


cdef extern from "Hypergraph/BeamSearch.h":
    cdef cppclass CBeamChartAlphabetPotential "BeamChart<AlphabetPotential>":
        CHyperpath *get_path(int result)
        vector[CBeamHypAlphabetPotential *] get_beam(const CHypernode *node)
        bool exact

cdef class BeamChartAlphabetPotential:
    cdef CBeamChartAlphabetPotential *thisptr
    cdef Hypergraph graph

    cdef init(self, CBeamChartAlphabetPotential *chart, Hypergraph graph)


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
        int get(const CHypernode *node)
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



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CViterbiChart *inside_Viterbi "general_inside<ViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta) except +

    CViterbiChart *outside_Viterbi "general_outside<ViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        CViterbiChart inside_chart) except +

    void viterbi_Viterbi"general_viterbi<ViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        CViterbiChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CViterbiMarginals "Marginals<ViterbiPotential>":
        double marginal(int edge)
        double marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const double &threshold)
        const CHypergraph *hypergraph()
        vector[double] node_marginals()

    cdef cppclass CViterbiChart "Chart<ViterbiPotential>":
        CViterbiChart(const CHypergraph *graph)
        double get(const CHypernode *node)
        void insert(const CHypernode& node, const double& val)
        vector[double] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<ViterbiPotential>":
    CViterbiMarginals *Viterbi_compute "Marginals<ViterbiPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphViterbiPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass ViterbiPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphViterbiPotentials "HypergraphPotentials<ViterbiPotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        CHypergraphViterbiPotentials *times(
            const CHypergraphViterbiPotentials &potentials)
        CHypergraphViterbiPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials,
            double bias) except +
        double bias()
        vector[double] &potentials()
        CHypergraphViterbiPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_potentials_Viterbi "HypergraphSparsePotentials<ViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_potentials_Viterbi "HypergraphVectorPotentials<ViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        double bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_projected_potentials_Viterbi "HypergraphMappedPotentials<ViterbiPotential>::make_potentials" (
        CHypergraphViterbiPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "ViterbiPotential":
    double Viterbi_one "ViterbiPotential::one" ()
    double Viterbi_zero "ViterbiPotential::zero" ()
    double Viterbi_add "ViterbiPotential::add" (double, const double&)
    double Viterbi_times "ViterbiPotential::times" (double, const double&)
    double Viterbi_safeadd "ViterbiPotential::safe_add" (double, const double&)
    double Viterbi_safetimes "ViterbiPotential::safe_times" (double, const double&)
    double Viterbi_normalize "ViterbiPotential::normalize" (double&)


cdef class ViterbiPotentials(Potentials):
    cdef CHypergraphViterbiPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphViterbiPotentials *ptr,
              HypergraphMap projection)

    cdef double _bias(self, bias)

cdef class ViterbiChart(Chart):
    cdef CViterbiChart *chart
    cdef kind

cdef class ViterbiValue:
    cdef double thisval
    cdef ViterbiValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CLogViterbiChart *inside_LogViterbi "general_inside<LogViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta) except +

    CLogViterbiChart *outside_LogViterbi "general_outside<LogViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        CLogViterbiChart inside_chart) except +

    void viterbi_LogViterbi"general_viterbi<LogViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        CLogViterbiChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CLogViterbiMarginals "Marginals<LogViterbiPotential>":
        double marginal(int edge)
        double marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const double &threshold)
        const CHypergraph *hypergraph()
        vector[double] node_marginals()

    cdef cppclass CLogViterbiChart "Chart<LogViterbiPotential>":
        CLogViterbiChart(const CHypergraph *graph)
        double get(const CHypernode *node)
        void insert(const CHypernode& node, const double& val)
        vector[double] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<LogViterbiPotential>":
    CLogViterbiMarginals *LogViterbi_compute "Marginals<LogViterbiPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphLogViterbiPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass LogViterbiPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphLogViterbiPotentials "HypergraphPotentials<LogViterbiPotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        CHypergraphLogViterbiPotentials *times(
            const CHypergraphLogViterbiPotentials &potentials)
        CHypergraphLogViterbiPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphLogViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials,
            double bias) except +
        double bias()
        vector[double] &potentials()
        CHypergraphLogViterbiPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_potentials_LogViterbi "HypergraphSparsePotentials<LogViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_potentials_LogViterbi "HypergraphVectorPotentials<LogViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        double bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_projected_potentials_LogViterbi "HypergraphMappedPotentials<LogViterbiPotential>::make_potentials" (
        CHypergraphLogViterbiPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "LogViterbiPotential":
    double LogViterbi_one "LogViterbiPotential::one" ()
    double LogViterbi_zero "LogViterbiPotential::zero" ()
    double LogViterbi_add "LogViterbiPotential::add" (double, const double&)
    double LogViterbi_times "LogViterbiPotential::times" (double, const double&)
    double LogViterbi_safeadd "LogViterbiPotential::safe_add" (double, const double&)
    double LogViterbi_safetimes "LogViterbiPotential::safe_times" (double, const double&)
    double LogViterbi_normalize "LogViterbiPotential::normalize" (double&)


cdef class LogViterbiPotentials(Potentials):
    cdef CHypergraphLogViterbiPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphLogViterbiPotentials *ptr,
              HypergraphMap projection)

    cdef double _bias(self, bias)

cdef class LogViterbiChart(Chart):
    cdef CLogViterbiChart *chart
    cdef kind

cdef class LogViterbiValue:
    cdef double thisval
    cdef LogViterbiValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CInsideChart *inside_Inside "general_inside<InsidePotential>" (
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta) except +

    CInsideChart *outside_Inside "general_outside<InsidePotential>" (
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        CInsideChart inside_chart) except +

    void viterbi_Inside"general_viterbi<InsidePotential>"(
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        CInsideChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CInsideMarginals "Marginals<InsidePotential>":
        double marginal(int edge)
        double marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const double &threshold)
        const CHypergraph *hypergraph()
        vector[double] node_marginals()

    cdef cppclass CInsideChart "Chart<InsidePotential>":
        CInsideChart(const CHypergraph *graph)
        double get(const CHypernode *node)
        void insert(const CHypernode& node, const double& val)
        vector[double] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<InsidePotential>":
    CInsideMarginals *Inside_compute "Marginals<InsidePotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphInsidePotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass InsidePotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphInsidePotentials "HypergraphPotentials<InsidePotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        CHypergraphInsidePotentials *times(
            const CHypergraphInsidePotentials &potentials)
        CHypergraphInsidePotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphInsidePotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials,
            double bias) except +
        double bias()
        vector[double] &potentials()
        CHypergraphInsidePotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_potentials_Inside "HypergraphSparsePotentials<InsidePotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_potentials_Inside "HypergraphVectorPotentials<InsidePotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        double bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_projected_potentials_Inside "HypergraphMappedPotentials<InsidePotential>::make_potentials" (
        CHypergraphInsidePotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "InsidePotential":
    double Inside_one "InsidePotential::one" ()
    double Inside_zero "InsidePotential::zero" ()
    double Inside_add "InsidePotential::add" (double, const double&)
    double Inside_times "InsidePotential::times" (double, const double&)
    double Inside_safeadd "InsidePotential::safe_add" (double, const double&)
    double Inside_safetimes "InsidePotential::safe_times" (double, const double&)
    double Inside_normalize "InsidePotential::normalize" (double&)


cdef class InsidePotentials(Potentials):
    cdef CHypergraphInsidePotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphInsidePotentials *ptr,
              HypergraphMap projection)

    cdef double _bias(self, bias)

cdef class InsideChart(Chart):
    cdef CInsideChart *chart
    cdef kind

cdef class InsideValue:
    cdef double thisval
    cdef InsideValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CMinMaxChart *inside_MinMax "general_inside<MinMaxPotential>" (
        const CHypergraph *graph,
        const CHypergraphMinMaxPotentials theta) except +

    CMinMaxChart *outside_MinMax "general_outside<MinMaxPotential>" (
        const CHypergraph *graph,
        const CHypergraphMinMaxPotentials theta,
        CMinMaxChart inside_chart) except +

    void viterbi_MinMax"general_viterbi<MinMaxPotential>"(
        const CHypergraph *graph,
        const CHypergraphMinMaxPotentials theta,
        CMinMaxChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CMinMaxMarginals "Marginals<MinMaxPotential>":
        double marginal(int edge)
        double marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const double &threshold)
        const CHypergraph *hypergraph()
        vector[double] node_marginals()

    cdef cppclass CMinMaxChart "Chart<MinMaxPotential>":
        CMinMaxChart(const CHypergraph *graph)
        double get(const CHypernode *node)
        void insert(const CHypernode& node, const double& val)
        vector[double] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<MinMaxPotential>":
    CMinMaxMarginals *MinMax_compute "Marginals<MinMaxPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphMinMaxPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass MinMaxPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphMinMaxPotentials "HypergraphPotentials<MinMaxPotential>":
        double dot(const CHyperpath &path) except +
        double score(int edge)
        CHypergraphMinMaxPotentials *times(
            const CHypergraphMinMaxPotentials &potentials)
        CHypergraphMinMaxPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphMinMaxPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials,
            double bias) except +
        double bias()
        vector[double] &potentials()
        CHypergraphMinMaxPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<MinMaxPotential>":
    CHypergraphMinMaxPotentials *cmake_potentials_MinMax "HypergraphSparsePotentials<MinMaxPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<MinMaxPotential>":
    CHypergraphMinMaxPotentials *cmake_potentials_MinMax "HypergraphVectorPotentials<MinMaxPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] *potentials,
        double bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<MinMaxPotential>":
    CHypergraphMinMaxPotentials *cmake_projected_potentials_MinMax "HypergraphMappedPotentials<MinMaxPotential>::make_potentials" (
        CHypergraphMinMaxPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "MinMaxPotential":
    double MinMax_one "MinMaxPotential::one" ()
    double MinMax_zero "MinMaxPotential::zero" ()
    double MinMax_add "MinMaxPotential::add" (double, const double&)
    double MinMax_times "MinMaxPotential::times" (double, const double&)
    double MinMax_safeadd "MinMaxPotential::safe_add" (double, const double&)
    double MinMax_safetimes "MinMaxPotential::safe_times" (double, const double&)
    double MinMax_normalize "MinMaxPotential::normalize" (double&)


cdef class MinMaxPotentials(Potentials):
    cdef CHypergraphMinMaxPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphMinMaxPotentials *ptr,
              HypergraphMap projection)

    cdef double _bias(self, bias)

cdef class MinMaxChart(Chart):
    cdef CMinMaxChart *chart
    cdef kind

cdef class MinMaxValue:
    cdef double thisval
    cdef MinMaxValue init(self, double val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CSparseVectorChart *inside_SparseVector "general_inside<SparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials theta) except +

    CSparseVectorChart *outside_SparseVector "general_outside<SparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials theta,
        CSparseVectorChart inside_chart) except +

    void viterbi_SparseVector"general_viterbi<SparseVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials theta,
        CSparseVectorChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CSparseVectorMarginals "Marginals<SparseVectorPotential>":
        vector[pair[int, int]] marginal(int edge)
        vector[pair[int, int]] marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const vector[pair[int, int]] &threshold)
        const CHypergraph *hypergraph()
        vector[vector[pair[int, int]]] node_marginals()

    cdef cppclass CSparseVectorChart "Chart<SparseVectorPotential>":
        CSparseVectorChart(const CHypergraph *graph)
        vector[pair[int, int]] get(const CHypernode *node)
        void insert(const CHypernode& node, const vector[pair[int, int]]& val)
        vector[vector[pair[int, int]]] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<SparseVectorPotential>":
    CSparseVectorMarginals *SparseVector_compute "Marginals<SparseVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphSparseVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass SparseVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphSparseVectorPotentials "HypergraphPotentials<SparseVectorPotential>":
        vector[pair[int, int]] dot(const CHyperpath &path) except +
        vector[pair[int, int]] score(int edge)
        CHypergraphSparseVectorPotentials *times(
            const CHypergraphSparseVectorPotentials &potentials)
        CHypergraphSparseVectorPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphSparseVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[pair[int, int]]] potentials,
            vector[pair[int, int]] bias) except +
        vector[pair[int, int]] bias()
        vector[vector[pair[int, int]]] &potentials()
        CHypergraphSparseVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<SparseVectorPotential>":
    CHypergraphSparseVectorPotentials *cmake_potentials_SparseVector "HypergraphSparsePotentials<SparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<SparseVectorPotential>":
    CHypergraphSparseVectorPotentials *cmake_potentials_SparseVector "HypergraphVectorPotentials<SparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[pair[int, int]]] *potentials,
        vector[pair[int, int]] bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<SparseVectorPotential>":
    CHypergraphSparseVectorPotentials *cmake_projected_potentials_SparseVector "HypergraphMappedPotentials<SparseVectorPotential>::make_potentials" (
        CHypergraphSparseVectorPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "SparseVectorPotential":
    vector[pair[int, int]] SparseVector_one "SparseVectorPotential::one" ()
    vector[pair[int, int]] SparseVector_zero "SparseVectorPotential::zero" ()
    vector[pair[int, int]] SparseVector_add "SparseVectorPotential::add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_times "SparseVectorPotential::times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_safeadd "SparseVectorPotential::safe_add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_safetimes "SparseVectorPotential::safe_times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_normalize "SparseVectorPotential::normalize" (vector[pair[int, int]]&)


cdef class SparseVectorPotentials(Potentials):
    cdef CHypergraphSparseVectorPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphSparseVectorPotentials *ptr,
              HypergraphMap projection)

    cdef vector[pair[int, int]] _bias(self, bias)

cdef class SparseVectorChart(Chart):
    cdef CSparseVectorChart *chart
    cdef kind

cdef class SparseVectorValue:
    cdef vector[pair[int, int]] thisval
    cdef SparseVectorValue init(self, vector[pair[int, int]] val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CAlphabetChart *inside_Alphabet "general_inside<AlphabetPotential>" (
        const CHypergraph *graph,
        const CHypergraphAlphabetPotentials theta) except +

    CAlphabetChart *outside_Alphabet "general_outside<AlphabetPotential>" (
        const CHypergraph *graph,
        const CHypergraphAlphabetPotentials theta,
        CAlphabetChart inside_chart) except +

    void viterbi_Alphabet"general_viterbi<AlphabetPotential>"(
        const CHypergraph *graph,
        const CHypergraphAlphabetPotentials theta,
        CAlphabetChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CAlphabetMarginals "Marginals<AlphabetPotential>":
        vector[int] marginal(int edge)
        vector[int] marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const vector[int] &threshold)
        const CHypergraph *hypergraph()
        vector[vector[int]] node_marginals()

    cdef cppclass CAlphabetChart "Chart<AlphabetPotential>":
        CAlphabetChart(const CHypergraph *graph)
        vector[int] get(const CHypernode *node)
        void insert(const CHypernode& node, const vector[int]& val)
        vector[vector[int]] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<AlphabetPotential>":
    CAlphabetMarginals *Alphabet_compute "Marginals<AlphabetPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphAlphabetPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass AlphabetPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphAlphabetPotentials "HypergraphPotentials<AlphabetPotential>":
        vector[int] dot(const CHyperpath &path) except +
        vector[int] score(int edge)
        CHypergraphAlphabetPotentials *times(
            const CHypergraphAlphabetPotentials &potentials)
        CHypergraphAlphabetPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphAlphabetPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[int]] potentials,
            vector[int] bias) except +
        vector[int] bias()
        vector[vector[int]] &potentials()
        CHypergraphAlphabetPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<AlphabetPotential>":
    CHypergraphAlphabetPotentials *cmake_potentials_Alphabet "HypergraphSparsePotentials<AlphabetPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[int]] potentials,
        vector[int] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<AlphabetPotential>":
    CHypergraphAlphabetPotentials *cmake_potentials_Alphabet "HypergraphVectorPotentials<AlphabetPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[int]] *potentials,
        vector[int] bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<AlphabetPotential>":
    CHypergraphAlphabetPotentials *cmake_projected_potentials_Alphabet "HypergraphMappedPotentials<AlphabetPotential>::make_potentials" (
        CHypergraphAlphabetPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "AlphabetPotential":
    vector[int] Alphabet_one "AlphabetPotential::one" ()
    vector[int] Alphabet_zero "AlphabetPotential::zero" ()
    vector[int] Alphabet_add "AlphabetPotential::add" (vector[int], const vector[int]&)
    vector[int] Alphabet_times "AlphabetPotential::times" (vector[int], const vector[int]&)
    vector[int] Alphabet_safeadd "AlphabetPotential::safe_add" (vector[int], const vector[int]&)
    vector[int] Alphabet_safetimes "AlphabetPotential::safe_times" (vector[int], const vector[int]&)
    vector[int] Alphabet_normalize "AlphabetPotential::normalize" (vector[int]&)


cdef class AlphabetPotentials(Potentials):
    cdef CHypergraphAlphabetPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphAlphabetPotentials *ptr,
              HypergraphMap projection)

    cdef vector[int] _bias(self, bias)

cdef class AlphabetChart(Chart):
    cdef CAlphabetChart *chart
    cdef kind

cdef class AlphabetValue:
    cdef vector[int] thisval
    cdef AlphabetValue init(self, vector[int] val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CMinSparseVectorChart *inside_MinSparseVector "general_inside<MinSparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphMinSparseVectorPotentials theta) except +

    CMinSparseVectorChart *outside_MinSparseVector "general_outside<MinSparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphMinSparseVectorPotentials theta,
        CMinSparseVectorChart inside_chart) except +

    void viterbi_MinSparseVector"general_viterbi<MinSparseVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphMinSparseVectorPotentials theta,
        CMinSparseVectorChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CMinSparseVectorMarginals "Marginals<MinSparseVectorPotential>":
        vector[pair[int, int]] marginal(int edge)
        vector[pair[int, int]] marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const vector[pair[int, int]] &threshold)
        const CHypergraph *hypergraph()
        vector[vector[pair[int, int]]] node_marginals()

    cdef cppclass CMinSparseVectorChart "Chart<MinSparseVectorPotential>":
        CMinSparseVectorChart(const CHypergraph *graph)
        vector[pair[int, int]] get(const CHypernode *node)
        void insert(const CHypernode& node, const vector[pair[int, int]]& val)
        vector[vector[pair[int, int]]] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<MinSparseVectorPotential>":
    CMinSparseVectorMarginals *MinSparseVector_compute "Marginals<MinSparseVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphMinSparseVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass MinSparseVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphMinSparseVectorPotentials "HypergraphPotentials<MinSparseVectorPotential>":
        vector[pair[int, int]] dot(const CHyperpath &path) except +
        vector[pair[int, int]] score(int edge)
        CHypergraphMinSparseVectorPotentials *times(
            const CHypergraphMinSparseVectorPotentials &potentials)
        CHypergraphMinSparseVectorPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphMinSparseVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[pair[int, int]]] potentials,
            vector[pair[int, int]] bias) except +
        vector[pair[int, int]] bias()
        vector[vector[pair[int, int]]] &potentials()
        CHypergraphMinSparseVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<MinSparseVectorPotential>":
    CHypergraphMinSparseVectorPotentials *cmake_potentials_MinSparseVector "HypergraphSparsePotentials<MinSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<MinSparseVectorPotential>":
    CHypergraphMinSparseVectorPotentials *cmake_potentials_MinSparseVector "HypergraphVectorPotentials<MinSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[pair[int, int]]] *potentials,
        vector[pair[int, int]] bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<MinSparseVectorPotential>":
    CHypergraphMinSparseVectorPotentials *cmake_projected_potentials_MinSparseVector "HypergraphMappedPotentials<MinSparseVectorPotential>::make_potentials" (
        CHypergraphMinSparseVectorPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "MinSparseVectorPotential":
    vector[pair[int, int]] MinSparseVector_one "MinSparseVectorPotential::one" ()
    vector[pair[int, int]] MinSparseVector_zero "MinSparseVectorPotential::zero" ()
    vector[pair[int, int]] MinSparseVector_add "MinSparseVectorPotential::add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_times "MinSparseVectorPotential::times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_safeadd "MinSparseVectorPotential::safe_add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_safetimes "MinSparseVectorPotential::safe_times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_normalize "MinSparseVectorPotential::normalize" (vector[pair[int, int]]&)


cdef class MinSparseVectorPotentials(Potentials):
    cdef CHypergraphMinSparseVectorPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphMinSparseVectorPotentials *ptr,
              HypergraphMap projection)

    cdef vector[pair[int, int]] _bias(self, bias)

cdef class MinSparseVectorChart(Chart):
    cdef CMinSparseVectorChart *chart
    cdef kind

cdef class MinSparseVectorValue:
    cdef vector[pair[int, int]] thisval
    cdef MinSparseVectorValue init(self, vector[pair[int, int]] val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CMaxSparseVectorChart *inside_MaxSparseVector "general_inside<MaxSparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphMaxSparseVectorPotentials theta) except +

    CMaxSparseVectorChart *outside_MaxSparseVector "general_outside<MaxSparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphMaxSparseVectorPotentials theta,
        CMaxSparseVectorChart inside_chart) except +

    void viterbi_MaxSparseVector"general_viterbi<MaxSparseVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphMaxSparseVectorPotentials theta,
        CMaxSparseVectorChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CMaxSparseVectorMarginals "Marginals<MaxSparseVectorPotential>":
        vector[pair[int, int]] marginal(int edge)
        vector[pair[int, int]] marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const vector[pair[int, int]] &threshold)
        const CHypergraph *hypergraph()
        vector[vector[pair[int, int]]] node_marginals()

    cdef cppclass CMaxSparseVectorChart "Chart<MaxSparseVectorPotential>":
        CMaxSparseVectorChart(const CHypergraph *graph)
        vector[pair[int, int]] get(const CHypernode *node)
        void insert(const CHypernode& node, const vector[pair[int, int]]& val)
        vector[vector[pair[int, int]]] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<MaxSparseVectorPotential>":
    CMaxSparseVectorMarginals *MaxSparseVector_compute "Marginals<MaxSparseVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphMaxSparseVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass MaxSparseVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphMaxSparseVectorPotentials "HypergraphPotentials<MaxSparseVectorPotential>":
        vector[pair[int, int]] dot(const CHyperpath &path) except +
        vector[pair[int, int]] score(int edge)
        CHypergraphMaxSparseVectorPotentials *times(
            const CHypergraphMaxSparseVectorPotentials &potentials)
        CHypergraphMaxSparseVectorPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphMaxSparseVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[pair[int, int]]] potentials,
            vector[pair[int, int]] bias) except +
        vector[pair[int, int]] bias()
        vector[vector[pair[int, int]]] &potentials()
        CHypergraphMaxSparseVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<MaxSparseVectorPotential>":
    CHypergraphMaxSparseVectorPotentials *cmake_potentials_MaxSparseVector "HypergraphSparsePotentials<MaxSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<MaxSparseVectorPotential>":
    CHypergraphMaxSparseVectorPotentials *cmake_potentials_MaxSparseVector "HypergraphVectorPotentials<MaxSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[pair[int, int]]] *potentials,
        vector[pair[int, int]] bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<MaxSparseVectorPotential>":
    CHypergraphMaxSparseVectorPotentials *cmake_projected_potentials_MaxSparseVector "HypergraphMappedPotentials<MaxSparseVectorPotential>::make_potentials" (
        CHypergraphMaxSparseVectorPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "MaxSparseVectorPotential":
    vector[pair[int, int]] MaxSparseVector_one "MaxSparseVectorPotential::one" ()
    vector[pair[int, int]] MaxSparseVector_zero "MaxSparseVectorPotential::zero" ()
    vector[pair[int, int]] MaxSparseVector_add "MaxSparseVectorPotential::add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_times "MaxSparseVectorPotential::times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_safeadd "MaxSparseVectorPotential::safe_add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_safetimes "MaxSparseVectorPotential::safe_times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_normalize "MaxSparseVectorPotential::normalize" (vector[pair[int, int]]&)


cdef class MaxSparseVectorPotentials(Potentials):
    cdef CHypergraphMaxSparseVectorPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphMaxSparseVectorPotentials *ptr,
              HypergraphMap projection)

    cdef vector[pair[int, int]] _bias(self, bias)

cdef class MaxSparseVectorChart(Chart):
    cdef CMaxSparseVectorChart *chart
    cdef kind

cdef class MaxSparseVectorValue:
    cdef vector[pair[int, int]] thisval
    cdef MaxSparseVectorValue init(self, vector[pair[int, int]] val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CCountingChart *inside_Counting "general_inside<CountingPotential>" (
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta) except +

    CCountingChart *outside_Counting "general_outside<CountingPotential>" (
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta,
        CCountingChart inside_chart) except +

    void viterbi_Counting"general_viterbi<CountingPotential>"(
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta,
        CCountingChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CCountingMarginals "Marginals<CountingPotential>":
        int marginal(int edge)
        int marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const int &threshold)
        const CHypergraph *hypergraph()
        vector[int] node_marginals()

    cdef cppclass CCountingChart "Chart<CountingPotential>":
        CCountingChart(const CHypergraph *graph)
        int get(const CHypernode *node)
        void insert(const CHypernode& node, const int& val)
        vector[int] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<CountingPotential>":
    CCountingMarginals *Counting_compute "Marginals<CountingPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphCountingPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass CountingPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphCountingPotentials "HypergraphPotentials<CountingPotential>":
        int dot(const CHyperpath &path) except +
        int score(int edge)
        CHypergraphCountingPotentials *times(
            const CHypergraphCountingPotentials &potentials)
        CHypergraphCountingPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphCountingPotentials(
            const CHypergraph *hypergraph,
            const vector[int] potentials,
            int bias) except +
        int bias()
        vector[int] &potentials()
        CHypergraphCountingPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_potentials_Counting "HypergraphSparsePotentials<CountingPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[int] potentials,
        int bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_potentials_Counting "HypergraphVectorPotentials<CountingPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[int] *potentials,
        int bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_projected_potentials_Counting "HypergraphMappedPotentials<CountingPotential>::make_potentials" (
        CHypergraphCountingPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "CountingPotential":
    int Counting_one "CountingPotential::one" ()
    int Counting_zero "CountingPotential::zero" ()
    int Counting_add "CountingPotential::add" (int, const int&)
    int Counting_times "CountingPotential::times" (int, const int&)
    int Counting_safeadd "CountingPotential::safe_add" (int, const int&)
    int Counting_safetimes "CountingPotential::safe_times" (int, const int&)
    int Counting_normalize "CountingPotential::normalize" (int&)


cdef class CountingPotentials(Potentials):
    cdef CHypergraphCountingPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphCountingPotentials *ptr,
              HypergraphMap projection)

    cdef int _bias(self, bias)

cdef class CountingChart(Chart):
    cdef CCountingChart *chart
    cdef kind

cdef class CountingValue:
    cdef int thisval
    cdef CountingValue init(self, int val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CBoolChart *inside_Bool "general_inside<BoolPotential>" (
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta) except +

    CBoolChart *outside_Bool "general_outside<BoolPotential>" (
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        CBoolChart inside_chart) except +

    void viterbi_Bool"general_viterbi<BoolPotential>"(
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        CBoolChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CBoolMarginals "Marginals<BoolPotential>":
        bool marginal(int edge)
        bool marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const bool &threshold)
        const CHypergraph *hypergraph()
        vector[bool] node_marginals()

    cdef cppclass CBoolChart "Chart<BoolPotential>":
        CBoolChart(const CHypergraph *graph)
        bool get(const CHypernode *node)
        void insert(const CHypernode& node, const bool& val)
        vector[bool] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<BoolPotential>":
    CBoolMarginals *Bool_compute "Marginals<BoolPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphBoolPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass BoolPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphBoolPotentials "HypergraphPotentials<BoolPotential>":
        bool dot(const CHyperpath &path) except +
        bool score(int edge)
        CHypergraphBoolPotentials *times(
            const CHypergraphBoolPotentials &potentials)
        CHypergraphBoolPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphBoolPotentials(
            const CHypergraph *hypergraph,
            const vector[bool] potentials,
            bool bias) except +
        bool bias()
        vector[bool] &potentials()
        CHypergraphBoolPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_potentials_Bool "HypergraphSparsePotentials<BoolPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[bool] potentials,
        bool bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_potentials_Bool "HypergraphVectorPotentials<BoolPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[bool] *potentials,
        bool bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_projected_potentials_Bool "HypergraphMappedPotentials<BoolPotential>::make_potentials" (
        CHypergraphBoolPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "BoolPotential":
    bool Bool_one "BoolPotential::one" ()
    bool Bool_zero "BoolPotential::zero" ()
    bool Bool_add "BoolPotential::add" (bool, const bool&)
    bool Bool_times "BoolPotential::times" (bool, const bool&)
    bool Bool_safeadd "BoolPotential::safe_add" (bool, const bool&)
    bool Bool_safetimes "BoolPotential::safe_times" (bool, const bool&)
    bool Bool_normalize "BoolPotential::normalize" (bool&)


cdef class BoolPotentials(Potentials):
    cdef CHypergraphBoolPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphBoolPotentials *ptr,
              HypergraphMap projection)

    cdef bool _bias(self, bias)

cdef class BoolChart(Chart):
    cdef CBoolChart *chart
    cdef kind

cdef class BoolValue:
    cdef bool thisval
    cdef BoolValue init(self, bool val)



# Type identifiers.

cdef extern from "Hypergraph/SemiringAlgorithms.h":
    CBinaryVectorChart *inside_BinaryVector "general_inside<BinaryVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta) except +

    CBinaryVectorChart *outside_BinaryVector "general_outside<BinaryVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta,
        CBinaryVectorChart inside_chart) except +

    void viterbi_BinaryVector"general_viterbi<BinaryVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta,
        CBinaryVectorChart * chart,
        CBackPointers *back
        ) except +

    cdef cppclass CBinaryVectorMarginals "Marginals<BinaryVectorPotential>":
        cbitset marginal(int edge)
        cbitset marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const cbitset &threshold)
        const CHypergraph *hypergraph()
        vector[cbitset] node_marginals()

    cdef cppclass CBinaryVectorChart "Chart<BinaryVectorPotential>":
        CBinaryVectorChart(const CHypergraph *graph)
        cbitset get(const CHypernode *node)
        void insert(const CHypernode& node, const cbitset& val)
        vector[cbitset] chart()


cdef extern from "Hypergraph/SemiringAlgorithms.h" namespace "Marginals<BinaryVectorPotential>":
    CBinaryVectorMarginals *BinaryVector_compute "Marginals<BinaryVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphBinaryVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass BinaryVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphBinaryVectorPotentials "HypergraphPotentials<BinaryVectorPotential>":
        cbitset dot(const CHyperpath &path) except +
        cbitset score(int edge)
        CHypergraphBinaryVectorPotentials *times(
            const CHypergraphBinaryVectorPotentials &potentials)
        CHypergraphBinaryVectorPotentials *project_potentials(
            const CHypergraphMap)
        CHypergraphBinaryVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[cbitset] potentials,
            cbitset bias) except +
        cbitset bias()
        vector[cbitset] &potentials()
        CHypergraphBinaryVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphSparsePotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_potentials_BinaryVector "HypergraphSparsePotentials<BinaryVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[cbitset] potentials,
        cbitset bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_potentials_BinaryVector "HypergraphVectorPotentials<BinaryVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[cbitset] *potentials,
        cbitset bias,
        bool copy) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMappedPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_projected_potentials_BinaryVector "HypergraphMappedPotentials<BinaryVectorPotential>::make_potentials" (
        CHypergraphBinaryVectorPotentials *base_potentials,
        const CHypergraphMap *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "BinaryVectorPotential":
    cbitset BinaryVector_one "BinaryVectorPotential::one" ()
    cbitset BinaryVector_zero "BinaryVectorPotential::zero" ()
    cbitset BinaryVector_add "BinaryVectorPotential::add" (cbitset, const cbitset&)
    cbitset BinaryVector_times "BinaryVectorPotential::times" (cbitset, const cbitset&)
    cbitset BinaryVector_safeadd "BinaryVectorPotential::safe_add" (cbitset, const cbitset&)
    cbitset BinaryVector_safetimes "BinaryVectorPotential::safe_times" (cbitset, const cbitset&)
    cbitset BinaryVector_normalize "BinaryVectorPotential::normalize" (cbitset&)


cdef class BinaryVectorPotentials(Potentials):
    cdef CHypergraphBinaryVectorPotentials *thisptr
    cdef HypergraphMap projection

    cdef init(self, CHypergraphBinaryVectorPotentials *ptr,
              HypergraphMap projection)

    cdef cbitset _bias(self, bias)

cdef class BinaryVectorChart(Chart):
    cdef CBinaryVectorChart *chart
    cdef kind

cdef class BinaryVectorValue:
    cdef cbitset thisval
    cdef BinaryVectorValue init(self, cbitset val)



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

    CHypergraphMap *cextend_hypergraph_by_dfa "extend_with_dfa" (
        CHypergraph *graph,
        CHypergraphCountingPotentials potentials,
        CDFA dfa,
        vector[CDFALabel] *labels) except +

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
