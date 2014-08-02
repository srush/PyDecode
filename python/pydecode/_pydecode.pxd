cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool

cdef extern from "Hypergraph/Hypergraph.hh":

    cdef cppclass CHypergraphBuilder "HypergraphBuilder":
        CHypergraphBuilder(bool)
        void set_expected_size(int, int, int)
        int start_node(int)
        int add_terminal_node(int)
        bool end_node()
        int add_edge(vector[int], int label) except +
        CHypergraph *finish(bool reconstruct) except +

    cdef cppclass CHypergraph "Hypergraph":
        CHypergraph(bool)
        int id()
        int root()
        const vector[int] edges(int)
        bool terminal(int)
        int tail_nodes(int)
        int label(int)
        int node_label(int)
        int *labels()
        int *node_labels()
        int tail_node(int, int)
        int head(int)
        const vector[int] &nodes()
        const vector[int] &edges()
        vector[int] heads()

    cdef cppclass CHyperpath "Hyperpath":
        CHyperpath(const CHypergraph *graph,
                   const vector[int] nodes,
                   const vector[int] edges) except +
        vector[int] edges()
        vector[int] nodes()
        int has_edge(int)
        bool equal(const CHyperpath path)

    CHyperpath *construct_path(CHypergraph *,
                                int * back_pointers)

cdef class Labeling:
    cdef edge_labels
    cdef node_labels

cdef class _LazyEdges:
    cdef Hypergraph _graph
    #cdef vector[int] _edges
    #cdef init(self, vector[int])

cdef class _LazyVertices:
    cdef Hypergraph _graph
    #cdef vector[int] _nodes
    #cdef init(self, vector[int])

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

# cdef extern from "Hypergraph/Map.hh":
#     cdef cppclass CHypergraphMap "HypergraphMap":
#         int map_node(int)
#         int map_edge(int)
#         const CHypergraph *domain_graph()
#         const CHypergraph *range_graph()
#         const vector[int] edge_map()

cimport numpy as np
cimport cython

cdef class ChartBuilder:
    cdef _init_buffer(self, long [:] indices)
    cdef _init_list(self, indices)
            
    cdef _set_transpose(self,
                        long index,
                        long [:] tails1,
                        long [:] tails2=*,
                        long [:] tails3=*,
                        long [:] out=*)
    cdef _set_list(self, long index, tuples, out=*)
    cdef _finish_node(self, long index, result)

    cdef CHypergraph *_hg_ptr
    cdef CHypergraphBuilder *_builder
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

# Cython template hack.
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector


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



cdef extern from "Hypergraph/BeamSearch.hh" namespace "BeamChart<LogViterbi>":
    cdef cppclass CBeamHypLogViterbi "BeamChart<LogViterbi>::BeamHyp":
        float sig
        double current_score
        double future_score

    CBeamChartLogViterbi *cbeam_searchLogViterbi "BeamChart<LogViterbi>::beam_search" (
            const CHypergraph *graph,
            const double *potentials,
            const float *constraints,
            const double *outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +

    CBeamChartLogViterbi *ccube_pruningLogViterbi "BeamChart<LogViterbi>::cube_pruning" (
            const CHypergraph *graph,
            const double *potentials,
            const float *constraints,
            const double *outside,
            double lower_bound,
            const CBeamGroups &groups,
            bool recombine) except +


cdef extern from "Hypergraph/BeamSearch.hh":
    cdef cppclass CBeamChartLogViterbi "BeamChart<LogViterbi>":
        CHyperpath *get_path(int result)
        vector[CBeamHypLogViterbi *] get_beam(int)
        bool exact

cdef class BeamChartLogViterbi:
    cdef CBeamChartLogViterbi *thisptr
    cdef Hypergraph graph

    cdef init(self, CBeamChartLogViterbi *chart, Hypergraph graph)


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

############# This is the templated semiring part. ##############



# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Viterbi "general_inside<Viterbi>" (
        const CHypergraph *graph,
        const double *weights,
        double *chart) except +

    void outside_Viterbi "general_outside<Viterbi>" (
        const CHypergraph *graph,
        const double *weights,
        const double *inside_chart,
        double *chart) except +

    void viterbi_Viterbi"general_viterbi<Viterbi>"(
        const CHypergraph *graph,
        const double *weights,
        double *chart,
        int *back,
        bool *mask) except +

    void node_marginals_Viterbi "node_marginals<Viterbi>"(
        const CHypergraph *graph,
        const double *in_chart,
        const double *out_chart,
        double *chart)

    void edge_marginals_Viterbi "edge_marginals<Viterbi>"(
        const CHypergraph *graph,
        const double *weights,
        const double *in_chart,
        const double *out_chart,
        double *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "Viterbi":
    double Viterbi_one "Viterbi::one" ()
    double Viterbi_zero "Viterbi::zero" ()
    double Viterbi_add "Viterbi::add" (double,
                                                    const double&)
    double Viterbi_times "Viterbi::times" (double,
                                                        const double&)

cdef class ViterbiValue:
    cdef double thisval
    cdef ViterbiValue init(self, double val)



# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_LogViterbi "general_inside<LogViterbi>" (
        const CHypergraph *graph,
        const double *weights,
        double *chart) except +

    void outside_LogViterbi "general_outside<LogViterbi>" (
        const CHypergraph *graph,
        const double *weights,
        const double *inside_chart,
        double *chart) except +

    void viterbi_LogViterbi"general_viterbi<LogViterbi>"(
        const CHypergraph *graph,
        const double *weights,
        double *chart,
        int *back,
        bool *mask) except +

    void node_marginals_LogViterbi "node_marginals<LogViterbi>"(
        const CHypergraph *graph,
        const double *in_chart,
        const double *out_chart,
        double *chart)

    void edge_marginals_LogViterbi "edge_marginals<LogViterbi>"(
        const CHypergraph *graph,
        const double *weights,
        const double *in_chart,
        const double *out_chart,
        double *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "LogViterbi":
    double LogViterbi_one "LogViterbi::one" ()
    double LogViterbi_zero "LogViterbi::zero" ()
    double LogViterbi_add "LogViterbi::add" (double,
                                                    const double&)
    double LogViterbi_times "LogViterbi::times" (double,
                                                        const double&)

cdef class LogViterbiValue:
    cdef double thisval
    cdef LogViterbiValue init(self, double val)



# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Real "general_inside<Real>" (
        const CHypergraph *graph,
        const double *weights,
        double *chart) except +

    void outside_Real "general_outside<Real>" (
        const CHypergraph *graph,
        const double *weights,
        const double *inside_chart,
        double *chart) except +

    void viterbi_Real"general_viterbi<Real>"(
        const CHypergraph *graph,
        const double *weights,
        double *chart,
        int *back,
        bool *mask) except +

    void node_marginals_Real "node_marginals<Real>"(
        const CHypergraph *graph,
        const double *in_chart,
        const double *out_chart,
        double *chart)

    void edge_marginals_Real "edge_marginals<Real>"(
        const CHypergraph *graph,
        const double *weights,
        const double *in_chart,
        const double *out_chart,
        double *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "Real":
    double Real_one "Real::one" ()
    double Real_zero "Real::zero" ()
    double Real_add "Real::add" (double,
                                                    const double&)
    double Real_times "Real::times" (double,
                                                        const double&)

cdef class RealValue:
    cdef double thisval
    cdef RealValue init(self, double val)



# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Log "general_inside<Log>" (
        const CHypergraph *graph,
        const double *weights,
        double *chart) except +

    void outside_Log "general_outside<Log>" (
        const CHypergraph *graph,
        const double *weights,
        const double *inside_chart,
        double *chart) except +

    void viterbi_Log"general_viterbi<Log>"(
        const CHypergraph *graph,
        const double *weights,
        double *chart,
        int *back,
        bool *mask) except +

    void node_marginals_Log "node_marginals<Log>"(
        const CHypergraph *graph,
        const double *in_chart,
        const double *out_chart,
        double *chart)

    void edge_marginals_Log "edge_marginals<Log>"(
        const CHypergraph *graph,
        const double *weights,
        const double *in_chart,
        const double *out_chart,
        double *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "Log":
    double Log_one "Log::one" ()
    double Log_zero "Log::zero" ()
    double Log_add "Log::add" (double,
                                                    const double&)
    double Log_times "Log::times" (double,
                                                        const double&)

cdef class LogValue:
    cdef double thisval
    cdef LogValue init(self, double val)



# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Boolean "general_inside<Boolean>" (
        const CHypergraph *graph,
        const char *weights,
        char *chart) except +

    void outside_Boolean "general_outside<Boolean>" (
        const CHypergraph *graph,
        const char *weights,
        const char *inside_chart,
        char *chart) except +

    void viterbi_Boolean"general_viterbi<Boolean>"(
        const CHypergraph *graph,
        const char *weights,
        char *chart,
        int *back,
        bool *mask) except +

    void node_marginals_Boolean "node_marginals<Boolean>"(
        const CHypergraph *graph,
        const char *in_chart,
        const char *out_chart,
        char *chart)

    void edge_marginals_Boolean "edge_marginals<Boolean>"(
        const CHypergraph *graph,
        const char *weights,
        const char *in_chart,
        const char *out_chart,
        char *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "Boolean":
    char Boolean_one "Boolean::one" ()
    char Boolean_zero "Boolean::zero" ()
    char Boolean_add "Boolean::add" (char,
                                                    const char&)
    char Boolean_times "Boolean::times" (char,
                                                        const char&)

cdef class BooleanValue:
    cdef char thisval
    cdef BooleanValue init(self, char val)



# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_Counting "general_inside<Counting>" (
        const CHypergraph *graph,
        const int *weights,
        int *chart) except +

    void outside_Counting "general_outside<Counting>" (
        const CHypergraph *graph,
        const int *weights,
        const int *inside_chart,
        int *chart) except +

    void viterbi_Counting"general_viterbi<Counting>"(
        const CHypergraph *graph,
        const int *weights,
        int *chart,
        int *back,
        bool *mask) except +

    void node_marginals_Counting "node_marginals<Counting>"(
        const CHypergraph *graph,
        const int *in_chart,
        const int *out_chart,
        int *chart)

    void edge_marginals_Counting "edge_marginals<Counting>"(
        const CHypergraph *graph,
        const int *weights,
        const int *in_chart,
        const int *out_chart,
        int *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "Counting":
    int Counting_one "Counting::one" ()
    int Counting_zero "Counting::zero" ()
    int Counting_add "Counting::add" (int,
                                                    const int&)
    int Counting_times "Counting::times" (int,
                                                        const int&)

cdef class CountingValue:
    cdef int thisval
    cdef CountingValue init(self, int val)



# The core semiring-parameterized algorithms.

cdef extern from "Hypergraph/SemiringAlgorithms.hh":
    void inside_MinMax "general_inside<MinMax>" (
        const CHypergraph *graph,
        const double *weights,
        double *chart) except +

    void outside_MinMax "general_outside<MinMax>" (
        const CHypergraph *graph,
        const double *weights,
        const double *inside_chart,
        double *chart) except +

    void viterbi_MinMax"general_viterbi<MinMax>"(
        const CHypergraph *graph,
        const double *weights,
        double *chart,
        int *back,
        bool *mask) except +

    void node_marginals_MinMax "node_marginals<MinMax>"(
        const CHypergraph *graph,
        const double *in_chart,
        const double *out_chart,
        double *chart)

    void edge_marginals_MinMax "edge_marginals<MinMax>"(
        const CHypergraph *graph,
        const double *weights,
        const double *in_chart,
        const double *out_chart,
        double *marginals)


cdef extern from "Hypergraph/Semirings.hh" namespace "MinMax":
    double MinMax_one "MinMax::one" ()
    double MinMax_zero "MinMax::zero" ()
    double MinMax_add "MinMax::add" (double,
                                                    const double&)
    double MinMax_times "MinMax::times" (double,
                                                        const double&)

cdef class MinMaxValue:
    cdef double thisval
    cdef MinMaxValue init(self, double val)



# python mo	cdef convert_to_sparse(vector[int] positions)
# python mo	cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)


cdef extern from "Hypergraph/Algorithms.hh":
    CHypergraph *cfilter "filter"(
        const CHypergraph *hypergraph,
        const bool *edge_mask) except +

    CHypergraph *cbinarize "binarize"(const CHypergraph *hypergraph) except +

# cdef convert_to_sparse(vector[int] positions)
# cdef convert_hypergraph_map(const CHypergraphMap *hyper_map, graph1, graph2)

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


# cpdef map_potentials(dp, out_potentials)
