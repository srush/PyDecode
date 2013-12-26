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



# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_Viterbi "count_constrained_viterbi<ViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CViterbiMarginals "Marginals<ViterbiPotential>":
        double marginal(const CHyperedge *edge)
        double marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const double &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CViterbiChart "Chart<ViterbiPotential>":
        CViterbiChart(const CHypergraph *graph)
        double get(const CHypernode *node)
        void insert(const CHypernode& node, const double& val)

    cdef cppclass CViterbiDynamicViterbi "DynamicViterbi<ViterbiPotential>":
        CViterbiDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphViterbiPotentials theta)
        void update(const CHypergraphViterbiPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<ViterbiPotential>":
    CViterbiMarginals *Viterbi_compute "Marginals<ViterbiPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphViterbiPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass ViterbiPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphViterbiPotentials "HypergraphPotentials<ViterbiPotential>":
        double dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphViterbiPotentials *times(
            const CHypergraphViterbiPotentials &potentials)
        CHypergraphViterbiPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials,
            double bias) except +
        double bias()
        CHypergraphViterbiPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_potentials_Viterbi "HypergraphMapPotentials<ViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_potentials_Viterbi "HypergraphVectorPotentials<ViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<ViterbiPotential>":
    CHypergraphViterbiPotentials *cmake_projected_potentials_Viterbi "HypergraphProjectedPotentials<ViterbiPotential>::make_potentials" (
        CHypergraphViterbiPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "ViterbiPotential":
    double Viterbi_one "ViterbiPotential::one" ()
    double Viterbi_zero "ViterbiPotential::zero" ()
    double Viterbi_add "ViterbiPotential::add" (double, const double&)
    double Viterbi_times "ViterbiPotential::times" (double, const double&)
    double Viterbi_safeadd "ViterbiPotential::safe_add" (double, const double&)
    double Viterbi_safetimes "ViterbiPotential::safe_times" (double, const double&)
    double Viterbi_normalize "ViterbiPotential::normalize" (double&)


cdef class ViterbiPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphViterbiPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphViterbiPotentials *ptr,
              Projection projection)

cdef class ViterbiChart:
    cdef CViterbiChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_LogViterbi "count_constrained_viterbi<LogViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CLogViterbiMarginals "Marginals<LogViterbiPotential>":
        double marginal(const CHyperedge *edge)
        double marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const double &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CLogViterbiChart "Chart<LogViterbiPotential>":
        CLogViterbiChart(const CHypergraph *graph)
        double get(const CHypernode *node)
        void insert(const CHypernode& node, const double& val)

    cdef cppclass CLogViterbiDynamicViterbi "DynamicViterbi<LogViterbiPotential>":
        CLogViterbiDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphLogViterbiPotentials theta)
        void update(const CHypergraphLogViterbiPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<LogViterbiPotential>":
    CLogViterbiMarginals *LogViterbi_compute "Marginals<LogViterbiPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphLogViterbiPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass LogViterbiPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphLogViterbiPotentials "HypergraphPotentials<LogViterbiPotential>":
        double dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphLogViterbiPotentials *times(
            const CHypergraphLogViterbiPotentials &potentials)
        CHypergraphLogViterbiPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphLogViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials,
            double bias) except +
        double bias()
        CHypergraphLogViterbiPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_potentials_LogViterbi "HypergraphMapPotentials<LogViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_potentials_LogViterbi "HypergraphVectorPotentials<LogViterbiPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<LogViterbiPotential>":
    CHypergraphLogViterbiPotentials *cmake_projected_potentials_LogViterbi "HypergraphProjectedPotentials<LogViterbiPotential>::make_potentials" (
        CHypergraphLogViterbiPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "LogViterbiPotential":
    double LogViterbi_one "LogViterbiPotential::one" ()
    double LogViterbi_zero "LogViterbiPotential::zero" ()
    double LogViterbi_add "LogViterbiPotential::add" (double, const double&)
    double LogViterbi_times "LogViterbiPotential::times" (double, const double&)
    double LogViterbi_safeadd "LogViterbiPotential::safe_add" (double, const double&)
    double LogViterbi_safetimes "LogViterbiPotential::safe_times" (double, const double&)
    double LogViterbi_normalize "LogViterbiPotential::normalize" (double&)


cdef class LogViterbiPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphLogViterbiPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphLogViterbiPotentials *ptr,
              Projection projection)

cdef class LogViterbiChart:
    cdef CLogViterbiChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_Inside "count_constrained_viterbi<InsidePotential>"(
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CInsideMarginals "Marginals<InsidePotential>":
        double marginal(const CHyperedge *edge)
        double marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const double &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CInsideChart "Chart<InsidePotential>":
        CInsideChart(const CHypergraph *graph)
        double get(const CHypernode *node)
        void insert(const CHypernode& node, const double& val)

    cdef cppclass CInsideDynamicViterbi "DynamicViterbi<InsidePotential>":
        CInsideDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphInsidePotentials theta)
        void update(const CHypergraphInsidePotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<InsidePotential>":
    CInsideMarginals *Inside_compute "Marginals<InsidePotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphInsidePotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass InsidePotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphInsidePotentials "HypergraphPotentials<InsidePotential>":
        double dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphInsidePotentials *times(
            const CHypergraphInsidePotentials &potentials)
        CHypergraphInsidePotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphInsidePotentials(
            const CHypergraph *hypergraph,
            const vector[double] potentials,
            double bias) except +
        double bias()
        CHypergraphInsidePotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_potentials_Inside "HypergraphMapPotentials<InsidePotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_potentials_Inside "HypergraphVectorPotentials<InsidePotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[double] potentials,
        double bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<InsidePotential>":
    CHypergraphInsidePotentials *cmake_projected_potentials_Inside "HypergraphProjectedPotentials<InsidePotential>::make_potentials" (
        CHypergraphInsidePotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "InsidePotential":
    double Inside_one "InsidePotential::one" ()
    double Inside_zero "InsidePotential::zero" ()
    double Inside_add "InsidePotential::add" (double, const double&)
    double Inside_times "InsidePotential::times" (double, const double&)
    double Inside_safeadd "InsidePotential::safe_add" (double, const double&)
    double Inside_safetimes "InsidePotential::safe_times" (double, const double&)
    double Inside_normalize "InsidePotential::normalize" (double&)


cdef class InsidePotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphInsidePotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphInsidePotentials *ptr,
              Projection projection)

cdef class InsideChart:
    cdef CInsideChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_Bool "count_constrained_viterbi<BoolPotential>"(
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CBoolMarginals "Marginals<BoolPotential>":
        bool marginal(const CHyperedge *edge)
        bool marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const bool &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CBoolChart "Chart<BoolPotential>":
        CBoolChart(const CHypergraph *graph)
        bool get(const CHypernode *node)
        void insert(const CHypernode& node, const bool& val)

    cdef cppclass CBoolDynamicViterbi "DynamicViterbi<BoolPotential>":
        CBoolDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphBoolPotentials theta)
        void update(const CHypergraphBoolPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<BoolPotential>":
    CBoolMarginals *Bool_compute "Marginals<BoolPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphBoolPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass BoolPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphBoolPotentials "HypergraphPotentials<BoolPotential>":
        bool dot(const CHyperpath &path) except +
        bool score(const CHyperedge *edge)
        CHypergraphBoolPotentials *times(
            const CHypergraphBoolPotentials &potentials)
        CHypergraphBoolPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphBoolPotentials(
            const CHypergraph *hypergraph,
            const vector[bool] potentials,
            bool bias) except +
        bool bias()
        CHypergraphBoolPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_potentials_Bool "HypergraphMapPotentials<BoolPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[bool] potentials,
        bool bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_potentials_Bool "HypergraphVectorPotentials<BoolPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[bool] potentials,
        bool bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<BoolPotential>":
    CHypergraphBoolPotentials *cmake_projected_potentials_Bool "HypergraphProjectedPotentials<BoolPotential>::make_potentials" (
        CHypergraphBoolPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "BoolPotential":
    bool Bool_one "BoolPotential::one" ()
    bool Bool_zero "BoolPotential::zero" ()
    bool Bool_add "BoolPotential::add" (bool, const bool&)
    bool Bool_times "BoolPotential::times" (bool, const bool&)
    bool Bool_safeadd "BoolPotential::safe_add" (bool, const bool&)
    bool Bool_safetimes "BoolPotential::safe_times" (bool, const bool&)
    bool Bool_normalize "BoolPotential::normalize" (bool&)


cdef class BoolPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphBoolPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphBoolPotentials *ptr,
              Projection projection)

cdef class BoolChart:
    cdef CBoolChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_SparseVector "count_constrained_viterbi<SparseVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CSparseVectorMarginals "Marginals<SparseVectorPotential>":
        vector[pair[int, int]] marginal(const CHyperedge *edge)
        vector[pair[int, int]] marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const vector[pair[int, int]] &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CSparseVectorChart "Chart<SparseVectorPotential>":
        CSparseVectorChart(const CHypergraph *graph)
        vector[pair[int, int]] get(const CHypernode *node)
        void insert(const CHypernode& node, const vector[pair[int, int]]& val)

    cdef cppclass CSparseVectorDynamicViterbi "DynamicViterbi<SparseVectorPotential>":
        CSparseVectorDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphSparseVectorPotentials theta)
        void update(const CHypergraphSparseVectorPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<SparseVectorPotential>":
    CSparseVectorMarginals *SparseVector_compute "Marginals<SparseVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphSparseVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass SparseVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphSparseVectorPotentials "HypergraphPotentials<SparseVectorPotential>":
        vector[pair[int, int]] dot(const CHyperpath &path) except +
        vector[pair[int, int]] score(const CHyperedge *edge)
        CHypergraphSparseVectorPotentials *times(
            const CHypergraphSparseVectorPotentials &potentials)
        CHypergraphSparseVectorPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphSparseVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[pair[int, int]]] potentials,
            vector[pair[int, int]] bias) except +
        vector[pair[int, int]] bias()
        CHypergraphSparseVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<SparseVectorPotential>":
    CHypergraphSparseVectorPotentials *cmake_potentials_SparseVector "HypergraphMapPotentials<SparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<SparseVectorPotential>":
    CHypergraphSparseVectorPotentials *cmake_potentials_SparseVector "HypergraphVectorPotentials<SparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<SparseVectorPotential>":
    CHypergraphSparseVectorPotentials *cmake_projected_potentials_SparseVector "HypergraphProjectedPotentials<SparseVectorPotential>::make_potentials" (
        CHypergraphSparseVectorPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "SparseVectorPotential":
    vector[pair[int, int]] SparseVector_one "SparseVectorPotential::one" ()
    vector[pair[int, int]] SparseVector_zero "SparseVectorPotential::zero" ()
    vector[pair[int, int]] SparseVector_add "SparseVectorPotential::add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_times "SparseVectorPotential::times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_safeadd "SparseVectorPotential::safe_add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_safetimes "SparseVectorPotential::safe_times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] SparseVector_normalize "SparseVectorPotential::normalize" (vector[pair[int, int]]&)


cdef class SparseVectorPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphSparseVectorPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphSparseVectorPotentials *ptr,
              Projection projection)

cdef class SparseVectorChart:
    cdef CSparseVectorChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_MinSparseVector "count_constrained_viterbi<MinSparseVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphMinSparseVectorPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CMinSparseVectorMarginals "Marginals<MinSparseVectorPotential>":
        vector[pair[int, int]] marginal(const CHyperedge *edge)
        vector[pair[int, int]] marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const vector[pair[int, int]] &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CMinSparseVectorChart "Chart<MinSparseVectorPotential>":
        CMinSparseVectorChart(const CHypergraph *graph)
        vector[pair[int, int]] get(const CHypernode *node)
        void insert(const CHypernode& node, const vector[pair[int, int]]& val)

    cdef cppclass CMinSparseVectorDynamicViterbi "DynamicViterbi<MinSparseVectorPotential>":
        CMinSparseVectorDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphMinSparseVectorPotentials theta)
        void update(const CHypergraphMinSparseVectorPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<MinSparseVectorPotential>":
    CMinSparseVectorMarginals *MinSparseVector_compute "Marginals<MinSparseVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphMinSparseVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass MinSparseVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphMinSparseVectorPotentials "HypergraphPotentials<MinSparseVectorPotential>":
        vector[pair[int, int]] dot(const CHyperpath &path) except +
        vector[pair[int, int]] score(const CHyperedge *edge)
        CHypergraphMinSparseVectorPotentials *times(
            const CHypergraphMinSparseVectorPotentials &potentials)
        CHypergraphMinSparseVectorPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphMinSparseVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[pair[int, int]]] potentials,
            vector[pair[int, int]] bias) except +
        vector[pair[int, int]] bias()
        CHypergraphMinSparseVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<MinSparseVectorPotential>":
    CHypergraphMinSparseVectorPotentials *cmake_potentials_MinSparseVector "HypergraphMapPotentials<MinSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<MinSparseVectorPotential>":
    CHypergraphMinSparseVectorPotentials *cmake_potentials_MinSparseVector "HypergraphVectorPotentials<MinSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<MinSparseVectorPotential>":
    CHypergraphMinSparseVectorPotentials *cmake_projected_potentials_MinSparseVector "HypergraphProjectedPotentials<MinSparseVectorPotential>::make_potentials" (
        CHypergraphMinSparseVectorPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "MinSparseVectorPotential":
    vector[pair[int, int]] MinSparseVector_one "MinSparseVectorPotential::one" ()
    vector[pair[int, int]] MinSparseVector_zero "MinSparseVectorPotential::zero" ()
    vector[pair[int, int]] MinSparseVector_add "MinSparseVectorPotential::add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_times "MinSparseVectorPotential::times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_safeadd "MinSparseVectorPotential::safe_add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_safetimes "MinSparseVectorPotential::safe_times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MinSparseVector_normalize "MinSparseVectorPotential::normalize" (vector[pair[int, int]]&)


cdef class MinSparseVectorPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphMinSparseVectorPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphMinSparseVectorPotentials *ptr,
              Projection projection)

cdef class MinSparseVectorChart:
    cdef CMinSparseVectorChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_MaxSparseVector "count_constrained_viterbi<MaxSparseVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphMaxSparseVectorPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CMaxSparseVectorMarginals "Marginals<MaxSparseVectorPotential>":
        vector[pair[int, int]] marginal(const CHyperedge *edge)
        vector[pair[int, int]] marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const vector[pair[int, int]] &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CMaxSparseVectorChart "Chart<MaxSparseVectorPotential>":
        CMaxSparseVectorChart(const CHypergraph *graph)
        vector[pair[int, int]] get(const CHypernode *node)
        void insert(const CHypernode& node, const vector[pair[int, int]]& val)

    cdef cppclass CMaxSparseVectorDynamicViterbi "DynamicViterbi<MaxSparseVectorPotential>":
        CMaxSparseVectorDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphMaxSparseVectorPotentials theta)
        void update(const CHypergraphMaxSparseVectorPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<MaxSparseVectorPotential>":
    CMaxSparseVectorMarginals *MaxSparseVector_compute "Marginals<MaxSparseVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphMaxSparseVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass MaxSparseVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphMaxSparseVectorPotentials "HypergraphPotentials<MaxSparseVectorPotential>":
        vector[pair[int, int]] dot(const CHyperpath &path) except +
        vector[pair[int, int]] score(const CHyperedge *edge)
        CHypergraphMaxSparseVectorPotentials *times(
            const CHypergraphMaxSparseVectorPotentials &potentials)
        CHypergraphMaxSparseVectorPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphMaxSparseVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[vector[pair[int, int]]] potentials,
            vector[pair[int, int]] bias) except +
        vector[pair[int, int]] bias()
        CHypergraphMaxSparseVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<MaxSparseVectorPotential>":
    CHypergraphMaxSparseVectorPotentials *cmake_potentials_MaxSparseVector "HypergraphMapPotentials<MaxSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<MaxSparseVectorPotential>":
    CHypergraphMaxSparseVectorPotentials *cmake_potentials_MaxSparseVector "HypergraphVectorPotentials<MaxSparseVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[vector[pair[int, int]]] potentials,
        vector[pair[int, int]] bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<MaxSparseVectorPotential>":
    CHypergraphMaxSparseVectorPotentials *cmake_projected_potentials_MaxSparseVector "HypergraphProjectedPotentials<MaxSparseVectorPotential>::make_potentials" (
        CHypergraphMaxSparseVectorPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "MaxSparseVectorPotential":
    vector[pair[int, int]] MaxSparseVector_one "MaxSparseVectorPotential::one" ()
    vector[pair[int, int]] MaxSparseVector_zero "MaxSparseVectorPotential::zero" ()
    vector[pair[int, int]] MaxSparseVector_add "MaxSparseVectorPotential::add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_times "MaxSparseVectorPotential::times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_safeadd "MaxSparseVectorPotential::safe_add" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_safetimes "MaxSparseVectorPotential::safe_times" (vector[pair[int, int]], const vector[pair[int, int]]&)
    vector[pair[int, int]] MaxSparseVector_normalize "MaxSparseVectorPotential::normalize" (vector[pair[int, int]]&)


cdef class MaxSparseVectorPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphMaxSparseVectorPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphMaxSparseVectorPotentials *ptr,
              Projection projection)

cdef class MaxSparseVectorChart:
    cdef CMaxSparseVectorChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_BinaryVector "count_constrained_viterbi<BinaryVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphBinaryVectorPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CBinaryVectorMarginals "Marginals<BinaryVectorPotential>":
        cbitset marginal(const CHyperedge *edge)
        cbitset marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const cbitset &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CBinaryVectorChart "Chart<BinaryVectorPotential>":
        CBinaryVectorChart(const CHypergraph *graph)
        cbitset get(const CHypernode *node)
        void insert(const CHypernode& node, const cbitset& val)

    cdef cppclass CBinaryVectorDynamicViterbi "DynamicViterbi<BinaryVectorPotential>":
        CBinaryVectorDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphBinaryVectorPotentials theta)
        void update(const CHypergraphBinaryVectorPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<BinaryVectorPotential>":
    CBinaryVectorMarginals *BinaryVector_compute "Marginals<BinaryVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphBinaryVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass BinaryVectorPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphBinaryVectorPotentials "HypergraphPotentials<BinaryVectorPotential>":
        cbitset dot(const CHyperpath &path) except +
        cbitset score(const CHyperedge *edge)
        CHypergraphBinaryVectorPotentials *times(
            const CHypergraphBinaryVectorPotentials &potentials)
        CHypergraphBinaryVectorPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphBinaryVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[cbitset] potentials,
            cbitset bias) except +
        cbitset bias()
        CHypergraphBinaryVectorPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_potentials_BinaryVector "HypergraphMapPotentials<BinaryVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[cbitset] potentials,
        cbitset bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_potentials_BinaryVector "HypergraphVectorPotentials<BinaryVectorPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[cbitset] potentials,
        cbitset bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<BinaryVectorPotential>":
    CHypergraphBinaryVectorPotentials *cmake_projected_potentials_BinaryVector "HypergraphProjectedPotentials<BinaryVectorPotential>::make_potentials" (
        CHypergraphBinaryVectorPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "BinaryVectorPotential":
    cbitset BinaryVector_one "BinaryVectorPotential::one" ()
    cbitset BinaryVector_zero "BinaryVectorPotential::zero" ()
    cbitset BinaryVector_add "BinaryVectorPotential::add" (cbitset, const cbitset&)
    cbitset BinaryVector_times "BinaryVectorPotential::times" (cbitset, const cbitset&)
    cbitset BinaryVector_safeadd "BinaryVectorPotential::safe_add" (cbitset, const cbitset&)
    cbitset BinaryVector_safetimes "BinaryVectorPotential::safe_times" (cbitset, const cbitset&)
    cbitset BinaryVector_normalize "BinaryVectorPotential::normalize" (cbitset&)


cdef class BinaryVectorPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphBinaryVectorPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphBinaryVectorPotentials *ptr,
              Projection projection)

cdef class BinaryVectorChart:
    cdef CBinaryVectorChart *chart
    cdef kind





# Type identifiers.

cdef extern from "Hypergraph/Algorithms.h":
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

    CHyperpath *count_constrained_viterbi_Counting "count_constrained_viterbi<CountingPotential>"(
        const CHypergraph *graph,
        const CHypergraphCountingPotentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass CCountingMarginals "Marginals<CountingPotential>":
        int marginal(const CHyperedge *edge)
        int marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const int &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CCountingChart "Chart<CountingPotential>":
        CCountingChart(const CHypergraph *graph)
        int get(const CHypernode *node)
        void insert(const CHypernode& node, const int& val)

    cdef cppclass CCountingDynamicViterbi "DynamicViterbi<CountingPotential>":
        CCountingDynamicViterbi(const CHypergraph *graph)
        void initialize(const CHypergraphCountingPotentials theta)
        void update(const CHypergraphCountingPotentials theta,
                    set[int] *update)
        const CBackPointers *back_pointers()


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<CountingPotential>":
    CCountingMarginals *Counting_compute "Marginals<CountingPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphCountingPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass CountingPotential:
        pass


cdef extern from "Hypergraph/Potentials.h":
    cdef cppclass CHypergraphCountingPotentials "HypergraphPotentials<CountingPotential>":
        int dot(const CHyperpath &path) except +
        int score(const CHyperedge *edge)
        CHypergraphCountingPotentials *times(
            const CHypergraphCountingPotentials &potentials)
        CHypergraphCountingPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphCountingPotentials(
            const CHypergraph *hypergraph,
            const vector[int] potentials,
            int bias) except +
        int bias()
        CHypergraphCountingPotentials *clone() const

cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphMapPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_potentials_Counting "HypergraphMapPotentials<CountingPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const c_map.map[int, int] map_potentials,
        const vector[int] potentials,
        int bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphVectorPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_potentials_Counting "HypergraphVectorPotentials<CountingPotential>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[int] potentials,
        int bias) except +


cdef extern from "Hypergraph/Potentials.h" namespace "HypergraphProjectedPotentials<CountingPotential>":
    CHypergraphCountingPotentials *cmake_projected_potentials_Counting "HypergraphProjectedPotentials<CountingPotential>::make_potentials" (
        CHypergraphCountingPotentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "CountingPotential":
    int Counting_one "CountingPotential::one" ()
    int Counting_zero "CountingPotential::zero" ()
    int Counting_add "CountingPotential::add" (int, const int&)
    int Counting_times "CountingPotential::times" (int, const int&)
    int Counting_safeadd "CountingPotential::safe_add" (int, const int&)
    int Counting_safetimes "CountingPotential::safe_times" (int, const int&)
    int Counting_normalize "CountingPotential::normalize" (int&)


cdef class CountingPotentials:
    cdef Hypergraph hypergraph
    cdef CHypergraphCountingPotentials *thisptr
    cdef Projection projection
    cdef kind

    cdef init(self, CHypergraphCountingPotentials *ptr,
              Projection projection)

cdef class CountingChart:
    cdef CCountingChart *chart
    cdef kind





cdef class LogViterbiDynamicViterbi:
    cdef CLogViterbiDynamicViterbi *thisptr
    cdef Hypergraph graph

#

cdef class Projection:
    cdef const CHypergraphProjection *thisptr
    cdef Hypergraph small_graph
    cdef Hypergraph big_graph

    cdef Projection init(self, const CHypergraphProjection *thisptr,
                         Hypergraph small_graph)




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
