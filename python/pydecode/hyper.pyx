#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

include "wrap.pxd"
include "hypergraph.pyx"



############# This is the templated semiring part. ##############



cdef extern from "Hypergraph/Algorithms.h":
    CViterbiChart *inside_Viterbi "general_inside<ViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta) except +

    CViterbiChart *outside_Viterbi "general_outside<ViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta,
        CViterbiChart inside_chart) except +

    CHyperpath *viterbi_Viterbi"general_viterbi<ViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphViterbiPotentials theta) except +

    cdef cppclass CViterbiMarginals "Marginals<ViterbiPotential>":
        ViterbiPotential marginal(const CHyperedge *edge)
        ViterbiPotential marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const ViterbiPotential &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CViterbiChart "Chart<ViterbiPotential>":
        ViterbiPotential get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<ViterbiPotential>":
    CViterbiMarginals *Viterbi_compute "Marginals<ViterbiPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphViterbiPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass ViterbiPotential:
        ViterbiPotential()
        ViterbiPotential(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "ViterbiPotential":
    ViterbiPotential Viterbi_one "ViterbiPotential::one" ()
    ViterbiPotential Viterbi_zero "ViterbiPotential::zero" ()
    ViterbiPotential Viterbi_add "ViterbiPotential::add" (ViterbiPotential, const ViterbiPotential)
    ViterbiPotential Viterbi_times "ViterbiPotential::times" (ViterbiPotential, const ViterbiPotential)


cdef extern from "Hypergraph/Algorithms.h" namespace "ViterbiPotential":
    cdef cppclass CHypergraphViterbiPotentials "HypergraphPotentials<ViterbiPotential>":
        ViterbiPotential dot(const CHyperpath &path) except +
        ViterbiPotential score(const CHyperedge *edge)
        CHypergraphViterbiPotentials *times(
            const CHypergraphViterbiPotentials &potentials)
        CHypergraphViterbiPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[ViterbiPotential] potentials,
            ViterbiPotential bias) except +


cdef class ViterbiPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphViterbiPotentials *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Viterbi

    def times(self, ViterbiPotentials other):
        cdef const CHypergraphViterbiPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return ViterbiPotentials(self.hypergraph).init(new_potentials)

    def project(self, Hypergraph graph, Projection projection):
        cdef ViterbiPotentials new_potentials = ViterbiPotentials(graph)
        cdef const CHypergraphViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return new_potentials.init(ptr)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef ViterbiPotential my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = ViterbiPotential(<double> bias)

        cdef vector[ViterbiPotential] potentials = \
             vector[ViterbiPotential](self.hypergraph.thisptr.edges().size(),
             Viterbi_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = Viterbi_zero()
            potentials[i] = ViterbiPotential(<double> result)
        self.thisptr =  \
          new CHypergraphViterbiPotentials(self.hypergraph.thisptr,
                                           potentials, my_bias)
        return self

    cdef init(self, const CHypergraphViterbiPotentials *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return _ViterbiW().init(self.thisptr.score(edge.edgeptr)).value

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _ViterbiW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _ViterbiW:
    cdef ViterbiPotential wrap

    def __cinit__(self, val=None):
        if val is not None:
            self.init(ViterbiPotential(<double>val))

    cdef init(self, ViterbiPotential wrap):
        self.wrap = wrap
        return self

    
    def __float__(self):
        return <float>self.wrap
    

    

    property value:
        def __get__(self):
            
            return <float>self.wrap
            

    def __repr__(self):
        return str(self.value)

    def __add__(_ViterbiW self, _ViterbiW other):
        return _ViterbiW().init(
            Viterbi_add(self.wrap, other.wrap))

    def __mul__(_ViterbiW self, _ViterbiW other):
        return _ViterbiW().init(
            Viterbi_times(self.wrap, other.wrap))

    @staticmethod
    def one():
        return _ViterbiW().init(Viterbi_one())

    @staticmethod
    def zero():
        return _ViterbiW().init(Viterbi_zero())

    def __cmp__(_ViterbiW self, _ViterbiW other):
        return cmp(self.value, other.value)

cdef class _ViterbiChart:
    cdef CViterbiChart *chart
    cdef kind

    def __init__(self):
        self.kind = Viterbi

    def __getitem__(self, Node node):
        return _ViterbiW().init(self.chart.get(node.nodeptr))

cdef class _ViterbiMarginals:
    cdef const CViterbiMarginals *thisptr

    cdef init(self, const CViterbiMarginals *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _ViterbiW().init(self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _ViterbiW().init(self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Viterbi marginal values." + \
                "Passed %s."%obj)
    
    def threshold(self, _ViterbiW semi):
        return BoolPotentials(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    

class Viterbi:
    Chart = _ViterbiChart
    Marginals = _ViterbiMarginals
    Semi = _ViterbiW
    Potentials = ViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               ViterbiPotentials potentials):
        cdef _ViterbiChart chart = _ViterbiChart()
        chart.chart = inside_Viterbi(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                ViterbiPotentials potentials,
                _ViterbiChart inside_chart):
        cdef _ViterbiChart out_chart = _ViterbiChart()
        out_chart.chart = outside_Viterbi(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    
    @staticmethod
    def viterbi(Hypergraph graph,
                ViterbiPotentials potentials):
        cdef CHyperpath *path = \
            viterbi_Viterbi(graph.thisptr,
                               deref(potentials.thisptr))
        return Path().init(path)
    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          ViterbiPotentials potentials):
        cdef const CViterbiMarginals *marginals = \
            Viterbi_compute(graph.thisptr, potentials.thisptr)
        return _ViterbiMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         ViterbiPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(_ViterbiW().init(ViterbiPotential(<double>threshold)))
        projection = Projection(graph, bool_potentials)
        new_graph = projection.project(graph)
        new_potential = potentials.project(new_graph, projection)
        return new_graph, new_potential







cdef extern from "Hypergraph/Algorithms.h":
    CLogViterbiChart *inside_LogViterbi "general_inside<LogViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta) except +

    CLogViterbiChart *outside_LogViterbi "general_outside<LogViterbiPotential>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta,
        CLogViterbiChart inside_chart) except +

    CHyperpath *viterbi_LogViterbi"general_viterbi<LogViterbiPotential>"(
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials theta) except +

    cdef cppclass CLogViterbiMarginals "Marginals<LogViterbiPotential>":
        LogViterbiPotential marginal(const CHyperedge *edge)
        LogViterbiPotential marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const LogViterbiPotential &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CLogViterbiChart "Chart<LogViterbiPotential>":
        LogViterbiPotential get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<LogViterbiPotential>":
    CLogViterbiMarginals *LogViterbi_compute "Marginals<LogViterbiPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphLogViterbiPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass LogViterbiPotential:
        LogViterbiPotential()
        LogViterbiPotential(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "LogViterbiPotential":
    LogViterbiPotential LogViterbi_one "LogViterbiPotential::one" ()
    LogViterbiPotential LogViterbi_zero "LogViterbiPotential::zero" ()
    LogViterbiPotential LogViterbi_add "LogViterbiPotential::add" (LogViterbiPotential, const LogViterbiPotential)
    LogViterbiPotential LogViterbi_times "LogViterbiPotential::times" (LogViterbiPotential, const LogViterbiPotential)


cdef extern from "Hypergraph/Algorithms.h" namespace "LogViterbiPotential":
    cdef cppclass CHypergraphLogViterbiPotentials "HypergraphPotentials<LogViterbiPotential>":
        LogViterbiPotential dot(const CHyperpath &path) except +
        LogViterbiPotential score(const CHyperedge *edge)
        CHypergraphLogViterbiPotentials *times(
            const CHypergraphLogViterbiPotentials &potentials)
        CHypergraphLogViterbiPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphLogViterbiPotentials(
            const CHypergraph *hypergraph,
            const vector[LogViterbiPotential] potentials,
            LogViterbiPotential bias) except +


cdef class LogViterbiPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphLogViterbiPotentials *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = LogViterbi

    def times(self, LogViterbiPotentials other):
        cdef const CHypergraphLogViterbiPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return LogViterbiPotentials(self.hypergraph).init(new_potentials)

    def project(self, Hypergraph graph, Projection projection):
        cdef LogViterbiPotentials new_potentials = LogViterbiPotentials(graph)
        cdef const CHypergraphLogViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return new_potentials.init(ptr)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef LogViterbiPotential my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = LogViterbiPotential(<double> bias)

        cdef vector[LogViterbiPotential] potentials = \
             vector[LogViterbiPotential](self.hypergraph.thisptr.edges().size(),
             LogViterbi_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = LogViterbi_zero()
            potentials[i] = LogViterbiPotential(<double> result)
        self.thisptr =  \
          new CHypergraphLogViterbiPotentials(self.hypergraph.thisptr,
                                           potentials, my_bias)
        return self

    cdef init(self, const CHypergraphLogViterbiPotentials *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return _LogViterbiW().init(self.thisptr.score(edge.edgeptr)).value

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _LogViterbiW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _LogViterbiW:
    cdef LogViterbiPotential wrap

    def __cinit__(self, val=None):
        if val is not None:
            self.init(LogViterbiPotential(<double>val))

    cdef init(self, LogViterbiPotential wrap):
        self.wrap = wrap
        return self

    
    def __float__(self):
        return <float>self.wrap
    

    

    property value:
        def __get__(self):
            
            return <float>self.wrap
            

    def __repr__(self):
        return str(self.value)

    def __add__(_LogViterbiW self, _LogViterbiW other):
        return _LogViterbiW().init(
            LogViterbi_add(self.wrap, other.wrap))

    def __mul__(_LogViterbiW self, _LogViterbiW other):
        return _LogViterbiW().init(
            LogViterbi_times(self.wrap, other.wrap))

    @staticmethod
    def one():
        return _LogViterbiW().init(LogViterbi_one())

    @staticmethod
    def zero():
        return _LogViterbiW().init(LogViterbi_zero())

    def __cmp__(_LogViterbiW self, _LogViterbiW other):
        return cmp(self.value, other.value)

cdef class _LogViterbiChart:
    cdef CLogViterbiChart *chart
    cdef kind

    def __init__(self):
        self.kind = LogViterbi

    def __getitem__(self, Node node):
        return _LogViterbiW().init(self.chart.get(node.nodeptr))

cdef class _LogViterbiMarginals:
    cdef const CLogViterbiMarginals *thisptr

    cdef init(self, const CLogViterbiMarginals *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _LogViterbiW().init(self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _LogViterbiW().init(self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have LogViterbi marginal values." + \
                "Passed %s."%obj)
    
    def threshold(self, _LogViterbiW semi):
        return BoolPotentials(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    

class LogViterbi:
    Chart = _LogViterbiChart
    Marginals = _LogViterbiMarginals
    Semi = _LogViterbiW
    Potentials = LogViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               LogViterbiPotentials potentials):
        cdef _LogViterbiChart chart = _LogViterbiChart()
        chart.chart = inside_LogViterbi(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                LogViterbiPotentials potentials,
                _LogViterbiChart inside_chart):
        cdef _LogViterbiChart out_chart = _LogViterbiChart()
        out_chart.chart = outside_LogViterbi(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    
    @staticmethod
    def viterbi(Hypergraph graph,
                LogViterbiPotentials potentials):
        cdef CHyperpath *path = \
            viterbi_LogViterbi(graph.thisptr,
                               deref(potentials.thisptr))
        return Path().init(path)
    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          LogViterbiPotentials potentials):
        cdef const CLogViterbiMarginals *marginals = \
            LogViterbi_compute(graph.thisptr, potentials.thisptr)
        return _LogViterbiMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         LogViterbiPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(_LogViterbiW().init(LogViterbiPotential(<double>threshold)))
        projection = Projection(graph, bool_potentials)
        new_graph = projection.project(graph)
        new_potential = potentials.project(new_graph, projection)
        return new_graph, new_potential







cdef extern from "Hypergraph/Algorithms.h":
    CInsideChart *inside_Inside "general_inside<InsidePotential>" (
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta) except +

    CInsideChart *outside_Inside "general_outside<InsidePotential>" (
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta,
        CInsideChart inside_chart) except +

    CHyperpath *viterbi_Inside"general_viterbi<InsidePotential>"(
        const CHypergraph *graph,
        const CHypergraphInsidePotentials theta) except +

    cdef cppclass CInsideMarginals "Marginals<InsidePotential>":
        InsidePotential marginal(const CHyperedge *edge)
        InsidePotential marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const InsidePotential &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CInsideChart "Chart<InsidePotential>":
        InsidePotential get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<InsidePotential>":
    CInsideMarginals *Inside_compute "Marginals<InsidePotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphInsidePotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass InsidePotential:
        InsidePotential()
        InsidePotential(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "InsidePotential":
    InsidePotential Inside_one "InsidePotential::one" ()
    InsidePotential Inside_zero "InsidePotential::zero" ()
    InsidePotential Inside_add "InsidePotential::add" (InsidePotential, const InsidePotential)
    InsidePotential Inside_times "InsidePotential::times" (InsidePotential, const InsidePotential)


cdef extern from "Hypergraph/Algorithms.h" namespace "InsidePotential":
    cdef cppclass CHypergraphInsidePotentials "HypergraphPotentials<InsidePotential>":
        InsidePotential dot(const CHyperpath &path) except +
        InsidePotential score(const CHyperedge *edge)
        CHypergraphInsidePotentials *times(
            const CHypergraphInsidePotentials &potentials)
        CHypergraphInsidePotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphInsidePotentials(
            const CHypergraph *hypergraph,
            const vector[InsidePotential] potentials,
            InsidePotential bias) except +


cdef class InsidePotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphInsidePotentials *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Inside

    def times(self, InsidePotentials other):
        cdef const CHypergraphInsidePotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return InsidePotentials(self.hypergraph).init(new_potentials)

    def project(self, Hypergraph graph, Projection projection):
        cdef InsidePotentials new_potentials = InsidePotentials(graph)
        cdef const CHypergraphInsidePotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return new_potentials.init(ptr)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef InsidePotential my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = InsidePotential(<double> bias)

        cdef vector[InsidePotential] potentials = \
             vector[InsidePotential](self.hypergraph.thisptr.edges().size(),
             Inside_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = Inside_zero()
            potentials[i] = InsidePotential(<double> result)
        self.thisptr =  \
          new CHypergraphInsidePotentials(self.hypergraph.thisptr,
                                           potentials, my_bias)
        return self

    cdef init(self, const CHypergraphInsidePotentials *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return _InsideW().init(self.thisptr.score(edge.edgeptr)).value

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _InsideW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _InsideW:
    cdef InsidePotential wrap

    def __cinit__(self, val=None):
        if val is not None:
            self.init(InsidePotential(<double>val))

    cdef init(self, InsidePotential wrap):
        self.wrap = wrap
        return self

    
    def __float__(self):
        return <float>self.wrap
    

    

    property value:
        def __get__(self):
            
            return <float>self.wrap
            

    def __repr__(self):
        return str(self.value)

    def __add__(_InsideW self, _InsideW other):
        return _InsideW().init(
            Inside_add(self.wrap, other.wrap))

    def __mul__(_InsideW self, _InsideW other):
        return _InsideW().init(
            Inside_times(self.wrap, other.wrap))

    @staticmethod
    def one():
        return _InsideW().init(Inside_one())

    @staticmethod
    def zero():
        return _InsideW().init(Inside_zero())

    def __cmp__(_InsideW self, _InsideW other):
        return cmp(self.value, other.value)

cdef class _InsideChart:
    cdef CInsideChart *chart
    cdef kind

    def __init__(self):
        self.kind = Inside

    def __getitem__(self, Node node):
        return _InsideW().init(self.chart.get(node.nodeptr))

cdef class _InsideMarginals:
    cdef const CInsideMarginals *thisptr

    cdef init(self, const CInsideMarginals *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _InsideW().init(self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _InsideW().init(self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Inside marginal values." + \
                "Passed %s."%obj)
    
    def threshold(self, _InsideW semi):
        return BoolPotentials(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    

class Inside:
    Chart = _InsideChart
    Marginals = _InsideMarginals
    Semi = _InsideW
    Potentials = InsidePotentials

    @staticmethod
    def inside(Hypergraph graph,
               InsidePotentials potentials):
        cdef _InsideChart chart = _InsideChart()
        chart.chart = inside_Inside(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                InsidePotentials potentials,
                _InsideChart inside_chart):
        cdef _InsideChart out_chart = _InsideChart()
        out_chart.chart = outside_Inside(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    
    @staticmethod
    def viterbi(Hypergraph graph,
                InsidePotentials potentials):
        cdef CHyperpath *path = \
            viterbi_Inside(graph.thisptr,
                               deref(potentials.thisptr))
        return Path().init(path)
    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          InsidePotentials potentials):
        cdef const CInsideMarginals *marginals = \
            Inside_compute(graph.thisptr, potentials.thisptr)
        return _InsideMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         InsidePotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(_InsideW().init(InsidePotential(<double>threshold)))
        projection = Projection(graph, bool_potentials)
        new_graph = projection.project(graph)
        new_potential = potentials.project(new_graph, projection)
        return new_graph, new_potential







cdef extern from "Hypergraph/Algorithms.h":
    CBoolChart *inside_Bool "general_inside<BoolPotential>" (
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta) except +

    CBoolChart *outside_Bool "general_outside<BoolPotential>" (
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta,
        CBoolChart inside_chart) except +

    CHyperpath *viterbi_Bool"general_viterbi<BoolPotential>"(
        const CHypergraph *graph,
        const CHypergraphBoolPotentials theta) except +

    cdef cppclass CBoolMarginals "Marginals<BoolPotential>":
        BoolPotential marginal(const CHyperedge *edge)
        BoolPotential marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const BoolPotential &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CBoolChart "Chart<BoolPotential>":
        BoolPotential get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<BoolPotential>":
    CBoolMarginals *Bool_compute "Marginals<BoolPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphBoolPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass BoolPotential:
        BoolPotential()
        BoolPotential(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "BoolPotential":
    BoolPotential Bool_one "BoolPotential::one" ()
    BoolPotential Bool_zero "BoolPotential::zero" ()
    BoolPotential Bool_add "BoolPotential::add" (BoolPotential, const BoolPotential)
    BoolPotential Bool_times "BoolPotential::times" (BoolPotential, const BoolPotential)


cdef extern from "Hypergraph/Algorithms.h" namespace "BoolPotential":
    cdef cppclass CHypergraphBoolPotentials "HypergraphPotentials<BoolPotential>":
        BoolPotential dot(const CHyperpath &path) except +
        BoolPotential score(const CHyperedge *edge)
        CHypergraphBoolPotentials *times(
            const CHypergraphBoolPotentials &potentials)
        CHypergraphBoolPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphBoolPotentials(
            const CHypergraph *hypergraph,
            const vector[BoolPotential] potentials,
            BoolPotential bias) except +


cdef class BoolPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphBoolPotentials *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Bool

    def times(self, BoolPotentials other):
        cdef const CHypergraphBoolPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return BoolPotentials(self.hypergraph).init(new_potentials)

    def project(self, Hypergraph graph, Projection projection):
        cdef BoolPotentials new_potentials = BoolPotentials(graph)
        cdef const CHypergraphBoolPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return new_potentials.init(ptr)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef BoolPotential my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = BoolPotential(<double> bias)

        cdef vector[BoolPotential] potentials = \
             vector[BoolPotential](self.hypergraph.thisptr.edges().size(),
             Bool_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = Bool_zero()
            potentials[i] = BoolPotential(<double> result)
        self.thisptr =  \
          new CHypergraphBoolPotentials(self.hypergraph.thisptr,
                                           potentials, my_bias)
        return self

    cdef init(self, const CHypergraphBoolPotentials *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return _BoolW().init(self.thisptr.score(edge.edgeptr)).value

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _BoolW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _BoolW:
    cdef BoolPotential wrap

    def __cinit__(self, val=None):
        if val is not None:
            self.init(BoolPotential(<double>val))

    cdef init(self, BoolPotential wrap):
        self.wrap = wrap
        return self

    

    
    def __bool__(self):
        return <bool>self.wrap
    

    property value:
        def __get__(self):
            
            return <bool>self.wrap
            

    def __repr__(self):
        return str(self.value)

    def __add__(_BoolW self, _BoolW other):
        return _BoolW().init(
            Bool_add(self.wrap, other.wrap))

    def __mul__(_BoolW self, _BoolW other):
        return _BoolW().init(
            Bool_times(self.wrap, other.wrap))

    @staticmethod
    def one():
        return _BoolW().init(Bool_one())

    @staticmethod
    def zero():
        return _BoolW().init(Bool_zero())

    def __cmp__(_BoolW self, _BoolW other):
        return cmp(self.value, other.value)

cdef class _BoolChart:
    cdef CBoolChart *chart
    cdef kind

    def __init__(self):
        self.kind = Bool

    def __getitem__(self, Node node):
        return _BoolW().init(self.chart.get(node.nodeptr))

cdef class _BoolMarginals:
    cdef const CBoolMarginals *thisptr

    cdef init(self, const CBoolMarginals *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _BoolW().init(self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _BoolW().init(self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have Bool marginal values." + \
                "Passed %s."%obj)
    
    def threshold(self, _BoolW semi):
        return BoolPotentials(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    

class Bool:
    Chart = _BoolChart
    Marginals = _BoolMarginals
    Semi = _BoolW
    Potentials = BoolPotentials

    @staticmethod
    def inside(Hypergraph graph,
               BoolPotentials potentials):
        cdef _BoolChart chart = _BoolChart()
        chart.chart = inside_Bool(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                BoolPotentials potentials,
                _BoolChart inside_chart):
        cdef _BoolChart out_chart = _BoolChart()
        out_chart.chart = outside_Bool(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    
    @staticmethod
    def viterbi(Hypergraph graph,
                BoolPotentials potentials):
        cdef CHyperpath *path = \
            viterbi_Bool(graph.thisptr,
                               deref(potentials.thisptr))
        return Path().init(path)
    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          BoolPotentials potentials):
        cdef const CBoolMarginals *marginals = \
            Bool_compute(graph.thisptr, potentials.thisptr)
        return _BoolMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         BoolPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(_BoolW().init(BoolPotential(<double>threshold)))
        projection = Projection(graph, bool_potentials)
        new_graph = projection.project(graph)
        new_potential = potentials.project(new_graph, projection)
        return new_graph, new_potential







cdef extern from "Hypergraph/Algorithms.h":
    CSparseVectorChart *inside_SparseVector "general_inside<SparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials theta) except +

    CSparseVectorChart *outside_SparseVector "general_outside<SparseVectorPotential>" (
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials theta,
        CSparseVectorChart inside_chart) except +

    CHyperpath *viterbi_SparseVector"general_viterbi<SparseVectorPotential>"(
        const CHypergraph *graph,
        const CHypergraphSparseVectorPotentials theta) except +

    cdef cppclass CSparseVectorMarginals "Marginals<SparseVectorPotential>":
        SparseVectorPotential marginal(const CHyperedge *edge)
        SparseVectorPotential marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const SparseVectorPotential &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CSparseVectorChart "Chart<SparseVectorPotential>":
        SparseVectorPotential get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<SparseVectorPotential>":
    CSparseVectorMarginals *SparseVector_compute "Marginals<SparseVectorPotential>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphSparseVectorPotentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass SparseVectorPotential:
        SparseVectorPotential()
        SparseVectorPotential(vector[pair[int, int]])
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "SparseVectorPotential":
    SparseVectorPotential SparseVector_one "SparseVectorPotential::one" ()
    SparseVectorPotential SparseVector_zero "SparseVectorPotential::zero" ()
    SparseVectorPotential SparseVector_add "SparseVectorPotential::add" (SparseVectorPotential, const SparseVectorPotential)
    SparseVectorPotential SparseVector_times "SparseVectorPotential::times" (SparseVectorPotential, const SparseVectorPotential)


cdef extern from "Hypergraph/Algorithms.h" namespace "SparseVectorPotential":
    cdef cppclass CHypergraphSparseVectorPotentials "HypergraphPotentials<SparseVectorPotential>":
        SparseVectorPotential dot(const CHyperpath &path) except +
        SparseVectorPotential score(const CHyperedge *edge)
        CHypergraphSparseVectorPotentials *times(
            const CHypergraphSparseVectorPotentials &potentials)
        CHypergraphSparseVectorPotentials *project_potentials(
            const CHypergraphProjection)
        CHypergraphSparseVectorPotentials(
            const CHypergraph *hypergraph,
            const vector[SparseVectorPotential] potentials,
            SparseVectorPotential bias) except +


cdef class SparseVectorPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphSparseVectorPotentials *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = SparseVector

    def times(self, SparseVectorPotentials other):
        cdef const CHypergraphSparseVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return SparseVectorPotentials(self.hypergraph).init(new_potentials)

    def project(self, Hypergraph graph, Projection projection):
        cdef SparseVectorPotentials new_potentials = SparseVectorPotentials(graph)
        cdef const CHypergraphSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return new_potentials.init(ptr)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef SparseVectorPotential my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = SparseVectorPotential(<vector[pair[int, int]]> bias)

        cdef vector[SparseVectorPotential] potentials = \
             vector[SparseVectorPotential](self.hypergraph.thisptr.edges().size(),
             SparseVector_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = SparseVector_zero()
            potentials[i] = SparseVectorPotential(<vector[pair[int, int]]> result)
        self.thisptr =  \
          new CHypergraphSparseVectorPotentials(self.hypergraph.thisptr,
                                           potentials, my_bias)
        return self

    cdef init(self, const CHypergraphSparseVectorPotentials *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return _SparseVectorW().init(self.thisptr.score(edge.edgeptr)).value

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _SparseVectorW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _SparseVectorW:
    cdef SparseVectorPotential wrap

    def __cinit__(self, val=None):
        if val is not None:
            self.init(SparseVectorPotential(<vector[pair[int, int]]>val))

    cdef init(self, SparseVectorPotential wrap):
        self.wrap = wrap
        return self

    

    

    property value:
        def __get__(self):
            
            
            d = {}
            cdef vector[pair[int,int]] s= <vector[pair[int,int]]> self.wrap
            for p in s:
                d[p.first] = p.second
            return d

            

    def __repr__(self):
        return str(self.value)

    def __add__(_SparseVectorW self, _SparseVectorW other):
        return _SparseVectorW().init(
            SparseVector_add(self.wrap, other.wrap))

    def __mul__(_SparseVectorW self, _SparseVectorW other):
        return _SparseVectorW().init(
            SparseVector_times(self.wrap, other.wrap))

    @staticmethod
    def one():
        return _SparseVectorW().init(SparseVector_one())

    @staticmethod
    def zero():
        return _SparseVectorW().init(SparseVector_zero())

    def __cmp__(_SparseVectorW self, _SparseVectorW other):
        return cmp(self.value, other.value)

cdef class _SparseVectorChart:
    cdef CSparseVectorChart *chart
    cdef kind

    def __init__(self):
        self.kind = SparseVector

    def __getitem__(self, Node node):
        return _SparseVectorW().init(self.chart.get(node.nodeptr))

cdef class _SparseVectorMarginals:
    cdef const CSparseVectorMarginals *thisptr

    cdef init(self, const CSparseVectorMarginals *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _SparseVectorW().init(self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _SparseVectorW().init(self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have SparseVector marginal values." + \
                "Passed %s."%obj)
    

class SparseVector:
    Chart = _SparseVectorChart
    Marginals = _SparseVectorMarginals
    Semi = _SparseVectorW
    Potentials = SparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               SparseVectorPotentials potentials):
        cdef _SparseVectorChart chart = _SparseVectorChart()
        chart.chart = inside_SparseVector(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                SparseVectorPotentials potentials,
                _SparseVectorChart inside_chart):
        cdef _SparseVectorChart out_chart = _SparseVectorChart()
        out_chart.chart = outside_SparseVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          SparseVectorPotentials potentials):
        cdef const CSparseVectorMarginals *marginals = \
            SparseVector_compute(graph.thisptr, potentials.thisptr)
        return _SparseVectorMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         SparseVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(_SparseVectorW().init(SparseVectorPotential(<vector[pair[int, int]]>threshold)))
        projection = Projection(graph, bool_potentials)
        new_graph = projection.project(graph)
        new_potential = potentials.project(new_graph, projection)
        return new_graph, new_potential







def inside(Hypergraph graph, potentials):
    r"""
    Find the inside path chart values.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    potentials : :py:class:`Potentials`
      The potentials :math:`\theta` of the hypergraph.

    Returns
    -------

    : :py:class:`Chart`
       The inside chart.
    """
    return potentials.kind.inside(graph, potentials)

def outside(Hypergraph graph,
            potentials,
            inside_chart):
    """
    Find the outside scores for the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials of the hypergraph.

    inside_chart : :py:class:`Chart`
       The inside chart.

    Returns
    ---------

    : :py:class:`Chart`
       The outside chart.

    """
    return potentials.kind.outside(graph, potentials, inside_chart)

def best_path(Hypergraph graph, potentials):
    r"""
    Find the highest-scoring path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    potentials : :py:class:`Potentials`
      The potentials :math:`\theta` of the hypergraph.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """
    return potentials.kind.viterbi(graph, potentials)

def prune_hypergraph(Hypergraph graph, potentials, thres):
    """
    Prune hyperedges with low max-marginal score from the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
    The hypergraph to search.

    potentials : :py:class:`Potentials`
    The potentials of the hypergraph.

    Returns
    --------

    The new hypergraphs and potentials.
    """
    return potentials.kind.prune_hypergraph(graph, potentials, thres)


def compute_marginals(Hypergraph graph, potentials):
    return potentials.kind.compute_marginals(graph, potentials)

class Potentials(LogViterbiPotentials):
    pass

class Chart(_LogViterbiChart):
    r"""
    Chart :math:`S^{|{\cal V}|}` associated with a hypergraph (V, E) and semiring S.

    Acts as a vector::
       >> print chart[node]
    """
    pass

class Marginals(_LogViterbiMarginals):
    r"""
    Marginal values :math:`S^{|{\cal E} \times {\cal V}|}` associated with a hypergraph ({\cal V}, {\cal E}) and semiring S.

    Acts as a dictionary::
       >> print marginals[edge]
       >> print marginals[node]
    """
    pass

inside_values = inside
outside_values = outside

####### Methods that use specific potential ########
cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass CHypergraphProjection "HypergraphProjection":
        const CHypergraph *new_graph
        const CHyperedge *project(const CHyperedge *edge)
        const CHypernode *project(const CHypernode *node)

    const CHypergraphLogViterbiPotentials * cpairwise_dot "pairwise_dot"(
        const CHypergraphSparseVectorPotentials sparse_potentials,
        const vector[double] vec)

cdef extern from "Hypergraph/Semirings.h" namespace "HypergraphProjection":
    CHypergraphProjection *cproject_hypergraph "HypergraphProjection::project_hypergraph"(
        const CHypergraph *hypergraph,
        const CHypergraphBoolPotentials edge_mask)

def pairwise_dot(SparseVectorPotentials potentials, vec):
    cdef vector[double] rvec
    for i in vec:
        rvec.push_back(<double>i)
    cdef const CHypergraphLogViterbiPotentials *rpotentials = \
        cpairwise_dot(deref(potentials.thisptr), rvec)
    return LogViterbiPotentials(potentials.hypergraph).init(rpotentials)

cdef class Projection:
    cdef const CHypergraphProjection *thisptr

    def __init__(self, Hypergraph graph, BoolPotentials filt):
        """
        Prune hyperedges with low max-marginal score from the hypergraph.

        Parameters
        -----------

        graph : :py:class:`Hypergraph`
           The hypergraph to search.

        potentials : :py:class:`Potentials`
           The potentials of the hypergraph.

        Returns
        --------
        The new hypergraphs and potentials.
        """
        cdef const CHypergraphProjection *projection = \
            cproject_hypergraph(graph.thisptr,
                               deref(filt.thisptr))

        self.init(projection)

    cdef Projection init(self, const CHypergraphProjection *thisptr):
        self.thisptr = thisptr


    def project(self, Hypergraph graph):
        cdef Hypergraph new_graph = Hypergraph()
        cdef const CHypergraphProjection *projection = self.thisptr

        # Map nodes.
        node_labels = [None] * projection.new_graph.nodes().size()
        cdef vector[const CHypernode*] old_nodes = graph.thisptr.nodes()
        cdef const CHypernode *node
        for i in range(old_nodes.size()):
            node = projection.project(old_nodes[i])
            if node != NULL and node.id() >= 0:
                node_labels[node.id()] = graph.node_labels[i]

        # Map edges.
        edge_labels = [None] * projection.new_graph.edges().size()
        cdef vector[const CHyperedge *] old_edges = graph.thisptr.edges()
        cdef const CHyperedge *edge
        for i in range(old_edges.size()):
            edge = projection.project(old_edges[i])
            if edge != NULL and edge.id() >= 0:
                edge_labels[edge.id()] = graph.edge_labels[i]

        new_graph.init(projection.new_graph, node_labels, edge_labels)
        return new_graph
