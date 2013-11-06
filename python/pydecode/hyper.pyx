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
    CViterbiChart *inside_Viterbi "general_inside<ViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphViterbiWeights theta) except +

    CViterbiChart *outside_Viterbi "general_outside<ViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphViterbiWeights theta,
        CViterbiChart inside_chart) except +

    CHyperpath *viterbi_Viterbi"general_viterbi<ViterbiWeight>"(
        const CHypergraph *graph,
        const CHypergraphViterbiWeights theta) except +

    cdef cppclass CViterbiMarginals "Marginals<ViterbiWeight>":
        ViterbiWeight marginal(const CHyperedge *edge)
        ViterbiWeight marginal(const CHypernode *node)
        CHypergraphBoolWeights *threshold(
            const ViterbiWeight &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CViterbiChart "Chart<ViterbiWeight>":
        ViterbiWeight get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<ViterbiWeight>":
    CViterbiMarginals *Viterbi_compute "Marginals<ViterbiWeight>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphViterbiWeights *weights)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass ViterbiWeight:
        ViterbiWeight()
        ViterbiWeight(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "ViterbiWeight":
    ViterbiWeight Viterbi_one "ViterbiWeight::one" ()
    ViterbiWeight Viterbi_zero "ViterbiWeight::zero" ()
    ViterbiWeight Viterbi_add "ViterbiWeight::add" (ViterbiWeight, const ViterbiWeight)
    ViterbiWeight Viterbi_times "ViterbiWeight::times" (ViterbiWeight, const ViterbiWeight)


cdef extern from "Hypergraph/Algorithms.h" namespace "ViterbiWeight":
    cdef cppclass CHypergraphViterbiWeights "HypergraphWeights<ViterbiWeight>":
        ViterbiWeight dot(const CHyperpath &path) except +
        ViterbiWeight score(const CHyperedge *edge)
        CHypergraphViterbiWeights *times(
            const CHypergraphViterbiWeights &weights)
        CHypergraphViterbiWeights *project_weights(
            const CHypergraphProjection)
        CHypergraphViterbiWeights(
            const CHypergraph *hypergraph,
            const vector[ViterbiWeight] weights,
            ViterbiWeight bias) except +

cdef class _ViterbiWeights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphViterbiWeights *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Viterbi

    def times(self, _ViterbiWeights other):
        cdef const CHypergraphViterbiWeights *new_weights = \
            self.thisptr.times(deref(other.thisptr))
        return _ViterbiWeights(self.hypergraph).init(new_weights)

    def project(self, Hypergraph graph, Projection projection):
        cdef _ViterbiWeights new_weights = _ViterbiWeights(graph)
        cdef const CHypergraphViterbiWeights *ptr = \
            self.thisptr.project_weights(deref(projection.thisptr))
        return new_weights.init(ptr)

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
        cdef ViterbiWeight my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = ViterbiWeight(<double> bias)

        cdef vector[ViterbiWeight] weights = \
             vector[ViterbiWeight](self.hypergraph.thisptr.edges().size(),
             Viterbi_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = Viterbi_zero()
            weights[i] = ViterbiWeight(<double> result)
        self.thisptr =  \
          new CHypergraphViterbiWeights(self.hypergraph.thisptr,
                                           weights, my_bias)
        return self

    cdef init(self, const CHypergraphViterbiWeights *ptr):
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
    cdef ViterbiWeight wrap

    cdef init(self, ViterbiWeight wrap):
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
        return _BoolWeights(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    

class Viterbi:
    Chart = _ViterbiChart
    Marginals = _ViterbiMarginals
    Semi = _ViterbiW
    Weights = _ViterbiWeights

    @staticmethod
    def inside(Hypergraph graph,
               _ViterbiWeights weights):
        cdef _ViterbiChart chart = _ViterbiChart()
        chart.chart = inside_Viterbi(graph.thisptr, deref(weights.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                _ViterbiWeights weights,
                _ViterbiChart inside_chart):
        cdef _ViterbiChart out_chart = _ViterbiChart()
        out_chart.chart = outside_Viterbi(graph.thisptr,
                                             deref(weights.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    
    @staticmethod
    def viterbi(Hypergraph graph,
                _ViterbiWeights weights):
        cdef CHyperpath *path = \
            viterbi_Viterbi(graph.thisptr,
                               deref(weights.thisptr))
        return Path().init(path)
    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _ViterbiWeights weights):
        cdef const CViterbiMarginals *marginals = \
            Viterbi_compute(graph.thisptr, weights.thisptr)
        return _ViterbiMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         _ViterbiWeights weights,
                         threshold):
        marginals = compute_marginals(graph, weights)
        bool_weights = marginals.threshold(_ViterbiW().init(ViterbiWeight(<double>threshold)))
        projection = Projection(graph, bool_weights)
        new_graph = projection.project(graph)
        new_weight = weights.project(new_graph, projection)
        return new_graph, new_weight




cdef extern from "Hypergraph/Algorithms.h":
    CLogViterbiChart *inside_LogViterbi "general_inside<LogViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiWeights theta) except +

    CLogViterbiChart *outside_LogViterbi "general_outside<LogViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiWeights theta,
        CLogViterbiChart inside_chart) except +

    CHyperpath *viterbi_LogViterbi"general_viterbi<LogViterbiWeight>"(
        const CHypergraph *graph,
        const CHypergraphLogViterbiWeights theta) except +

    cdef cppclass CLogViterbiMarginals "Marginals<LogViterbiWeight>":
        LogViterbiWeight marginal(const CHyperedge *edge)
        LogViterbiWeight marginal(const CHypernode *node)
        CHypergraphBoolWeights *threshold(
            const LogViterbiWeight &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CLogViterbiChart "Chart<LogViterbiWeight>":
        LogViterbiWeight get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<LogViterbiWeight>":
    CLogViterbiMarginals *LogViterbi_compute "Marginals<LogViterbiWeight>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphLogViterbiWeights *weights)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass LogViterbiWeight:
        LogViterbiWeight()
        LogViterbiWeight(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "LogViterbiWeight":
    LogViterbiWeight LogViterbi_one "LogViterbiWeight::one" ()
    LogViterbiWeight LogViterbi_zero "LogViterbiWeight::zero" ()
    LogViterbiWeight LogViterbi_add "LogViterbiWeight::add" (LogViterbiWeight, const LogViterbiWeight)
    LogViterbiWeight LogViterbi_times "LogViterbiWeight::times" (LogViterbiWeight, const LogViterbiWeight)


cdef extern from "Hypergraph/Algorithms.h" namespace "LogViterbiWeight":
    cdef cppclass CHypergraphLogViterbiWeights "HypergraphWeights<LogViterbiWeight>":
        LogViterbiWeight dot(const CHyperpath &path) except +
        LogViterbiWeight score(const CHyperedge *edge)
        CHypergraphLogViterbiWeights *times(
            const CHypergraphLogViterbiWeights &weights)
        CHypergraphLogViterbiWeights *project_weights(
            const CHypergraphProjection)
        CHypergraphLogViterbiWeights(
            const CHypergraph *hypergraph,
            const vector[LogViterbiWeight] weights,
            LogViterbiWeight bias) except +

cdef class _LogViterbiWeights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphLogViterbiWeights *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = LogViterbi

    def times(self, _LogViterbiWeights other):
        cdef const CHypergraphLogViterbiWeights *new_weights = \
            self.thisptr.times(deref(other.thisptr))
        return _LogViterbiWeights(self.hypergraph).init(new_weights)

    def project(self, Hypergraph graph, Projection projection):
        cdef _LogViterbiWeights new_weights = _LogViterbiWeights(graph)
        cdef const CHypergraphLogViterbiWeights *ptr = \
            self.thisptr.project_weights(deref(projection.thisptr))
        return new_weights.init(ptr)

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
        cdef LogViterbiWeight my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = LogViterbiWeight(<double> bias)

        cdef vector[LogViterbiWeight] weights = \
             vector[LogViterbiWeight](self.hypergraph.thisptr.edges().size(),
             LogViterbi_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = LogViterbi_zero()
            weights[i] = LogViterbiWeight(<double> result)
        self.thisptr =  \
          new CHypergraphLogViterbiWeights(self.hypergraph.thisptr,
                                           weights, my_bias)
        return self

    cdef init(self, const CHypergraphLogViterbiWeights *ptr):
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
    cdef LogViterbiWeight wrap

    cdef init(self, LogViterbiWeight wrap):
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
        return _BoolWeights(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    

class LogViterbi:
    Chart = _LogViterbiChart
    Marginals = _LogViterbiMarginals
    Semi = _LogViterbiW
    Weights = _LogViterbiWeights

    @staticmethod
    def inside(Hypergraph graph,
               _LogViterbiWeights weights):
        cdef _LogViterbiChart chart = _LogViterbiChart()
        chart.chart = inside_LogViterbi(graph.thisptr, deref(weights.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                _LogViterbiWeights weights,
                _LogViterbiChart inside_chart):
        cdef _LogViterbiChart out_chart = _LogViterbiChart()
        out_chart.chart = outside_LogViterbi(graph.thisptr,
                                             deref(weights.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    
    @staticmethod
    def viterbi(Hypergraph graph,
                _LogViterbiWeights weights):
        cdef CHyperpath *path = \
            viterbi_LogViterbi(graph.thisptr,
                               deref(weights.thisptr))
        return Path().init(path)
    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _LogViterbiWeights weights):
        cdef const CLogViterbiMarginals *marginals = \
            LogViterbi_compute(graph.thisptr, weights.thisptr)
        return _LogViterbiMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         _LogViterbiWeights weights,
                         threshold):
        marginals = compute_marginals(graph, weights)
        bool_weights = marginals.threshold(_LogViterbiW().init(LogViterbiWeight(<double>threshold)))
        projection = Projection(graph, bool_weights)
        new_graph = projection.project(graph)
        new_weight = weights.project(new_graph, projection)
        return new_graph, new_weight




cdef extern from "Hypergraph/Algorithms.h":
    CInsideChart *inside_Inside "general_inside<InsideWeight>" (
        const CHypergraph *graph,
        const CHypergraphInsideWeights theta) except +

    CInsideChart *outside_Inside "general_outside<InsideWeight>" (
        const CHypergraph *graph,
        const CHypergraphInsideWeights theta,
        CInsideChart inside_chart) except +

    CHyperpath *viterbi_Inside"general_viterbi<InsideWeight>"(
        const CHypergraph *graph,
        const CHypergraphInsideWeights theta) except +

    cdef cppclass CInsideMarginals "Marginals<InsideWeight>":
        InsideWeight marginal(const CHyperedge *edge)
        InsideWeight marginal(const CHypernode *node)
        CHypergraphBoolWeights *threshold(
            const InsideWeight &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CInsideChart "Chart<InsideWeight>":
        InsideWeight get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<InsideWeight>":
    CInsideMarginals *Inside_compute "Marginals<InsideWeight>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphInsideWeights *weights)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass InsideWeight:
        InsideWeight()
        InsideWeight(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "InsideWeight":
    InsideWeight Inside_one "InsideWeight::one" ()
    InsideWeight Inside_zero "InsideWeight::zero" ()
    InsideWeight Inside_add "InsideWeight::add" (InsideWeight, const InsideWeight)
    InsideWeight Inside_times "InsideWeight::times" (InsideWeight, const InsideWeight)


cdef extern from "Hypergraph/Algorithms.h" namespace "InsideWeight":
    cdef cppclass CHypergraphInsideWeights "HypergraphWeights<InsideWeight>":
        InsideWeight dot(const CHyperpath &path) except +
        InsideWeight score(const CHyperedge *edge)
        CHypergraphInsideWeights *times(
            const CHypergraphInsideWeights &weights)
        CHypergraphInsideWeights *project_weights(
            const CHypergraphProjection)
        CHypergraphInsideWeights(
            const CHypergraph *hypergraph,
            const vector[InsideWeight] weights,
            InsideWeight bias) except +

cdef class _InsideWeights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphInsideWeights *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Inside

    def times(self, _InsideWeights other):
        cdef const CHypergraphInsideWeights *new_weights = \
            self.thisptr.times(deref(other.thisptr))
        return _InsideWeights(self.hypergraph).init(new_weights)

    def project(self, Hypergraph graph, Projection projection):
        cdef _InsideWeights new_weights = _InsideWeights(graph)
        cdef const CHypergraphInsideWeights *ptr = \
            self.thisptr.project_weights(deref(projection.thisptr))
        return new_weights.init(ptr)

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
        cdef InsideWeight my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = InsideWeight(<double> bias)

        cdef vector[InsideWeight] weights = \
             vector[InsideWeight](self.hypergraph.thisptr.edges().size(),
             Inside_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = Inside_zero()
            weights[i] = InsideWeight(<double> result)
        self.thisptr =  \
          new CHypergraphInsideWeights(self.hypergraph.thisptr,
                                           weights, my_bias)
        return self

    cdef init(self, const CHypergraphInsideWeights *ptr):
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
    cdef InsideWeight wrap

    cdef init(self, InsideWeight wrap):
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
    

class Inside:
    Chart = _InsideChart
    Marginals = _InsideMarginals
    Semi = _InsideW
    Weights = _InsideWeights

    @staticmethod
    def inside(Hypergraph graph,
               _InsideWeights weights):
        cdef _InsideChart chart = _InsideChart()
        chart.chart = inside_Inside(graph.thisptr, deref(weights.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                _InsideWeights weights,
                _InsideChart inside_chart):
        cdef _InsideChart out_chart = _InsideChart()
        out_chart.chart = outside_Inside(graph.thisptr,
                                             deref(weights.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _InsideWeights weights):
        cdef const CInsideMarginals *marginals = \
            Inside_compute(graph.thisptr, weights.thisptr)
        return _InsideMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         _InsideWeights weights,
                         threshold):
        marginals = compute_marginals(graph, weights)
        bool_weights = marginals.threshold(_InsideW().init(InsideWeight(<double>threshold)))
        projection = Projection(graph, bool_weights)
        new_graph = projection.project(graph)
        new_weight = weights.project(new_graph, projection)
        return new_graph, new_weight




cdef extern from "Hypergraph/Algorithms.h":
    CBoolChart *inside_Bool "general_inside<BoolWeight>" (
        const CHypergraph *graph,
        const CHypergraphBoolWeights theta) except +

    CBoolChart *outside_Bool "general_outside<BoolWeight>" (
        const CHypergraph *graph,
        const CHypergraphBoolWeights theta,
        CBoolChart inside_chart) except +

    CHyperpath *viterbi_Bool"general_viterbi<BoolWeight>"(
        const CHypergraph *graph,
        const CHypergraphBoolWeights theta) except +

    cdef cppclass CBoolMarginals "Marginals<BoolWeight>":
        BoolWeight marginal(const CHyperedge *edge)
        BoolWeight marginal(const CHypernode *node)
        CHypergraphBoolWeights *threshold(
            const BoolWeight &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CBoolChart "Chart<BoolWeight>":
        BoolWeight get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<BoolWeight>":
    CBoolMarginals *Bool_compute "Marginals<BoolWeight>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphBoolWeights *weights)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass BoolWeight:
        BoolWeight()
        BoolWeight(double)
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "BoolWeight":
    BoolWeight Bool_one "BoolWeight::one" ()
    BoolWeight Bool_zero "BoolWeight::zero" ()
    BoolWeight Bool_add "BoolWeight::add" (BoolWeight, const BoolWeight)
    BoolWeight Bool_times "BoolWeight::times" (BoolWeight, const BoolWeight)


cdef extern from "Hypergraph/Algorithms.h" namespace "BoolWeight":
    cdef cppclass CHypergraphBoolWeights "HypergraphWeights<BoolWeight>":
        BoolWeight dot(const CHyperpath &path) except +
        BoolWeight score(const CHyperedge *edge)
        CHypergraphBoolWeights *times(
            const CHypergraphBoolWeights &weights)
        CHypergraphBoolWeights *project_weights(
            const CHypergraphProjection)
        CHypergraphBoolWeights(
            const CHypergraph *hypergraph,
            const vector[BoolWeight] weights,
            BoolWeight bias) except +

cdef class _BoolWeights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphBoolWeights *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Bool

    def times(self, _BoolWeights other):
        cdef const CHypergraphBoolWeights *new_weights = \
            self.thisptr.times(deref(other.thisptr))
        return _BoolWeights(self.hypergraph).init(new_weights)

    def project(self, Hypergraph graph, Projection projection):
        cdef _BoolWeights new_weights = _BoolWeights(graph)
        cdef const CHypergraphBoolWeights *ptr = \
            self.thisptr.project_weights(deref(projection.thisptr))
        return new_weights.init(ptr)

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
        cdef BoolWeight my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = BoolWeight(<double> bias)

        cdef vector[BoolWeight] weights = \
             vector[BoolWeight](self.hypergraph.thisptr.edges().size(),
             Bool_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = Bool_zero()
            weights[i] = BoolWeight(<double> result)
        self.thisptr =  \
          new CHypergraphBoolWeights(self.hypergraph.thisptr,
                                           weights, my_bias)
        return self

    cdef init(self, const CHypergraphBoolWeights *ptr):
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
    cdef BoolWeight wrap

    cdef init(self, BoolWeight wrap):
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
        return _BoolWeights(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    

class Bool:
    Chart = _BoolChart
    Marginals = _BoolMarginals
    Semi = _BoolW
    Weights = _BoolWeights

    @staticmethod
    def inside(Hypergraph graph,
               _BoolWeights weights):
        cdef _BoolChart chart = _BoolChart()
        chart.chart = inside_Bool(graph.thisptr, deref(weights.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                _BoolWeights weights,
                _BoolChart inside_chart):
        cdef _BoolChart out_chart = _BoolChart()
        out_chart.chart = outside_Bool(graph.thisptr,
                                             deref(weights.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    
    @staticmethod
    def viterbi(Hypergraph graph,
                _BoolWeights weights):
        cdef CHyperpath *path = \
            viterbi_Bool(graph.thisptr,
                               deref(weights.thisptr))
        return Path().init(path)
    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _BoolWeights weights):
        cdef const CBoolMarginals *marginals = \
            Bool_compute(graph.thisptr, weights.thisptr)
        return _BoolMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         _BoolWeights weights,
                         threshold):
        marginals = compute_marginals(graph, weights)
        bool_weights = marginals.threshold(_BoolW().init(BoolWeight(<double>threshold)))
        projection = Projection(graph, bool_weights)
        new_graph = projection.project(graph)
        new_weight = weights.project(new_graph, projection)
        return new_graph, new_weight




cdef extern from "Hypergraph/Algorithms.h":
    CSparseVectorChart *inside_SparseVector "general_inside<SparseVectorWeight>" (
        const CHypergraph *graph,
        const CHypergraphSparseVectorWeights theta) except +

    CSparseVectorChart *outside_SparseVector "general_outside<SparseVectorWeight>" (
        const CHypergraph *graph,
        const CHypergraphSparseVectorWeights theta,
        CSparseVectorChart inside_chart) except +

    CHyperpath *viterbi_SparseVector"general_viterbi<SparseVectorWeight>"(
        const CHypergraph *graph,
        const CHypergraphSparseVectorWeights theta) except +

    cdef cppclass CSparseVectorMarginals "Marginals<SparseVectorWeight>":
        SparseVectorWeight marginal(const CHyperedge *edge)
        SparseVectorWeight marginal(const CHypernode *node)
        CHypergraphBoolWeights *threshold(
            const SparseVectorWeight &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass CSparseVectorChart "Chart<SparseVectorWeight>":
        SparseVectorWeight get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<SparseVectorWeight>":
    CSparseVectorMarginals *SparseVector_compute "Marginals<SparseVectorWeight>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraphSparseVectorWeights *weights)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass SparseVectorWeight:
        SparseVectorWeight()
        SparseVectorWeight(vector[pair[int, int]])
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "SparseVectorWeight":
    SparseVectorWeight SparseVector_one "SparseVectorWeight::one" ()
    SparseVectorWeight SparseVector_zero "SparseVectorWeight::zero" ()
    SparseVectorWeight SparseVector_add "SparseVectorWeight::add" (SparseVectorWeight, const SparseVectorWeight)
    SparseVectorWeight SparseVector_times "SparseVectorWeight::times" (SparseVectorWeight, const SparseVectorWeight)


cdef extern from "Hypergraph/Algorithms.h" namespace "SparseVectorWeight":
    cdef cppclass CHypergraphSparseVectorWeights "HypergraphWeights<SparseVectorWeight>":
        SparseVectorWeight dot(const CHyperpath &path) except +
        SparseVectorWeight score(const CHyperedge *edge)
        CHypergraphSparseVectorWeights *times(
            const CHypergraphSparseVectorWeights &weights)
        CHypergraphSparseVectorWeights *project_weights(
            const CHypergraphProjection)
        CHypergraphSparseVectorWeights(
            const CHypergraph *hypergraph,
            const vector[SparseVectorWeight] weights,
            SparseVectorWeight bias) except +

cdef class _SparseVectorWeights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraphSparseVectorWeights *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = SparseVector

    def times(self, _SparseVectorWeights other):
        cdef const CHypergraphSparseVectorWeights *new_weights = \
            self.thisptr.times(deref(other.thisptr))
        return _SparseVectorWeights(self.hypergraph).init(new_weights)

    def project(self, Hypergraph graph, Projection projection):
        cdef _SparseVectorWeights new_weights = _SparseVectorWeights(graph)
        cdef const CHypergraphSparseVectorWeights *ptr = \
            self.thisptr.project_weights(deref(projection.thisptr))
        return new_weights.init(ptr)

    property kind:
        def __get__(self):
            return self.kind

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
        cdef SparseVectorWeight my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = SparseVectorWeight(<vector[pair[int, int]]> bias)

        cdef vector[SparseVectorWeight] weights = \
             vector[SparseVectorWeight](self.hypergraph.thisptr.edges().size(),
             SparseVector_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = SparseVector_zero()
            weights[i] = SparseVectorWeight(<vector[pair[int, int]]> result)
        self.thisptr =  \
          new CHypergraphSparseVectorWeights(self.hypergraph.thisptr,
                                           weights, my_bias)
        return self

    cdef init(self, const CHypergraphSparseVectorWeights *ptr):
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
    cdef SparseVectorWeight wrap

    cdef init(self, SparseVectorWeight wrap):
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
    Weights = _SparseVectorWeights

    @staticmethod
    def inside(Hypergraph graph,
               _SparseVectorWeights weights):
        cdef _SparseVectorChart chart = _SparseVectorChart()
        chart.chart = inside_SparseVector(graph.thisptr, deref(weights.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                _SparseVectorWeights weights,
                _SparseVectorChart inside_chart):
        cdef _SparseVectorChart out_chart = _SparseVectorChart()
        out_chart.chart = outside_SparseVector(graph.thisptr,
                                             deref(weights.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _SparseVectorWeights weights):
        cdef const CSparseVectorMarginals *marginals = \
            SparseVector_compute(graph.thisptr, weights.thisptr)
        return _SparseVectorMarginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         _SparseVectorWeights weights,
                         threshold):
        marginals = compute_marginals(graph, weights)
        bool_weights = marginals.threshold(_SparseVectorW().init(SparseVectorWeight(<vector[pair[int, int]]>threshold)))
        projection = Projection(graph, bool_weights)
        new_graph = projection.project(graph)
        new_weight = weights.project(new_graph, projection)
        return new_graph, new_weight




def inside(Hypergraph graph, weights):
    r"""
    Find the inside path chart values.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    weights : :py:class:`Weights`
      The weights :math:`\theta` of the hypergraph.

    Returns
    -------

    : :py:class:`Chart`
       The inside chart.
    """
    return weights.kind.inside(graph, weights)

def outside(Hypergraph graph,
            weights,
            inside_chart):
    """
    Find the outside scores for the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    weights : :py:class:`Weights`
       The weights of the hypergraph.

    inside_chart : :py:class:`Chart`
       The inside chart.

    Returns
    ---------

    : :py:class:`Chart`
       The outside chart.

    """
    return weights.kind.outside(graph, weights, inside_chart)

def best_path(Hypergraph graph, weights):
    r"""
    Find the highest-scoring path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    weights : :py:class:`Weights`
      The weights :math:`\theta` of the hypergraph.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """
    return weights.kind.viterbi(graph, weights)

def prune_hypergraph(Hypergraph graph, weights, thres):
    """
    Prune hyperedges with low max-marginal score from the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
    The hypergraph to search.

    weights : :py:class:`Weights`
    The weights of the hypergraph.

    Returns
    --------

    The new hypergraphs and weights.
    """
    return weights.kind.prune_hypergraph(graph, weights, thres)


def compute_marginals(Hypergraph graph, weights):
    return weights.kind.compute_marginals(graph, weights)

def Weights(Hypergraph graph, kind):
    return kind.Weights(graph)

inside_values = inside
outside_values = outside


####### These are methods that use specific weight ########
cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass CHypergraphProjection "HypergraphProjection":
        const CHypergraph *new_graph
        const CHyperedge *project(const CHyperedge *edge)
        const CHypernode *project(const CHypernode *node)

    const CHypergraphLogViterbiWeights * cpairwise_dot "pairwise_dot"(
        const CHypergraphSparseVectorWeights sparse_weights,
        const vector[double] vec)

cdef extern from "Hypergraph/Semirings.h" namespace "HypergraphProjection":
    CHypergraphProjection *cproject_hypergraph "HypergraphProjection::project_hypergraph"(
        const CHypergraph *hypergraph,
        const CHypergraphBoolWeights edge_mask)


cdef extern from "Hypergraph/Algorithms.h":
    const CHyperpath *best_constrained_path(
        const CHypergraph *graph,
        const CHypergraphLogViterbiWeights theta,
        const CHypergraphSparseVectorWeights constraints) except +


def pairwise_dot(_SparseVectorWeights weights, vec):
    cdef vector[double] rvec
    for i in vec:
        rvec.push_back(<double>i)
    cdef const CHypergraphLogViterbiWeights *rweights = \
        cpairwise_dot(deref(weights.thisptr), rvec)
    return _LogViterbiWeights(weights.hypergraph).init(rweights)

cdef class Projection:
    cdef const CHypergraphProjection *thisptr

    def __init__(self, Hypergraph graph, _BoolWeights filt):
        """
        Prune hyperedges with low max-marginal score from the hypergraph.

        Parameters
        -----------

        graph : :py:class:`Hypergraph`
           The hypergraph to search.

        weights : :py:class:`Weights`
           The weights of the hypergraph.

        Returns
        --------

        The new hypergraphs and weights.
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




# def best_constrained_path(
#     Hypergraph graph,
#     _LogViterbiWeights weights,
#     constraints):
#     """
#     Find the highest-scoring path satisfying constraints.


#     Parameters
#     -----------

#     graph : :py:class:`Hypergraph`
#        The hypergraph to search.

#     weights : :py:class:`Weights`
#        The weights of the hypergraph.

#     constraints : :py:class:`Constraints`
#         The hyperedge constraints.

#     Returns
#     ---------

#     The best path and the dual values.
#     """
#     #cdef vector[CConstrainedResult] results
#     cdef const CHyperpath *cpath = \
#         best_constrained_path(graph.thisptr,
#                               deref(weights.thisptr),
#                               deref(constraints.thisptr))

#     return Path().init(cpath)


# cdef convert_results(vector[CConstrainedResult] c):
#     return [ConstrainedResult().init(cresult) for cresult in c]




# cdef class ConstrainedResult:
#     r"""
#     A sub-result from the constrained solver.


#     Attributes
#     -----------

#     path : :py:class:`Path`
#       The hyperpath :math:`y \in {\cal X}`
#       associated with this round.

#     dual : float
#        The dual value for this round.

#     primal : float
#        The primal value for this round.

#     constraints : list of :py:class:`Constraint`
#        The constraints violated for this round.
#     """

#     cdef CConstrainedResult thisptr
#     cdef ConstrainedResult init(self, CConstrainedResult ptr):
#         self.thisptr = ptr
#         return self

#     property path:
#         def __get__(self):
#             return Path().init(self.thisptr.path)

#     property dual:
#         def __get__(self):
#             return self.thisptr.dual

#     property primal:
#         def __get__(self):
#             return self.thisptr.primal

#     property constraints:
#         def __get__(self):
#             return None #convert_constraints(self.thisptr.constraints)

    # cdef Hypergraph new_graph = Hypergraph()

    # # Map nodes.
    # node_labels = [None] * projection.new_graph.nodes().size()
    # cdef vector[const CHypernode*] old_nodes = graph.thisptr.nodes()
    # cdef const CHypernode *node
    # for i in range(old_nodes.size()):
    #     node = projection.project(old_nodes[i])
    #     if node != NULL and node.id() >= 0:
    #         node_labels[node.id()] = graph.node_labels[i]

    # # Map edges.
    # edge_labels = [None] * projection.new_graph.edges().size()
    # cdef vector[const CHyperedge *] old_edges = graph.thisptr.edges()
    # cdef const CHyperedge *edge
    # for i in range(old_edges.size()):
    #     edge = projection.project(old_edges[i])
    #     if edge != NULL and edge.id() >= 0:
    #         edge_labels[edge.id()] = graph.edge_labels[i]

    # new_graph.init(projection.new_graph, node_labels, edge_labels)
    # cdef Weights new_weights = Weights(new_graph)
    # new_weights.init(
    #     weights.thisptr.project_weights(deref(projection)))
    # return new_graph, new_weights
