#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

include "wrap.pxd"
include "hypergraph.pyx"
# include "constraints.pyx"
include "algorithms.pyx"



############# This is the templated semiring part. ##############



cdef extern from "Hypergraph/Algorithms.h":
    CViterbiChart *inside_Viterbi "general_inside<ViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphViterbiWeights theta) except +

    CViterbiChart *outside_Viterbi "general_outside<ViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphViterbiWeights theta,
        CViterbiChart inside_chart) except +

    cdef cppclass CViterbiMarginals "Marginals<ViterbiWeight>":
        ViterbiWeight marginal(const CHyperedge *edge)
        ViterbiWeight marginal(const CHypernode *node)

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

cdef extern from "Hypergraph/Algorithms.h" namespace "ViterbiWeight":
    cdef cppclass CHypergraphViterbiWeights "HypergraphWeights<ViterbiWeight>":
        ViterbiWeight dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphViterbiWeights *project_weights(
            const CHypergraphProjection )
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

    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph

    def build(self, fn):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
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
                                                  weights, Viterbi_one())
        return self

    cdef init(self, const CHypergraphViterbiWeights *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return self.thisptr.score(edge.edgeptr)

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _ViterbiW().init(self.thisptr.dot(deref(path.thisptr)))

cdef class _ViterbiW:
    cdef ViterbiWeight wrap

    cdef init(self, ViterbiWeight wrap):
        self.wrap = wrap
        return self

    
    def __float__(self):
        return <float>self.wrap
    

cdef class _ViterbiChart:
    cdef CViterbiChart *chart

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
        out_chart.chart = outside_Viterbi(graph.thisptr, deref(weights.thisptr), deref(inside_chart.chart))
        return out_chart

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _ViterbiWeights weights):
        cdef const CViterbiMarginals *marginals = \
            Viterbi_compute(graph.thisptr, weights.thisptr)
        return _ViterbiMarginals().init(marginals)



cdef extern from "Hypergraph/Algorithms.h":
    CLogViterbiChart *inside_LogViterbi "general_inside<LogViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiWeights theta) except +

    CLogViterbiChart *outside_LogViterbi "general_outside<LogViterbiWeight>" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiWeights theta,
        CLogViterbiChart inside_chart) except +

    cdef cppclass CLogViterbiMarginals "Marginals<LogViterbiWeight>":
        LogViterbiWeight marginal(const CHyperedge *edge)
        LogViterbiWeight marginal(const CHypernode *node)

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

cdef extern from "Hypergraph/Algorithms.h" namespace "LogViterbiWeight":
    cdef cppclass CHypergraphLogViterbiWeights "HypergraphWeights<LogViterbiWeight>":
        LogViterbiWeight dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphLogViterbiWeights *project_weights(
            const CHypergraphProjection )
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

    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph

    def build(self, fn):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
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
                                                  weights, LogViterbi_one())
        return self

    cdef init(self, const CHypergraphLogViterbiWeights *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return self.thisptr.score(edge.edgeptr)

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _LogViterbiW().init(self.thisptr.dot(deref(path.thisptr)))

cdef class _LogViterbiW:
    cdef LogViterbiWeight wrap

    cdef init(self, LogViterbiWeight wrap):
        self.wrap = wrap
        return self

    
    def __float__(self):
        return <float>self.wrap
    

cdef class _LogViterbiChart:
    cdef CLogViterbiChart *chart

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
        out_chart.chart = outside_LogViterbi(graph.thisptr, deref(weights.thisptr), deref(inside_chart.chart))
        return out_chart

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _LogViterbiWeights weights):
        cdef const CLogViterbiMarginals *marginals = \
            LogViterbi_compute(graph.thisptr, weights.thisptr)
        return _LogViterbiMarginals().init(marginals)



cdef extern from "Hypergraph/Algorithms.h":
    CInsideChart *inside_Inside "general_inside<InsideWeight>" (
        const CHypergraph *graph,
        const CHypergraphInsideWeights theta) except +

    CInsideChart *outside_Inside "general_outside<InsideWeight>" (
        const CHypergraph *graph,
        const CHypergraphInsideWeights theta,
        CInsideChart inside_chart) except +

    cdef cppclass CInsideMarginals "Marginals<InsideWeight>":
        InsideWeight marginal(const CHyperedge *edge)
        InsideWeight marginal(const CHypernode *node)

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

cdef extern from "Hypergraph/Algorithms.h" namespace "InsideWeight":
    cdef cppclass CHypergraphInsideWeights "HypergraphWeights<InsideWeight>":
        InsideWeight dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraphInsideWeights *project_weights(
            const CHypergraphProjection )
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

    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph

    def build(self, fn):
        """
        build(fn)

        Build the weight vector for a hypergraph.

        :param fn: A function from edge labels to weights.
        """
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
                                                  weights, Inside_one())
        return self

    cdef init(self, const CHypergraphInsideWeights *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return self.thisptr.score(edge.edgeptr)

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _InsideW().init(self.thisptr.dot(deref(path.thisptr)))

cdef class _InsideW:
    cdef InsideWeight wrap

    cdef init(self, InsideWeight wrap):
        self.wrap = wrap
        return self

    
    def __float__(self):
        return <float>self.wrap
    

cdef class _InsideChart:
    cdef CInsideChart *chart

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
        out_chart.chart = outside_Inside(graph.thisptr, deref(weights.thisptr), deref(inside_chart.chart))
        return out_chart

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _InsideWeights weights):
        cdef const CInsideMarginals *marginals = \
            Inside_compute(graph.thisptr, weights.thisptr)
        return _InsideMarginals().init(marginals)




####### These are the non-templated versions, now obsolete ########
