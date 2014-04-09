#cython: embedsignature=True

##
## DO NOT MODIFY THIS GENERATED FILE.
##


from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
cimport libcpp.map as c_map
from libcpp.pair cimport pair
from libcpp cimport bool

from wrap cimport *
from libhypergraph cimport *
import libhypergraph as py_hypergraph


############# This is the templated semiring part. ##############



cdef class ViterbiPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Viterbi
        self.thisptr = NULL

    def times(self, ViterbiPotentials other):
        cdef CHypergraphViterbiPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return ViterbiPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return ViterbiPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return ViterbiPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphViterbiPotentials *ptr = \
            cmake_projected_potentials_Viterbi(self.thisptr,
                                                  projection.thisptr)
        return ViterbiPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _Viterbi_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef double my_bias
    #     if bias is None:
    #         my_bias = Viterbi_one()
    #     else:
    #         my_bias = _Viterbi_to_cpp(bias)

    #     cdef vector[double] potentials = \
    #          vector[double](self.hypergraph.thisptr.edges().size(),
    #          Viterbi_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Viterbi_zero()
    #         potentials[i] = _Viterbi_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Viterbi(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Viterbi_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        potentials,
                                        _Viterbi_to_cpp(other_potentials.bias))

        return self


    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = _Viterbi_to_cpp(bias)

        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Viterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = _Viterbi_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Viterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphViterbiPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Viterbi_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _Viterbi_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _Viterbi:
    cdef _Viterbi init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = _Viterbi()
        created.thisval = _Viterbi_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Viterbi_from_cpp(Viterbi_zero())

    @staticmethod
    def one_raw():
        return _Viterbi_from_cpp(Viterbi_one())

    @staticmethod
    def zero():
        return _Viterbi().init(Viterbi_zero())

    @staticmethod
    def one():
        return _Viterbi().init(Viterbi_one())

    def __add__(_Viterbi self, _Viterbi other):
        return _Viterbi().init(Viterbi_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Viterbi self, _Viterbi other):
        return _Viterbi().init(Viterbi_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Viterbi_from_cpp(self.thisval)

cdef double _Viterbi_to_cpp(double val):
    return val


cdef _Viterbi_from_cpp(double val):
    return val

cdef class ViterbiChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Viterbi
        self.chart = NULL
        if graph is not None:
            self.chart = new CViterbiChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _Viterbi_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _ViterbiMarginals:
    cdef const CViterbiMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CViterbiMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Viterbi_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _Viterbi_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Viterbi marginal values." +
                "Passed %s." % obj)



    def threshold(self, double semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)



class Viterbi:
    Chart = ViterbiChart
    Marginals = _ViterbiMarginals
    #Semi = _Viterbi
    Potentials = ViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               ViterbiPotentials potentials):
        cdef ViterbiChart chart = ViterbiChart()
        chart.chart = inside_Viterbi(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                ViterbiPotentials potentials,
                ViterbiChart inside_chart):
        cdef ViterbiChart out_chart = ViterbiChart()
        out_chart.chart = outside_Viterbi(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def viterbi(Hypergraph graph,
                ViterbiPotentials potentials,
                ViterbiChart chart=None):
        cdef CViterbiChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CViterbiChart(graph.thisptr)
        viterbi_Viterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          ViterbiPotentials potentials):
        cdef const CViterbiMarginals *marginals = \
            Viterbi_compute(graph.thisptr, potentials.thisptr)
        return _ViterbiMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         ViterbiPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class LogViterbiPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = LogViterbi
        self.thisptr = NULL

    def times(self, LogViterbiPotentials other):
        cdef CHypergraphLogViterbiPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return LogViterbiPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return LogViterbiPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphLogViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return LogViterbiPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphLogViterbiPotentials *ptr = \
            cmake_projected_potentials_LogViterbi(self.thisptr,
                                                  projection.thisptr)
        return LogViterbiPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _LogViterbi_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef double my_bias
    #     if bias is None:
    #         my_bias = LogViterbi_one()
    #     else:
    #         my_bias = _LogViterbi_to_cpp(bias)

    #     cdef vector[double] potentials = \
    #          vector[double](self.hypergraph.thisptr.edges().size(),
    #          LogViterbi_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = LogViterbi_zero()
    #         potentials[i] = _LogViterbi_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_LogViterbi(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _LogViterbi_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        potentials,
                                        _LogViterbi_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = _LogViterbi_to_cpp(bias)

        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _LogViterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = _LogViterbi_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _LogViterbi_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphLogViterbiPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _LogViterbi_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _LogViterbi_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _LogViterbi:
    cdef _LogViterbi init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = _LogViterbi()
        created.thisval = _LogViterbi_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _LogViterbi_from_cpp(LogViterbi_zero())

    @staticmethod
    def one_raw():
        return _LogViterbi_from_cpp(LogViterbi_one())

    @staticmethod
    def zero():
        return _LogViterbi().init(LogViterbi_zero())

    @staticmethod
    def one():
        return _LogViterbi().init(LogViterbi_one())

    def __add__(_LogViterbi self, _LogViterbi other):
        return _LogViterbi().init(LogViterbi_add(self.thisval,
                                                  other.thisval))

    def __mul__(_LogViterbi self, _LogViterbi other):
        return _LogViterbi().init(LogViterbi_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _LogViterbi_from_cpp(self.thisval)

cdef double _LogViterbi_to_cpp(double val):
    return val


cdef _LogViterbi_from_cpp(double val):
    return val

cdef class LogViterbiChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = LogViterbi
        self.chart = NULL
        if graph is not None:
            self.chart = new CLogViterbiChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _LogViterbi_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _LogViterbiMarginals:
    cdef const CLogViterbiMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CLogViterbiMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _LogViterbi_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _LogViterbi_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have LogViterbi marginal values." +
                "Passed %s." % obj)



    def threshold(self, double semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)



class LogViterbi:
    Chart = LogViterbiChart
    Marginals = _LogViterbiMarginals
    #Semi = _LogViterbi
    Potentials = LogViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               LogViterbiPotentials potentials):
        cdef LogViterbiChart chart = LogViterbiChart()
        chart.chart = inside_LogViterbi(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                LogViterbiPotentials potentials,
                LogViterbiChart inside_chart):
        cdef LogViterbiChart out_chart = LogViterbiChart()
        out_chart.chart = outside_LogViterbi(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def viterbi(Hypergraph graph,
                LogViterbiPotentials potentials,
                LogViterbiChart chart=None):
        cdef CLogViterbiChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CLogViterbiChart(graph.thisptr)
        viterbi_LogViterbi(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          LogViterbiPotentials potentials):
        cdef const CLogViterbiMarginals *marginals = \
            LogViterbi_compute(graph.thisptr, potentials.thisptr)
        return _LogViterbiMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         LogViterbiPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class InsidePotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Inside
        self.thisptr = NULL

    def times(self, InsidePotentials other):
        cdef CHypergraphInsidePotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return InsidePotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return InsidePotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphInsidePotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return InsidePotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphInsidePotentials *ptr = \
            cmake_projected_potentials_Inside(self.thisptr,
                                                  projection.thisptr)
        return InsidePotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _Inside_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef double my_bias
    #     if bias is None:
    #         my_bias = Inside_one()
    #     else:
    #         my_bias = _Inside_to_cpp(bias)

    #     cdef vector[double] potentials = \
    #          vector[double](self.hypergraph.thisptr.edges().size(),
    #          Inside_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Inside_zero()
    #         potentials[i] = _Inside_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Inside(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Inside_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        potentials,
                                        _Inside_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = _Inside_to_cpp(bias)

        cdef vector[double] potentials = \
            vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Inside_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = _Inside_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Inside_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphInsidePotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Inside_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _Inside_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _Inside:
    cdef _Inside init(self, double val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(double val):
        created = _Inside()
        created.thisval = _Inside_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Inside_from_cpp(Inside_zero())

    @staticmethod
    def one_raw():
        return _Inside_from_cpp(Inside_one())

    @staticmethod
    def zero():
        return _Inside().init(Inside_zero())

    @staticmethod
    def one():
        return _Inside().init(Inside_one())

    def __add__(_Inside self, _Inside other):
        return _Inside().init(Inside_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Inside self, _Inside other):
        return _Inside().init(Inside_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Inside_from_cpp(self.thisval)

cdef double _Inside_to_cpp(double val):
    return val


cdef _Inside_from_cpp(double val):
    return val

cdef class InsideChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Inside
        self.chart = NULL
        if graph is not None:
            self.chart = new CInsideChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _Inside_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _InsideMarginals:
    cdef const CInsideMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CInsideMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Inside_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _Inside_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Inside marginal values." +
                "Passed %s." % obj)



    def threshold(self, double semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)



class Inside:
    Chart = InsideChart
    Marginals = _InsideMarginals
    #Semi = _Inside
    Potentials = InsidePotentials

    @staticmethod
    def inside(Hypergraph graph,
               InsidePotentials potentials):
        cdef InsideChart chart = InsideChart()
        chart.chart = inside_Inside(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                InsidePotentials potentials,
                InsideChart inside_chart):
        cdef InsideChart out_chart = InsideChart()
        out_chart.chart = outside_Inside(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def viterbi(Hypergraph graph,
                InsidePotentials potentials,
                InsideChart chart=None):
        cdef CInsideChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CInsideChart(graph.thisptr)
        viterbi_Inside(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          InsidePotentials potentials):
        cdef const CInsideMarginals *marginals = \
            Inside_compute(graph.thisptr, potentials.thisptr)
        return _InsideMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         InsidePotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class BoolPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Bool
        self.thisptr = NULL

    def times(self, BoolPotentials other):
        cdef CHypergraphBoolPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return BoolPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return BoolPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphBoolPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return BoolPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphBoolPotentials *ptr = \
            cmake_projected_potentials_Bool(self.thisptr,
                                                  projection.thisptr)
        return BoolPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _Bool_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef bool my_bias
    #     if bias is None:
    #         my_bias = Bool_one()
    #     else:
    #         my_bias = _Bool_to_cpp(bias)

    #     cdef vector[bool] potentials = \
    #          vector[bool](self.hypergraph.thisptr.edges().size(),
    #          Bool_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Bool_zero()
    #         potentials[i] = _Bool_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Bool(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[bool] potentials = \
            vector[bool](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Bool_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        potentials,
                                        _Bool_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef bool my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = _Bool_to_cpp(bias)

        cdef vector[bool] potentials = \
            vector[bool](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Bool_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef bool my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = _Bool_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[bool] potentials = \
            vector[bool](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Bool_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphBoolPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Bool_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _Bool_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _Bool:
    cdef _Bool init(self, bool val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(bool val):
        created = _Bool()
        created.thisval = _Bool_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Bool_from_cpp(Bool_zero())

    @staticmethod
    def one_raw():
        return _Bool_from_cpp(Bool_one())

    @staticmethod
    def zero():
        return _Bool().init(Bool_zero())

    @staticmethod
    def one():
        return _Bool().init(Bool_one())

    def __add__(_Bool self, _Bool other):
        return _Bool().init(Bool_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Bool self, _Bool other):
        return _Bool().init(Bool_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Bool_from_cpp(self.thisval)

cdef bool _Bool_to_cpp(bool val):
    return val


cdef _Bool_from_cpp(bool val):
    return val

cdef class BoolChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Bool
        self.chart = NULL
        if graph is not None:
            self.chart = new CBoolChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _Bool_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _BoolMarginals:
    cdef const CBoolMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CBoolMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Bool_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _Bool_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Bool marginal values." +
                "Passed %s." % obj)



    def threshold(self, bool semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)



class Bool:
    Chart = BoolChart
    Marginals = _BoolMarginals
    #Semi = _Bool
    Potentials = BoolPotentials

    @staticmethod
    def inside(Hypergraph graph,
               BoolPotentials potentials):
        cdef BoolChart chart = BoolChart()
        chart.chart = inside_Bool(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                BoolPotentials potentials,
                BoolChart inside_chart):
        cdef BoolChart out_chart = BoolChart()
        out_chart.chart = outside_Bool(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def viterbi(Hypergraph graph,
                BoolPotentials potentials,
                BoolChart chart=None):
        cdef CBoolChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CBoolChart(graph.thisptr)
        viterbi_Bool(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          BoolPotentials potentials):
        cdef const CBoolMarginals *marginals = \
            Bool_compute(graph.thisptr, potentials.thisptr)
        return _BoolMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         BoolPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class SparseVectorPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = SparseVector
        self.thisptr = NULL

    def times(self, SparseVectorPotentials other):
        cdef CHypergraphSparseVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return SparseVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return SparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return SparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphSparseVectorPotentials *ptr = \
            cmake_projected_potentials_SparseVector(self.thisptr,
                                                  projection.thisptr)
        return SparseVectorPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _SparseVector_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef vector[pair[int, int]] my_bias
    #     if bias is None:
    #         my_bias = SparseVector_one()
    #     else:
    #         my_bias = _SparseVector_to_cpp(bias)

    #     cdef vector[vector[pair[int, int]]] potentials = \
    #          vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
    #          SparseVector_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = SparseVector_zero()
    #         potentials[i] = _SparseVector_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_SparseVector(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _SparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _SparseVector_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = _SparseVector_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _SparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = _SparseVector_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _SparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphSparseVectorPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _SparseVector_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _SparseVector_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _SparseVector:
    cdef _SparseVector init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = _SparseVector()
        created.thisval = _SparseVector_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _SparseVector_from_cpp(SparseVector_zero())

    @staticmethod
    def one_raw():
        return _SparseVector_from_cpp(SparseVector_one())

    @staticmethod
    def zero():
        return _SparseVector().init(SparseVector_zero())

    @staticmethod
    def one():
        return _SparseVector().init(SparseVector_one())

    def __add__(_SparseVector self, _SparseVector other):
        return _SparseVector().init(SparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(_SparseVector self, _SparseVector other):
        return _SparseVector().init(SparseVector_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _SparseVector_from_cpp(self.thisval)

cdef vector[pair[int, int]] _SparseVector_to_cpp(vector[pair[int, int]] val):
    return val


cdef _SparseVector_from_cpp(vector[pair[int, int]] val):
    return val

cdef class SparseVectorChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = SparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CSparseVectorChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _SparseVector_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _SparseVectorMarginals:
    cdef const CSparseVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CSparseVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _SparseVector_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _SparseVector_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have SparseVector marginal values." +
                "Passed %s." % obj)




class SparseVector:
    Chart = SparseVectorChart
    Marginals = _SparseVectorMarginals
    #Semi = _SparseVector
    Potentials = SparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               SparseVectorPotentials potentials):
        cdef SparseVectorChart chart = SparseVectorChart()
        chart.chart = inside_SparseVector(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                SparseVectorPotentials potentials,
                SparseVectorChart inside_chart):
        cdef SparseVectorChart out_chart = SparseVectorChart()
        out_chart.chart = outside_SparseVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          SparseVectorPotentials potentials):
        cdef const CSparseVectorMarginals *marginals = \
            SparseVector_compute(graph.thisptr, potentials.thisptr)
        return _SparseVectorMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         SparseVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class MinSparseVectorPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = MinSparseVector
        self.thisptr = NULL

    def times(self, MinSparseVectorPotentials other):
        cdef CHypergraphMinSparseVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return MinSparseVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return MinSparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMinSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return MinSparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMinSparseVectorPotentials *ptr = \
            cmake_projected_potentials_MinSparseVector(self.thisptr,
                                                  projection.thisptr)
        return MinSparseVectorPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _MinSparseVector_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef vector[pair[int, int]] my_bias
    #     if bias is None:
    #         my_bias = MinSparseVector_one()
    #     else:
    #         my_bias = _MinSparseVector_to_cpp(bias)

    #     cdef vector[vector[pair[int, int]]] potentials = \
    #          vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
    #          MinSparseVector_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = MinSparseVector_zero()
    #         potentials[i] = _MinSparseVector_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _MinSparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _MinSparseVector_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MinSparseVector_one()
        else:
            my_bias = _MinSparseVector_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _MinSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MinSparseVector_one()
        else:
            my_bias = _MinSparseVector_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MinSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphMinSparseVectorPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _MinSparseVector_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _MinSparseVector_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _MinSparseVector:
    cdef _MinSparseVector init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = _MinSparseVector()
        created.thisval = _MinSparseVector_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _MinSparseVector_from_cpp(MinSparseVector_zero())

    @staticmethod
    def one_raw():
        return _MinSparseVector_from_cpp(MinSparseVector_one())

    @staticmethod
    def zero():
        return _MinSparseVector().init(MinSparseVector_zero())

    @staticmethod
    def one():
        return _MinSparseVector().init(MinSparseVector_one())

    def __add__(_MinSparseVector self, _MinSparseVector other):
        return _MinSparseVector().init(MinSparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(_MinSparseVector self, _MinSparseVector other):
        return _MinSparseVector().init(MinSparseVector_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _MinSparseVector_from_cpp(self.thisval)

cdef vector[pair[int, int]] _MinSparseVector_to_cpp(vector[pair[int, int]] val):
    return val


cdef _MinSparseVector_from_cpp(vector[pair[int, int]] val):
    return val

cdef class MinSparseVectorChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = MinSparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CMinSparseVectorChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _MinSparseVector_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _MinSparseVectorMarginals:
    cdef const CMinSparseVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CMinSparseVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _MinSparseVector_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _MinSparseVector_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have MinSparseVector marginal values." +
                "Passed %s." % obj)




class MinSparseVector:
    Chart = MinSparseVectorChart
    Marginals = _MinSparseVectorMarginals
    #Semi = _MinSparseVector
    Potentials = MinSparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               MinSparseVectorPotentials potentials):
        cdef MinSparseVectorChart chart = MinSparseVectorChart()
        chart.chart = inside_MinSparseVector(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                MinSparseVectorPotentials potentials,
                MinSparseVectorChart inside_chart):
        cdef MinSparseVectorChart out_chart = MinSparseVectorChart()
        out_chart.chart = outside_MinSparseVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          MinSparseVectorPotentials potentials):
        cdef const CMinSparseVectorMarginals *marginals = \
            MinSparseVector_compute(graph.thisptr, potentials.thisptr)
        return _MinSparseVectorMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         MinSparseVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class MaxSparseVectorPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = MaxSparseVector
        self.thisptr = NULL

    def times(self, MaxSparseVectorPotentials other):
        cdef CHypergraphMaxSparseVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return MaxSparseVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return MaxSparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMaxSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return MaxSparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphMaxSparseVectorPotentials *ptr = \
            cmake_projected_potentials_MaxSparseVector(self.thisptr,
                                                  projection.thisptr)
        return MaxSparseVectorPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _MaxSparseVector_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef vector[pair[int, int]] my_bias
    #     if bias is None:
    #         my_bias = MaxSparseVector_one()
    #     else:
    #         my_bias = _MaxSparseVector_to_cpp(bias)

    #     cdef vector[vector[pair[int, int]]] potentials = \
    #          vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
    #          MaxSparseVector_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = MaxSparseVector_zero()
    #         potentials[i] = _MaxSparseVector_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _MaxSparseVector_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        potentials,
                                        _MaxSparseVector_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MaxSparseVector_one()
        else:
            my_bias = _MaxSparseVector_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _MaxSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MaxSparseVector_one()
        else:
            my_bias = _MaxSparseVector_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MaxSparseVector_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphMaxSparseVectorPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _MaxSparseVector_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _MaxSparseVector_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _MaxSparseVector:
    cdef _MaxSparseVector init(self, vector[pair[int, int]] val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(vector[pair[int, int]] val):
        created = _MaxSparseVector()
        created.thisval = _MaxSparseVector_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _MaxSparseVector_from_cpp(MaxSparseVector_zero())

    @staticmethod
    def one_raw():
        return _MaxSparseVector_from_cpp(MaxSparseVector_one())

    @staticmethod
    def zero():
        return _MaxSparseVector().init(MaxSparseVector_zero())

    @staticmethod
    def one():
        return _MaxSparseVector().init(MaxSparseVector_one())

    def __add__(_MaxSparseVector self, _MaxSparseVector other):
        return _MaxSparseVector().init(MaxSparseVector_add(self.thisval,
                                                  other.thisval))

    def __mul__(_MaxSparseVector self, _MaxSparseVector other):
        return _MaxSparseVector().init(MaxSparseVector_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _MaxSparseVector_from_cpp(self.thisval)

cdef vector[pair[int, int]] _MaxSparseVector_to_cpp(vector[pair[int, int]] val):
    return val


cdef _MaxSparseVector_from_cpp(vector[pair[int, int]] val):
    return val

cdef class MaxSparseVectorChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = MaxSparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CMaxSparseVectorChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _MaxSparseVector_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _MaxSparseVectorMarginals:
    cdef const CMaxSparseVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CMaxSparseVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _MaxSparseVector_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _MaxSparseVector_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have MaxSparseVector marginal values." +
                "Passed %s." % obj)




class MaxSparseVector:
    Chart = MaxSparseVectorChart
    Marginals = _MaxSparseVectorMarginals
    #Semi = _MaxSparseVector
    Potentials = MaxSparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               MaxSparseVectorPotentials potentials):
        cdef MaxSparseVectorChart chart = MaxSparseVectorChart()
        chart.chart = inside_MaxSparseVector(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                MaxSparseVectorPotentials potentials,
                MaxSparseVectorChart inside_chart):
        cdef MaxSparseVectorChart out_chart = MaxSparseVectorChart()
        out_chart.chart = outside_MaxSparseVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          MaxSparseVectorPotentials potentials):
        cdef const CMaxSparseVectorMarginals *marginals = \
            MaxSparseVector_compute(graph.thisptr, potentials.thisptr)
        return _MaxSparseVectorMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         MaxSparseVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




cdef class CountingPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated
    with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = Counting
        self.thisptr = NULL

    def times(self, CountingPotentials other):
        cdef CHypergraphCountingPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return CountingPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return CountingPotentials(self.hypergraph).init(self.thisptr.clone(),
                                                          None)

    def project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphCountingPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return CountingPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, HypergraphMap projection):
        cdef CHypergraphCountingPotentials *ptr = \
            cmake_projected_potentials_Counting(self.thisptr,
                                                  projection.thisptr)
        return CountingPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s" % (edge.label, self[edge])
                          for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _Counting_from_cpp(self.thisptr.bias())

    # def build(self, fn, bias=None):
    #     """
    #     build(fn)

    #     Build the potential vector for a hypergraph.

    #     :param fn: A function from edge labels to potentials.
    #     """
    #     cdef int my_bias
    #     if bias is None:
    #         my_bias = Counting_one()
    #     else:
    #         my_bias = _Counting_to_cpp(bias)

    #     cdef vector[int] potentials = \
    #          vector[int](self.hypergraph.thisptr.edges().size(),
    #          Counting_zero())
    #     # cdef d result
    #     for i, ty in enumerate(self.hypergraph.labeling.edge_labels):
    #         result = fn(ty)
    #         if result is None: potentials[i] = Counting_zero()
    #         potentials[i] = _Counting_to_cpp(result)
    #     self.thisptr =  \
    #         cmake_potentials_Counting(self.hypergraph.thisptr,
    #                                    potentials, my_bias)
    #     return self

    def from_potentials(self, other_potentials):
        cdef vector[int] potentials = \
            vector[int](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _Counting_to_cpp(other_potentials[edge])

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        potentials,
                                        _Counting_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef int my_bias
        if bias is None:
            my_bias = Counting_one()
        else:
            my_bias = _Counting_to_cpp(bias)

        cdef vector[int] potentials = \
            vector[int](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _Counting_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef int my_bias
        if bias is None:
            my_bias = Counting_one()
        else:
            my_bias = _Counting_to_cpp(bias)

        cdef c_map.map[int, int] map_potentials
        cdef vector[int] potentials = \
            vector[int](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _Counting_to_cpp(v)

        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                        map_potentials,
                                        potentials, my_bias)
        return self

    cdef init(self, CHypergraphCountingPotentials *ptr,
              HypergraphMap projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _Counting_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _Counting_from_cpp(self.thisptr.dot(deref(path.thisptr)))


cdef class _Counting:
    cdef _Counting init(self, int val):
        self.thisval = val
        return self

    @staticmethod
    def from_value(int val):
        created = _Counting()
        created.thisval = _Counting_to_cpp(val)
        return created

    @staticmethod
    def zero_raw():
        return _Counting_from_cpp(Counting_zero())

    @staticmethod
    def one_raw():
        return _Counting_from_cpp(Counting_one())

    @staticmethod
    def zero():
        return _Counting().init(Counting_zero())

    @staticmethod
    def one():
        return _Counting().init(Counting_one())

    def __add__(_Counting self, _Counting other):
        return _Counting().init(Counting_add(self.thisval,
                                                  other.thisval))

    def __mul__(_Counting self, _Counting other):
        return _Counting().init(Counting_times(self.thisval,
                                                    other.thisval))

    property value:
        def __get__(self):
            return _Counting_from_cpp(self.thisval)

cdef int _Counting_to_cpp(int val):
    return val


cdef _Counting_from_cpp(int val):
    return val

cdef class CountingChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Counting
        self.chart = NULL
        if graph is not None:
            self.chart = new CCountingChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _Counting_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _CountingMarginals:
    cdef const CCountingMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CCountingMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _Counting_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _Counting_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Counting marginal values." +
                "Passed %s." % obj)



    def threshold(self, int semi):
        """
        TODO: fill in
        """
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi),
                                               None)



class Counting:
    Chart = CountingChart
    Marginals = _CountingMarginals
    #Semi = _Counting
    Potentials = CountingPotentials

    @staticmethod
    def inside(Hypergraph graph,
               CountingPotentials potentials):
        cdef CountingChart chart = CountingChart()
        chart.chart = inside_Counting(graph.thisptr,
                                        deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                CountingPotentials potentials,
                CountingChart inside_chart):
        cdef CountingChart out_chart = CountingChart()
        out_chart.chart = outside_Counting(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart



    @staticmethod
    def viterbi(Hypergraph graph,
                CountingPotentials potentials,
                CountingChart chart=None):
        cdef CCountingChart *used_chart
        cdef CBackPointers *used_back = \
            new CBackPointers(graph.thisptr)
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CCountingChart(graph.thisptr)
        viterbi_Counting(graph.thisptr,
                           deref(potentials.thisptr),
                           used_chart,
                           used_back)
        bp = BackPointers().init(used_back, graph)
        if chart is None:
            del used_chart
        return bp



    @staticmethod
    def compute_marginals(Hypergraph graph,
                          CountingPotentials potentials):
        cdef const CCountingMarginals *marginals = \
            Counting_compute(graph.thisptr, potentials.thisptr)
        return _CountingMarginals().init(marginals, graph)

    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         CountingPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)




####### Methods that use specific potential ########


class Potentials(LogViterbiPotentials):
    pass


cdef class BackPointers:
    """
    The back pointers generated by the Viterbi algorithm.

    Acts as a map::
       >> print bp[node]

    Gives the best back edge for node.

    Attributes
    -----------

    path: Hyperpath
       The best hyperpath from the root.
    """

    cdef BackPointers init(self, CBackPointers *ptr,
                           Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    property path:
        def __get__(self):
            cdef CHyperpath *path = self.thisptr.construct_path()
            return Path().init(path, self.graph)

    def __getitem__(self, Node node):
        return Edge().init(self.thisptr.get(node.nodeptr), self.graph)

    # def __dealloc__(self):
    #     del self.thisptr
    #     self.thisptr = NULL


def inside(Hypergraph graph, potentials):
    r"""
    inside(Hypergraph graph, Potentials potentials):

    Compute inside chart values for the given potentials.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph :math:`({\cal V}, {\cal E})` to search.

    potentials : :py:class:`Potentials`
      The potentials :math:`\theta` to use for inside computations.

    Returns
    -------

    chart : :py:class:`Chart`
       The inside chart. Type depends on potentials type, i.e.
       for inside potentials this will be the probability paths
       reaching this node.
    """
    return potentials.kind.inside(graph, potentials)


def outside(Hypergraph graph, potentials, inside_chart):
    r"""
    outside(Hypergraph graph, Potentials potentials, Chart inside_chart)

    Compute the outside scores for the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials :math:`\theta` to use for outside computations.

    inside_chart : :py:class:`Chart`
       The associated inside chart. Compute by calling
       :py:function:`inside`.  Must be the same type as potentials.

    Returns
    ---------

    chart : :py:class:`Chart`
       The outside chart. Type depends on potentials type, i.e. for
       inside potentials this will be the probability paths reaching
       this node.

    """
    return potentials.kind.outside(graph, potentials, inside_chart)


def best_path(Hypergraph graph, potentials, chart=None):
    r"""
    best_path(Hypergraph graph, Potentials potentials):

    Find the highest-scoring path
    :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
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
    bp = potentials.kind.viterbi(graph, potentials, chart)
    return bp.path


def prune_hypergraph(Hypergraph graph, potentials, thres):
    r"""
    prune_hypergraph(Hypergraph graph, potentials, thres)

    Prune hyperedges with low max-marginal score from the hypergraph.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials of the hypergraph.

    Returns
    --------
    (hypergraph, potentials) : :py:class:`Hypergraph`, :py:class:`Potentials`
       The new hypergraphs and potentials.
    """
    return potentials.kind.prune_hypergraph(graph, potentials, thres)


def compute_marginals(Hypergraph graph, potentials):
    r"""
    compute_marginals(Hypergraph graph, Potentials potentials):

    Compute marginals for hypergraph and potentials.

    Parameters
    -----------
    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    potentials : :py:class:`Potentials`
       The potentials of the hypergraph.

    Returns
    --------
    marginals : :py:class:`Marginals`
       The node and edge marginals associated with these potentials.
    """
    return potentials.kind.compute_marginals(graph, potentials)


class Chart(LogViterbiChart):
    r"""
    Chart :math:`S^{|{\cal V}|}` associated with a hypergraph (V, E)
    and semiring S.

    Acts as a vector::
       >> print chart[node]
    """
    pass


class Marginals(_LogViterbiMarginals):
    r"""
    Marginal values :math:`S^{|{\cal E} \times {\cal V}|}` associated
    with a hypergraph ({\cal V}, {\cal E}) and semiring S.

    Acts as a dictionary::
       >> print marginals[edge]
       >> print marginals[node]
    """
    pass

inside_values = inside
outside_values = outside


def make_pruning_projections(Hypergraph graph, BoolPotentials filt):
    """
    DEPRECATED

    Use project.
    """
    cdef const CHypergraphMap *projection = \
        cproject_hypergraph(graph.thisptr,
                            deref(filt.thisptr))
    return HypergraphMap().init(projection, graph, None)


def project(Hypergraph graph, BoolPotentials filter):
    """
    Prune a graph based on a set of boolean potentials.
    Edges with value 0 are pruned, edges with value
    1 are pruned if they are no longer in a path.

    Parameters
    -----------
    graph : Hypergraph

    filter : BoolPotentials
        The pruning filter to use.

    Returns
    --------
    map : HypergraphMap
        A map from the original graph to the new graph produced.
    """
    return make_pruning_projections(graph, filter)

def binarize(Hypergraph graph):
    """
    Binarize a hypergraph by making all k-ary edges right branching.

    Parameters
    ----------
    graph : Hypergraph

    Returns
    --------
    map : HypergraphMap
        A map from the original graph to the binary branching graph.
    """
    cdef CHypergraphMap *hypergraph_map = cbinarize(graph.thisptr)
    return HypergraphMap().init(hypergraph_map, graph, None)


def pairwise_dot(SparseVectorPotentials potentials,
                 vec,
                 LogViterbiPotentials weights):
    """
    DEPRECATED.

    1) Take the dot produce of each element of potentials and vector.
    2) Add this value to each element of weights.

    Parameters
    -----------
    potentials: SparseVectorPotentials
        A vector associated with each edge.

    vec: list-like
        A float vector

    weights: LogViterbiPotentials
        A mutable set of potentials.
    """
    cdef vector[double] rvec = vector[double]()
    for i in vec:
        rvec.push_back(<double>i)
    cpairwise_dot(deref(potentials.thisptr), rvec, weights.thisptr)

def extend_hypergraph_by_count(Hypergraph graph,
                               CountingPotentials potentials,
                               int lower_limit,
                               int upper_limit,
                               int goal):
    """
    DEPRECATED
    """

    cdef CHypergraphMap *projection = \
        cextend_hypergraph_by_count(graph.thisptr,
                                    deref(potentials.thisptr),
                                    lower_limit,
                                    upper_limit,
                                    goal)

    return HypergraphMap().init(projection, None, graph)


# def valid_binary_vectors(Bitset lhs, Bitset rhs):
#     return cvalid_binary_vectors(lhs.data, rhs.data)


# cdef class NodeUpdates:
#     def __cinit__(self, Hypergraph graph,
#                   SparseVectorPotentials potentials):
#         self.graph = graph
#         self.children = \
#             children_sparse(graph.thisptr,
#                             deref(potentials.thisptr))

#     def update(self, set[int] updates):
#         cdef set[int] *up = \
#             updated_nodes(self.graph.thisptr,
#                           deref(self.children),
#                           updates)
#         return deref(up)
