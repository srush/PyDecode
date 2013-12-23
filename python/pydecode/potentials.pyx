#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
cimport libcpp.map as c_map
from libcpp.pair cimport pair
from libcpp cimport bool

from wrap cimport *
from hypergraph cimport *
import hypergraph as py_hypergraph


cdef class Bitset:
    cdef init(self, cbitset data):
        self.data = data
        return self

    def __setitem__(self, int position, bool val):
        self.data.set(position, val)

    def __getitem__(self, int position):
        return self.data[position]

############# This is the templated semiring part. ##############



cdef class ViterbiPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return ViterbiPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return ViterbiPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphViterbiPotentials *ptr = \
            cmake_projected_potentials_Viterbi(self.thisptr, projection.thisptr)
        return ViterbiPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _ViterbiW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef double my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = _ViterbiW_to_cpp(bias)

        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size(),
             Viterbi_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = Viterbi_zero()
            potentials[i] = _ViterbiW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _ViterbiW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_Viterbi(
            self.hypergraph.thisptr,
            potentials,
            _ViterbiW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = _ViterbiW_to_cpp(bias)

        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _ViterbiW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Viterbi_one()
        else:
            my_bias = _ViterbiW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _ViterbiW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Viterbi(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphViterbiPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _ViterbiW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _ViterbiW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _ViterbiW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _ViterbiW:
    @staticmethod
    def one():
        return _ViterbiW_from_cpp(Viterbi_one())

    @staticmethod
    def zero():
        return _ViterbiW_from_cpp(Viterbi_zero())


cdef double _ViterbiW_to_cpp(double val):
    
    return val
    


cdef _ViterbiW_from_cpp(double val):
    
    return val
    


    # cdef double wrap

    # def __cmp__(_ViterbiW self, _ViterbiW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, double wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_ViterbiW self, _ViterbiW other):
    #     return _ViterbiW().init(
    #         Viterbi_add(self.wrap, other.wrap))

    # def __mul__(_ViterbiW self, _ViterbiW other):
    #     return _ViterbiW().init(
    #         Viterbi_times(self.wrap, other.wrap))

cdef class ViterbiChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Viterbi
        self.chart = NULL
        if graph is not None:
            self.chart = new CViterbiChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _ViterbiW_from_cpp(self.chart.get(node.nodeptr))

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
            return _ViterbiW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _ViterbiW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Viterbi marginal values." + \
                "Passed %s."%obj)

    
    def threshold(self, double semi):
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi), None)
    

class Viterbi:
    Chart = ViterbiChart
    Marginals = _ViterbiMarginals
    #Semi = _ViterbiW
    Potentials = ViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               ViterbiPotentials potentials):
        cdef ViterbiChart chart = ViterbiChart()
        chart.chart = inside_Viterbi(graph.thisptr, deref(potentials.thisptr))
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
        cdef vector[const CHyperedge *] *used_back = \
            new vector[const CHyperedge *]()
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CViterbiChart(graph.thisptr)
        cdef CHyperpath *path = \
            viterbi_Viterbi(graph.thisptr,
                               deref(potentials.thisptr),
                               used_chart,
                               used_back)
        if chart is None:
            del used_chart
        del used_back
        return Path().init(path, graph)

    @staticmethod
    def count_constrained_viterbi(Hypergraph graph,
                                  ViterbiPotentials potentials,
                                  CountingPotentials count_potentials,
                                  int limit):
        cdef CHyperpath *path = \
            count_constrained_viterbi_Viterbi(graph.thisptr,
                                                 deref(potentials.thisptr),
                                                 deref(count_potentials.thisptr),
                                                 limit)
        return Path().init(path, graph)

    

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
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return LogViterbiPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphLogViterbiPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return LogViterbiPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphLogViterbiPotentials *ptr = \
            cmake_projected_potentials_LogViterbi(self.thisptr, projection.thisptr)
        return LogViterbiPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _LogViterbiW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef double my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = _LogViterbiW_to_cpp(bias)

        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size(),
             LogViterbi_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = LogViterbi_zero()
            potentials[i] = _LogViterbiW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _LogViterbiW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_LogViterbi(
            self.hypergraph.thisptr,
            potentials,
            _LogViterbiW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = _LogViterbiW_to_cpp(bias)

        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _LogViterbiW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = LogViterbi_one()
        else:
            my_bias = _LogViterbiW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _LogViterbiW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_LogViterbi(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphLogViterbiPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _LogViterbiW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _LogViterbiW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _LogViterbiW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _LogViterbiW:
    @staticmethod
    def one():
        return _LogViterbiW_from_cpp(LogViterbi_one())

    @staticmethod
    def zero():
        return _LogViterbiW_from_cpp(LogViterbi_zero())


cdef double _LogViterbiW_to_cpp(double val):
    
    return val
    


cdef _LogViterbiW_from_cpp(double val):
    
    return val
    


    # cdef double wrap

    # def __cmp__(_LogViterbiW self, _LogViterbiW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, double wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_LogViterbiW self, _LogViterbiW other):
    #     return _LogViterbiW().init(
    #         LogViterbi_add(self.wrap, other.wrap))

    # def __mul__(_LogViterbiW self, _LogViterbiW other):
    #     return _LogViterbiW().init(
    #         LogViterbi_times(self.wrap, other.wrap))

cdef class LogViterbiChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = LogViterbi
        self.chart = NULL
        if graph is not None:
            self.chart = new CLogViterbiChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _LogViterbiW_from_cpp(self.chart.get(node.nodeptr))

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
            return _LogViterbiW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _LogViterbiW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have LogViterbi marginal values." + \
                "Passed %s."%obj)

    
    def threshold(self, double semi):
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi), None)
    

class LogViterbi:
    Chart = LogViterbiChart
    Marginals = _LogViterbiMarginals
    #Semi = _LogViterbiW
    Potentials = LogViterbiPotentials

    @staticmethod
    def inside(Hypergraph graph,
               LogViterbiPotentials potentials):
        cdef LogViterbiChart chart = LogViterbiChart()
        chart.chart = inside_LogViterbi(graph.thisptr, deref(potentials.thisptr))
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
        cdef vector[const CHyperedge *] *used_back = \
            new vector[const CHyperedge *]()
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CLogViterbiChart(graph.thisptr)
        cdef CHyperpath *path = \
            viterbi_LogViterbi(graph.thisptr,
                               deref(potentials.thisptr),
                               used_chart,
                               used_back)
        if chart is None:
            del used_chart
        del used_back
        return Path().init(path, graph)

    @staticmethod
    def count_constrained_viterbi(Hypergraph graph,
                                  LogViterbiPotentials potentials,
                                  CountingPotentials count_potentials,
                                  int limit):
        cdef CHyperpath *path = \
            count_constrained_viterbi_LogViterbi(graph.thisptr,
                                                 deref(potentials.thisptr),
                                                 deref(count_potentials.thisptr),
                                                 limit)
        return Path().init(path, graph)

    

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
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return InsidePotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphInsidePotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return InsidePotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphInsidePotentials *ptr = \
            cmake_projected_potentials_Inside(self.thisptr, projection.thisptr)
        return InsidePotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _InsideW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef double my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = _InsideW_to_cpp(bias)

        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size(),
             Inside_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = Inside_zero()
            potentials[i] = _InsideW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_Inside(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _InsideW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_Inside(
            self.hypergraph.thisptr,
            potentials,
            _InsideW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = _InsideW_to_cpp(bias)

        cdef vector[double] potentials = \
             vector[double](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _InsideW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Inside(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef double my_bias
        if bias is None:
            my_bias = Inside_one()
        else:
            my_bias = _InsideW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[double] potentials = \
            vector[double](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _InsideW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Inside(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphInsidePotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _InsideW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _InsideW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _InsideW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _InsideW:
    @staticmethod
    def one():
        return _InsideW_from_cpp(Inside_one())

    @staticmethod
    def zero():
        return _InsideW_from_cpp(Inside_zero())


cdef double _InsideW_to_cpp(double val):
    
    return val
    


cdef _InsideW_from_cpp(double val):
    
    return val
    


    # cdef double wrap

    # def __cmp__(_InsideW self, _InsideW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, double wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_InsideW self, _InsideW other):
    #     return _InsideW().init(
    #         Inside_add(self.wrap, other.wrap))

    # def __mul__(_InsideW self, _InsideW other):
    #     return _InsideW().init(
    #         Inside_times(self.wrap, other.wrap))

cdef class InsideChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Inside
        self.chart = NULL
        if graph is not None:
            self.chart = new CInsideChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _InsideW_from_cpp(self.chart.get(node.nodeptr))

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
            return _InsideW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _InsideW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Inside marginal values." + \
                "Passed %s."%obj)

    
    def threshold(self, double semi):
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi), None)
    

class Inside:
    Chart = InsideChart
    Marginals = _InsideMarginals
    #Semi = _InsideW
    Potentials = InsidePotentials

    @staticmethod
    def inside(Hypergraph graph,
               InsidePotentials potentials):
        cdef InsideChart chart = InsideChart()
        chart.chart = inside_Inside(graph.thisptr, deref(potentials.thisptr))
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
        cdef vector[const CHyperedge *] *used_back = \
            new vector[const CHyperedge *]()
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CInsideChart(graph.thisptr)
        cdef CHyperpath *path = \
            viterbi_Inside(graph.thisptr,
                               deref(potentials.thisptr),
                               used_chart,
                               used_back)
        if chart is None:
            del used_chart
        del used_back
        return Path().init(path, graph)

    @staticmethod
    def count_constrained_viterbi(Hypergraph graph,
                                  InsidePotentials potentials,
                                  CountingPotentials count_potentials,
                                  int limit):
        cdef CHyperpath *path = \
            count_constrained_viterbi_Inside(graph.thisptr,
                                                 deref(potentials.thisptr),
                                                 deref(count_potentials.thisptr),
                                                 limit)
        return Path().init(path, graph)

    

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
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return BoolPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphBoolPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return BoolPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphBoolPotentials *ptr = \
            cmake_projected_potentials_Bool(self.thisptr, projection.thisptr)
        return BoolPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _BoolW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef bool my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = _BoolW_to_cpp(bias)

        cdef vector[bool] potentials = \
             vector[bool](self.hypergraph.thisptr.edges().size(),
             Bool_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = Bool_zero()
            potentials[i] = _BoolW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_Bool(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[bool] potentials = \
             vector[bool](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _BoolW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_Bool(
            self.hypergraph.thisptr,
            potentials,
            _BoolW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef bool my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = _BoolW_to_cpp(bias)

        cdef vector[bool] potentials = \
             vector[bool](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _BoolW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Bool(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef bool my_bias
        if bias is None:
            my_bias = Bool_one()
        else:
            my_bias = _BoolW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[bool] potentials = \
            vector[bool](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _BoolW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Bool(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphBoolPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _BoolW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _BoolW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _BoolW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _BoolW:
    @staticmethod
    def one():
        return _BoolW_from_cpp(Bool_one())

    @staticmethod
    def zero():
        return _BoolW_from_cpp(Bool_zero())


cdef bool _BoolW_to_cpp(bool val):
    
    return val
    


cdef _BoolW_from_cpp(bool val):
    
    return val
    


    # cdef bool wrap

    # def __cmp__(_BoolW self, _BoolW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, bool wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_BoolW self, _BoolW other):
    #     return _BoolW().init(
    #         Bool_add(self.wrap, other.wrap))

    # def __mul__(_BoolW self, _BoolW other):
    #     return _BoolW().init(
    #         Bool_times(self.wrap, other.wrap))

cdef class BoolChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Bool
        self.chart = NULL
        if graph is not None:
            self.chart = new CBoolChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _BoolW_from_cpp(self.chart.get(node.nodeptr))

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
            return _BoolW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _BoolW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Bool marginal values." + \
                "Passed %s."%obj)

    
    def threshold(self, bool semi):
        return BoolPotentials(self.graph).init(self.thisptr.threshold(semi), None)
    

class Bool:
    Chart = BoolChart
    Marginals = _BoolMarginals
    #Semi = _BoolW
    Potentials = BoolPotentials

    @staticmethod
    def inside(Hypergraph graph,
               BoolPotentials potentials):
        cdef BoolChart chart = BoolChart()
        chart.chart = inside_Bool(graph.thisptr, deref(potentials.thisptr))
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
        cdef vector[const CHyperedge *] *used_back = \
            new vector[const CHyperedge *]()
        if chart is not None:
            used_chart = chart.chart
        else:
            used_chart = new CBoolChart(graph.thisptr)
        cdef CHyperpath *path = \
            viterbi_Bool(graph.thisptr,
                               deref(potentials.thisptr),
                               used_chart,
                               used_back)
        if chart is None:
            del used_chart
        del used_back
        return Path().init(path, graph)

    @staticmethod
    def count_constrained_viterbi(Hypergraph graph,
                                  BoolPotentials potentials,
                                  CountingPotentials count_potentials,
                                  int limit):
        cdef CHyperpath *path = \
            count_constrained_viterbi_Bool(graph.thisptr,
                                                 deref(potentials.thisptr),
                                                 deref(count_potentials.thisptr),
                                                 limit)
        return Path().init(path, graph)

    

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
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return SparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return SparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphSparseVectorPotentials *ptr = \
            cmake_projected_potentials_SparseVector(self.thisptr, projection.thisptr)
        return SparseVectorPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _SparseVectorW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = _SparseVectorW_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
             SparseVector_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = SparseVector_zero()
            potentials[i] = _SparseVectorW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _SparseVectorW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_SparseVector(
            self.hypergraph.thisptr,
            potentials,
            _SparseVectorW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = _SparseVectorW_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _SparseVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = SparseVector_one()
        else:
            my_bias = _SparseVectorW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _SparseVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_SparseVector(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphSparseVectorPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _SparseVectorW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _SparseVectorW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _SparseVectorW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _SparseVectorW:
    @staticmethod
    def one():
        return _SparseVectorW_from_cpp(SparseVector_one())

    @staticmethod
    def zero():
        return _SparseVectorW_from_cpp(SparseVector_zero())


cdef vector[pair[int, int]] _SparseVectorW_to_cpp(vector[pair[int, int]] val):
    
    return val
    


cdef _SparseVectorW_from_cpp(vector[pair[int, int]] val):
    
    return val
    


    # cdef vector[pair[int, int]] wrap

    # def __cmp__(_SparseVectorW self, _SparseVectorW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, vector[pair[int, int]] wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_SparseVectorW self, _SparseVectorW other):
    #     return _SparseVectorW().init(
    #         SparseVector_add(self.wrap, other.wrap))

    # def __mul__(_SparseVectorW self, _SparseVectorW other):
    #     return _SparseVectorW().init(
    #         SparseVector_times(self.wrap, other.wrap))

cdef class SparseVectorChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = SparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CSparseVectorChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _SparseVectorW_from_cpp(self.chart.get(node.nodeptr))

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
            return _SparseVectorW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _SparseVectorW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have SparseVector marginal values." + \
                "Passed %s."%obj)

    

class SparseVector:
    Chart = SparseVectorChart
    Marginals = _SparseVectorMarginals
    #Semi = _SparseVectorW
    Potentials = SparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               SparseVectorPotentials potentials):
        cdef SparseVectorChart chart = SparseVectorChart()
        chart.chart = inside_SparseVector(graph.thisptr, deref(potentials.thisptr))
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
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return MinSparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphMinSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return MinSparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphMinSparseVectorPotentials *ptr = \
            cmake_projected_potentials_MinSparseVector(self.thisptr, projection.thisptr)
        return MinSparseVectorPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _MinSparseVectorW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MinSparseVector_one()
        else:
            my_bias = _MinSparseVectorW_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
             MinSparseVector_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = MinSparseVector_zero()
            potentials[i] = _MinSparseVectorW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _MinSparseVectorW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_MinSparseVector(
            self.hypergraph.thisptr,
            potentials,
            _MinSparseVectorW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MinSparseVector_one()
        else:
            my_bias = _MinSparseVectorW_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _MinSparseVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MinSparseVector_one()
        else:
            my_bias = _MinSparseVectorW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MinSparseVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_MinSparseVector(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphMinSparseVectorPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _MinSparseVectorW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _MinSparseVectorW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _MinSparseVectorW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _MinSparseVectorW:
    @staticmethod
    def one():
        return _MinSparseVectorW_from_cpp(MinSparseVector_one())

    @staticmethod
    def zero():
        return _MinSparseVectorW_from_cpp(MinSparseVector_zero())


cdef vector[pair[int, int]] _MinSparseVectorW_to_cpp(vector[pair[int, int]] val):
    
    return val
    


cdef _MinSparseVectorW_from_cpp(vector[pair[int, int]] val):
    
    return val
    


    # cdef vector[pair[int, int]] wrap

    # def __cmp__(_MinSparseVectorW self, _MinSparseVectorW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, vector[pair[int, int]] wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_MinSparseVectorW self, _MinSparseVectorW other):
    #     return _MinSparseVectorW().init(
    #         MinSparseVector_add(self.wrap, other.wrap))

    # def __mul__(_MinSparseVectorW self, _MinSparseVectorW other):
    #     return _MinSparseVectorW().init(
    #         MinSparseVector_times(self.wrap, other.wrap))

cdef class MinSparseVectorChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = MinSparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CMinSparseVectorChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _MinSparseVectorW_from_cpp(self.chart.get(node.nodeptr))

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
            return _MinSparseVectorW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _MinSparseVectorW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have MinSparseVector marginal values." + \
                "Passed %s."%obj)

    

class MinSparseVector:
    Chart = MinSparseVectorChart
    Marginals = _MinSparseVectorMarginals
    #Semi = _MinSparseVectorW
    Potentials = MinSparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               MinSparseVectorPotentials potentials):
        cdef MinSparseVectorChart chart = MinSparseVectorChart()
        chart.chart = inside_MinSparseVector(graph.thisptr, deref(potentials.thisptr))
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
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return MaxSparseVectorPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphMaxSparseVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return MaxSparseVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphMaxSparseVectorPotentials *ptr = \
            cmake_projected_potentials_MaxSparseVector(self.thisptr, projection.thisptr)
        return MaxSparseVectorPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _MaxSparseVectorW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MaxSparseVector_one()
        else:
            my_bias = _MaxSparseVectorW_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size(),
             MaxSparseVector_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = MaxSparseVector_zero()
            potentials[i] = _MaxSparseVectorW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _MaxSparseVectorW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_MaxSparseVector(
            self.hypergraph.thisptr,
            potentials,
            _MaxSparseVectorW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MaxSparseVector_one()
        else:
            my_bias = _MaxSparseVectorW_to_cpp(bias)

        cdef vector[vector[pair[int, int]]] potentials = \
             vector[vector[pair[int, int]]](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _MaxSparseVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef vector[pair[int, int]] my_bias
        if bias is None:
            my_bias = MaxSparseVector_one()
        else:
            my_bias = _MaxSparseVectorW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[vector[pair[int, int]]] potentials = \
            vector[vector[pair[int, int]]](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _MaxSparseVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_MaxSparseVector(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphMaxSparseVectorPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _MaxSparseVectorW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _MaxSparseVectorW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _MaxSparseVectorW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _MaxSparseVectorW:
    @staticmethod
    def one():
        return _MaxSparseVectorW_from_cpp(MaxSparseVector_one())

    @staticmethod
    def zero():
        return _MaxSparseVectorW_from_cpp(MaxSparseVector_zero())


cdef vector[pair[int, int]] _MaxSparseVectorW_to_cpp(vector[pair[int, int]] val):
    
    return val
    


cdef _MaxSparseVectorW_from_cpp(vector[pair[int, int]] val):
    
    return val
    


    # cdef vector[pair[int, int]] wrap

    # def __cmp__(_MaxSparseVectorW self, _MaxSparseVectorW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, vector[pair[int, int]] wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_MaxSparseVectorW self, _MaxSparseVectorW other):
    #     return _MaxSparseVectorW().init(
    #         MaxSparseVector_add(self.wrap, other.wrap))

    # def __mul__(_MaxSparseVectorW self, _MaxSparseVectorW other):
    #     return _MaxSparseVectorW().init(
    #         MaxSparseVector_times(self.wrap, other.wrap))

cdef class MaxSparseVectorChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = MaxSparseVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CMaxSparseVectorChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _MaxSparseVectorW_from_cpp(self.chart.get(node.nodeptr))

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
            return _MaxSparseVectorW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _MaxSparseVectorW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have MaxSparseVector marginal values." + \
                "Passed %s."%obj)

    

class MaxSparseVector:
    Chart = MaxSparseVectorChart
    Marginals = _MaxSparseVectorMarginals
    #Semi = _MaxSparseVectorW
    Potentials = MaxSparseVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               MaxSparseVectorPotentials potentials):
        cdef MaxSparseVectorChart chart = MaxSparseVectorChart()
        chart.chart = inside_MaxSparseVector(graph.thisptr, deref(potentials.thisptr))
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



cdef class BinaryVectorPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        self.kind = BinaryVector
        self.thisptr = NULL

    def times(self, BinaryVectorPotentials other):
        cdef CHypergraphBinaryVectorPotentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return BinaryVectorPotentials(self.hypergraph).init(new_potentials, None)

    def clone(self):
        return BinaryVectorPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphBinaryVectorPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return BinaryVectorPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphBinaryVectorPotentials *ptr = \
            cmake_projected_potentials_BinaryVector(self.thisptr, projection.thisptr)
        return BinaryVectorPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _BinaryVectorW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef cbitset my_bias
        if bias is None:
            my_bias = BinaryVector_one()
        else:
            my_bias = _BinaryVectorW_to_cpp(bias)

        cdef vector[cbitset] potentials = \
             vector[cbitset](self.hypergraph.thisptr.edges().size(),
             BinaryVector_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = BinaryVector_zero()
            potentials[i] = _BinaryVectorW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_BinaryVector(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[cbitset] potentials = \
             vector[cbitset](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _BinaryVectorW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_BinaryVector(
            self.hypergraph.thisptr,
            potentials,
            _BinaryVectorW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef cbitset my_bias
        if bias is None:
            my_bias = BinaryVector_one()
        else:
            my_bias = _BinaryVectorW_to_cpp(bias)

        cdef vector[cbitset] potentials = \
             vector[cbitset](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _BinaryVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_BinaryVector(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef cbitset my_bias
        if bias is None:
            my_bias = BinaryVector_one()
        else:
            my_bias = _BinaryVectorW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[cbitset] potentials = \
            vector[cbitset](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _BinaryVectorW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_BinaryVector(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphBinaryVectorPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _BinaryVectorW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _BinaryVectorW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _BinaryVectorW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _BinaryVectorW:
    @staticmethod
    def one():
        return _BinaryVectorW_from_cpp(BinaryVector_one())

    @staticmethod
    def zero():
        return _BinaryVectorW_from_cpp(BinaryVector_zero())


cdef cbitset _BinaryVectorW_to_cpp(Bitset val):
    
    return val.data
    


cdef _BinaryVectorW_from_cpp(cbitset val):
    
    return Bitset().init(val)
    


    # cdef cbitset wrap

    # def __cmp__(_BinaryVectorW self, _BinaryVectorW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, cbitset wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_BinaryVectorW self, _BinaryVectorW other):
    #     return _BinaryVectorW().init(
    #         BinaryVector_add(self.wrap, other.wrap))

    # def __mul__(_BinaryVectorW self, _BinaryVectorW other):
    #     return _BinaryVectorW().init(
    #         BinaryVector_times(self.wrap, other.wrap))

cdef class BinaryVectorChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = BinaryVector
        self.chart = NULL
        if graph is not None:
            self.chart = new CBinaryVectorChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _BinaryVectorW_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart
        self.chart = NULL

cdef class _BinaryVectorMarginals:
    cdef const CBinaryVectorMarginals *thisptr
    cdef Hypergraph graph

    def __init__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    cdef init(self, const CBinaryVectorMarginals *ptr, Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _BinaryVectorW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _BinaryVectorW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have BinaryVector marginal values." + \
                "Passed %s."%obj)

    

class BinaryVector:
    Chart = BinaryVectorChart
    Marginals = _BinaryVectorMarginals
    #Semi = _BinaryVectorW
    Potentials = BinaryVectorPotentials

    @staticmethod
    def inside(Hypergraph graph,
               BinaryVectorPotentials potentials):
        cdef BinaryVectorChart chart = BinaryVectorChart()
        chart.chart = inside_BinaryVector(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                BinaryVectorPotentials potentials,
                BinaryVectorChart inside_chart):
        cdef BinaryVectorChart out_chart = BinaryVectorChart()
        out_chart.chart = outside_BinaryVector(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart


    

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          BinaryVectorPotentials potentials):
        cdef const CBinaryVectorMarginals *marginals = \
            BinaryVector_compute(graph.thisptr, potentials.thisptr)
        return _BinaryVectorMarginals().init(marginals, graph)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         BinaryVectorPotentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(threshold)
        return make_pruning_projections(graph, bool_potentials)



cdef class CountingPotentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

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
        return CountingPotentials(self.hypergraph).init(self.thisptr.clone(), None)

    def project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphCountingPotentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return CountingPotentials(graph).init(ptr, None)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraphCountingPotentials *ptr = \
            cmake_projected_potentials_Counting(self.thisptr, projection.thisptr)
        return CountingPotentials(graph).init(ptr, projection)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])


    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _CountingW_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef int my_bias
        if bias is None:
            my_bias = Counting_one()
        else:
            my_bias = _CountingW_to_cpp(bias)

        cdef vector[int] potentials = \
             vector[int](self.hypergraph.thisptr.edges().size(),
             Counting_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = Counting_zero()
            potentials[i] = _CountingW_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_Counting(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[int] potentials = \
             vector[int](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _CountingW_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_Counting(
            self.hypergraph.thisptr,
            potentials,
            _CountingW_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef int my_bias
        if bias is None:
            my_bias = Counting_one()
        else:
            my_bias = _CountingW_to_cpp(bias)

        cdef vector[int] potentials = \
             vector[int](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _CountingW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Counting(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self

    def from_map(self, in_map, bias=None):
        cdef int my_bias
        if bias is None:
            my_bias = Counting_one()
        else:
            my_bias = _CountingW_to_cpp(bias)

        cdef c_map.map[int,int] map_potentials
        cdef vector[int] potentials = \
            vector[int](len(in_map))

        for j, (key, v) in enumerate(in_map.iteritems()):
            map_potentials[key] = j
            potentials[j] = _CountingW_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_Counting(self.hypergraph.thisptr,
                                      map_potentials,
                                      potentials, my_bias)
        return self

    cdef init(self, CHypergraphCountingPotentials *ptr,
              Projection projection):
        self.thisptr = ptr
        self.projection = projection
        return self

    def __getitem__(self, Edge edge not None):
        return _CountingW_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _CountingW_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _CountingW().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _CountingW:
    @staticmethod
    def one():
        return _CountingW_from_cpp(Counting_one())

    @staticmethod
    def zero():
        return _CountingW_from_cpp(Counting_zero())


cdef int _CountingW_to_cpp(int val):
    
    return val
    


cdef _CountingW_from_cpp(int val):
    
    return val
    


    # cdef int wrap

    # def __cmp__(_CountingW self, _CountingW other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, int wrap):
    #     self.wrap = wrap
    #     return self

    # 

    # 

    # property value:
    #     def __get__(self):
    #         
    #         
    #         

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_CountingW self, _CountingW other):
    #     return _CountingW().init(
    #         Counting_add(self.wrap, other.wrap))

    # def __mul__(_CountingW self, _CountingW other):
    #     return _CountingW().init(
    #         Counting_times(self.wrap, other.wrap))

cdef class CountingChart:

    def __init__(self, Hypergraph graph=None):
        self.kind = Counting
        self.chart = NULL
        if graph is not None:
            self.chart = new CCountingChart(graph.thisptr)

    def __getitem__(self, Node node):
        return _CountingW_from_cpp(self.chart.get(node.nodeptr))

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
            return _CountingW_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _CountingW_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise py_hypergraph.HypergraphAccessException(
                "Only nodes and edges have Counting marginal values." + \
                "Passed %s."%obj)

    

class Counting:
    Chart = CountingChart
    Marginals = _CountingMarginals
    #Semi = _CountingW
    Potentials = CountingPotentials

    @staticmethod
    def inside(Hypergraph graph,
               CountingPotentials potentials):
        cdef CountingChart chart = CountingChart()
        chart.chart = inside_Counting(graph.thisptr, deref(potentials.thisptr))
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
       The inside chart. Type depends on potentials type, i.e. for inside potentials this
       will be the probability paths reaching this node.
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
    return potentials.kind.viterbi(graph, potentials, chart)

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

class Potentials(LogViterbiPotentials):
    pass


class Chart(LogViterbiChart):
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

def pairwise_dot(SparseVectorPotentials potentials, vec, LogViterbiPotentials weights):
    cdef vector[double] rvec
    for i in vec:
        rvec.push_back(<double>i)
    cpairwise_dot(deref(potentials.thisptr), rvec, weights.thisptr)


cdef class Projection:
    def __cinit__(self):
        self.thisptr = NULL

    cdef Projection init(self, const CHypergraphProjection *thisptr,
                         Hypergraph small_graph):
        self.thisptr = thisptr
        assert thisptr.big_graph().id() >= 0
        assert thisptr.new_graph().id() >= 0
        if small_graph == None:
            self.small_graph = self.small_hypergraph()
        else:
            self.small_graph = small_graph
        self.big_graph = self.big_hypergraph()
        return self

    def compose(self, Projection other, bool reverse):
        cdef CHypergraphProjection *newptr = \
            ccompose_projections(other.thisptr, reverse, self.thisptr)
        return Projection().init(newptr, None)


    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    def __getitem__(self, obj):
        cdef const CHyperedge *edge
        cdef const CHypernode *node
        if isinstance(obj, Edge):
            edge = self.thisptr.project((<Edge>obj).edgeptr)
            assert edge.id() >= 0
            assert edge.id() == self.small_graph.edges[edge.id()].id
            return self.small_graph.edges[edge.id()]
        if isinstance(obj, Node):
            node = self.thisptr.project((<Node>obj).nodeptr)
            assert node.id() >= 0
            return self.small_graph.nodes[node.id()]

    def project(self, Hypergraph graph):
        cdef Hypergraph new_graph = Hypergraph()
        cdef const CHypergraphProjection *projection = self.thisptr

        # Map nodes.
        node_labels = [None] * projection.new_graph().nodes().size()
        cdef vector[const CHypernode*] old_nodes = graph.thisptr.nodes()
        cdef const CHypernode *node
        for i in range(old_nodes.size()):
            node = projection.project(old_nodes[i])
            if node != NULL and node.id() >= 0:
                node_labels[node.id()] = graph.node_labels[i]

        # Map edges.
        edge_labels = [None] * projection.new_graph().edges().size()
        cdef vector[const CHyperedge *] old_edges = graph.thisptr.edges()
        cdef const CHyperedge *edge
        for i in range(old_edges.size()):
            edge = projection.project(old_edges[i])
            if edge != NULL and edge.id() >= 0:
                edge_labels[edge.id()] = graph.edge_labels[i]

        new_graph.init(projection.new_graph(), node_labels, edge_labels)
        return new_graph

    def small_hypergraph(self):
        cdef const CHypergraph *graph = self.thisptr.new_graph()
        return Hypergraph().init(graph, [], [])

    def big_hypergraph(self):
        cdef const CHypergraph *graph = self.thisptr.big_graph()
        assert graph.id() >= 0
        return Hypergraph().init(graph, [], [])


def make_pruning_projections(Hypergraph graph, BoolPotentials filt):
    cdef const CHypergraphProjection *projection = \
        cproject_hypergraph(graph.thisptr,
                            deref(filt.thisptr))
    return Projection().init(projection, None)

def valid_binary_vectors(Bitset lhs, Bitset rhs):
    return cvalid_binary_vectors(lhs.data, rhs.data)


def extend_hypergraph_by_count(Hypergraph graph,
                               CountingPotentials potentials,
                               int lower_limit,
                               int upper_limit,
                               int goal):
    cdef CHypergraphProjection *projection = \
        cextend_hypergraph_by_count(graph.thisptr,
                                    deref(potentials.thisptr),
                                    lower_limit,
                                    upper_limit,
                                    goal)
    return Projection().init(projection, graph)
