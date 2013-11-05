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

{% for S in semirings %}

cdef extern from "Hypergraph/Algorithms.h":
    C{{S.type}}Chart *inside_{{S.type}} "general_inside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Weights theta) except +

    C{{S.type}}Chart *outside_{{S.type}} "general_outside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Weights theta,
        C{{S.type}}Chart inside_chart) except +

    cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
        {{S.ctype}} marginal(const CHyperedge *edge)
        {{S.ctype}} marginal(const CHypernode *node)

    cdef cppclass C{{S.type}}Chart "Chart<{{S.ctype}}>":
        {{S.ctype}} get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<{{S.ctype}}>":
    C{{S.type}}Marginals *{{S.type}}_compute "Marginals<{{S.ctype}}>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraph{{S.type}}Weights *weights)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass {{S.ctype}}:
        {{S.ctype}}()
        {{S.ctype}}({{S.vtype}})
        double normalize(double)

cdef extern from "Hypergraph/Semirings.h" namespace "{{S.ctype}}":
    {{S.ctype}} {{S.type}}_one "{{S.ctype}}::one" ()
    {{S.ctype}} {{S.type}}_zero "{{S.ctype}}::zero" ()

cdef extern from "Hypergraph/Algorithms.h" namespace "{{S.ctype}}":
    cdef cppclass CHypergraph{{S.type}}Weights "HypergraphWeights<{{S.ctype}}>":
        {{S.ctype}} dot(const CHyperpath &path) except +
        double score(const CHyperedge *edge)
        CHypergraph{{S.type}}Weights *project_weights(
            const CHypergraphProjection )
        CHypergraph{{S.type}}Weights(
            const CHypergraph *hypergraph,
            const vector[{{S.ctype}}] weights,
            {{S.ctype}} bias) except +


cdef class _{{S.type}}Weights:
    r"""
    Weight vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print weights[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraph{{S.type}}Weights *thisptr

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
        cdef vector[{{S.ctype}}] weights = \
             vector[{{S.ctype}}](self.hypergraph.thisptr.edges().size(),
             {{S.type}}_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: weights[i] = {{S.type}}_zero()
            weights[i] = {{S.ctype}}(<{{S.vtype}}> result)
        self.thisptr =  \
          new CHypergraph{{S.type}}Weights(self.hypergraph.thisptr,
                                                  weights, {{S.type}}_one())
        return self

    cdef init(self, const CHypergraph{{S.type}}Weights *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return self.thisptr.score(edge.edgeptr)

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _{{S.ptype}}().init(self.thisptr.dot(deref(path.thisptr)))

cdef class _{{S.ptype}}:
    cdef {{S.ctype}} wrap

    cdef init(self, {{S.ctype}} wrap):
        self.wrap = wrap
        return self

    {% if S.float %}
    def __float__(self):
        return <float>self.wrap
    {% endif %}

cdef class _{{S.type}}Chart:
    cdef C{{S.type}}Chart *chart

    def __getitem__(self, Node node):
        return _{{S.ptype}}().init(self.chart.get(node.nodeptr))

cdef class _{{S.type}}Marginals:
    cdef const C{{S.type}}Marginals *thisptr

    cdef init(self, const C{{S.type}}Marginals *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _{{S.ptype}}().init(self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _{{S.ptype}}().init(self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have {{S.type}} marginal values." + \
                "Passed %s."%obj)




class {{S.type}}:

    Chart = _{{S.type}}Chart
    Marginals = _{{S.type}}Marginals
    Semi = _{{S.ptype}}
    Weights = _{{S.type}}Weights

    @staticmethod
    def inside(Hypergraph graph,
               _{{S.type}}Weights weights):
        cdef _{{S.type}}Chart chart = _{{S.type}}Chart()
        chart.chart = inside_{{S.type}}(graph.thisptr, deref(weights.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                _{{S.type}}Weights weights,
                _{{S.type}}Chart inside_chart):
        cdef _{{S.type}}Chart out_chart = _{{S.type}}Chart()
        out_chart.chart = outside_{{S.type}}(graph.thisptr, deref(weights.thisptr), deref(inside_chart.chart))
        return out_chart

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _{{S.type}}Weights weights):
        cdef const C{{S.type}}Marginals *marginals = \
            {{S.type}}_compute(graph.thisptr, weights.thisptr)
        return _{{S.type}}Marginals().init(marginals)

{% endfor %}


####### These are the non-templated versions, now obsolete ########
