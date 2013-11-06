#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

include "wrap.pxd"
include "hypergraph.pyx"

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

    CHyperpath *viterbi_{{S.type}}"general_viterbi<{{S.ctype}}>"(
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Weights theta) except +

    cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
        {{S.ctype}} marginal(const CHyperedge *edge)
        {{S.ctype}} marginal(const CHypernode *node)
        CHypergraphBoolWeights *threshold(
            const {{S.ctype}} &threshold)
        const CHypergraph *hypergraph()

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
    {{S.ctype}} {{S.type}}_add "{{S.ctype}}::add" ({{S.ctype}}, const {{S.ctype}})
    {{S.ctype}} {{S.type}}_times "{{S.ctype}}::times" ({{S.ctype}}, const {{S.ctype}})


cdef extern from "Hypergraph/Algorithms.h" namespace "{{S.ctype}}":
    cdef cppclass CHypergraph{{S.type}}Weights "HypergraphWeights<{{S.ctype}}>":
        {{S.ctype}} dot(const CHyperpath &path) except +
        {{S.ctype}} score(const CHyperedge *edge)
        CHypergraph{{S.type}}Weights *times(
            const CHypergraph{{S.type}}Weights &weights)
        CHypergraph{{S.type}}Weights *project_weights(
            const CHypergraphProjection)
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
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the weight vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = {{S.type}}

    def times(self, _{{S.type}}Weights other):
        cdef const CHypergraph{{S.type}}Weights *new_weights = \
            self.thisptr.times(deref(other.thisptr))
        return _{{S.type}}Weights(self.hypergraph).init(new_weights)

    def project(self, Hypergraph graph, Projection projection):
        cdef _{{S.type}}Weights new_weights = _{{S.type}}Weights(graph)
        cdef const CHypergraph{{S.type}}Weights *ptr = \
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
        cdef {{S.ctype}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = {{S.ctype}}(<{{S.vtype}}> bias)

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
                                           weights, my_bias)
        return self

    cdef init(self, const CHypergraph{{S.type}}Weights *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return _{{S.ptype}}().init(self.thisptr.score(edge.edgeptr)).value

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """
        return _{{S.ptype}}().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _{{S.ptype}}:
    cdef {{S.ctype}} wrap

    cdef init(self, {{S.ctype}} wrap):
        self.wrap = wrap
        return self

    {% if S.float %}
    def __float__(self):
        return <float>self.wrap
    {% endif %}

    {% if S.bool %}
    def __bool__(self):
        return <bool>self.wrap
    {% endif %}

    property value:
        def __get__(self):
            {% if S.float %}
            return <float>self.wrap
            {% elif S.bool %}
            return <bool>self.wrap
            {% else %}
            {{S.conversion}}
            {% endif %}

    def __repr__(self):
        return str(self.value)

    def __add__(_{{S.ptype}} self, _{{S.ptype}} other):
        return _{{S.ptype}}().init(
            {{S.type}}_add(self.wrap, other.wrap))

    def __mul__(_{{S.ptype}} self, _{{S.ptype}} other):
        return _{{S.ptype}}().init(
            {{S.type}}_times(self.wrap, other.wrap))

    @staticmethod
    def one():
        return _{{S.ptype}}().init({{S.type}}_one())

    @staticmethod
    def zero():
        return _{{S.ptype}}().init({{S.type}}_zero())

    def __cmp__(_{{S.ptype}} self, _{{S.ptype}} other):
        return cmp(self.value, other.value)

cdef class _{{S.type}}Chart:
    cdef C{{S.type}}Chart *chart
    cdef kind

    def __init__(self):
        self.kind = {{S.type}}

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
    {% if S.viterbi %}
    def threshold(self, _{{S.ptype}} semi):
        return _BoolWeights(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    {% endif %}

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
        out_chart.chart = outside_{{S.type}}(graph.thisptr,
                                             deref(weights.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    {% if S.viterbi %}
    @staticmethod
    def viterbi(Hypergraph graph,
                _{{S.type}}Weights weights):
        cdef CHyperpath *path = \
            viterbi_{{S.type}}(graph.thisptr,
                               deref(weights.thisptr))
        return Path().init(path)
    {% endif %}

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _{{S.type}}Weights weights):
        cdef const C{{S.type}}Marginals *marginals = \
            {{S.type}}_compute(graph.thisptr, weights.thisptr)
        return _{{S.type}}Marginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         _{{S.type}}Weights weights,
                         threshold):
        marginals = compute_marginals(graph, weights)
        bool_weights = marginals.threshold(_{{S.ptype}}().init({{S.ctype}}(<{{S.vtype}}>threshold)))
        projection = Projection(graph, bool_weights)
        new_graph = projection.project(graph)
        new_weight = weights.project(new_graph, projection)
        return new_graph, new_weight


def {{S.type}}Weights(Hypergraph graph):
    return {{S.type}}.Weights(graph)

{% endfor %}

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

def Weights(Hypergraph graph, kind=LogViterbi):
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
