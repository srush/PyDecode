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
        const CHypergraph{{S.type}}Potentials theta) except +

    C{{S.type}}Chart *outside_{{S.type}} "general_outside<{{S.ctype}}>" (
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        C{{S.type}}Chart inside_chart) except +

    CHyperpath *viterbi_{{S.type}}"general_viterbi<{{S.ctype}}>"(
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta) except +

    cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
        {{S.ctype}} marginal(const CHyperedge *edge)
        {{S.ctype}} marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const {{S.ctype}} &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass C{{S.type}}Chart "Chart<{{S.ctype}}>":
        {{S.ctype}} get(const CHypernode *node)

cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<{{S.ctype}}>":
    C{{S.type}}Marginals *{{S.type}}_compute "Marginals<{{S.ctype}}>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraph{{S.type}}Potentials *potentials)

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
    cdef cppclass CHypergraph{{S.type}}Potentials "HypergraphPotentials<{{S.ctype}}>":
        {{S.ctype}} dot(const CHyperpath &path) except +
        {{S.ctype}} score(const CHyperedge *edge)
        CHypergraph{{S.type}}Potentials *times(
            const CHypergraph{{S.type}}Potentials &potentials)
        CHypergraph{{S.type}}Potentials *project_potentials(
            const CHypergraphProjection)
        CHypergraph{{S.type}}Potentials(
            const CHypergraph *hypergraph,
            const vector[{{S.ctype}}] potentials,
            {{S.ctype}} bias) except +

cdef class _{{S.type}}Potentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """
    cdef Hypergraph hypergraph
    cdef const CHypergraph{{S.type}}Potentials *thisptr
    cdef kind
    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = {{S.type}}

    def times(self, _{{S.type}}Potentials other):
        cdef const CHypergraph{{S.type}}Potentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return _{{S.type}}Potentials(self.hypergraph).init(new_potentials)

    def project(self, Hypergraph graph, Projection projection):
        cdef _{{S.type}}Potentials new_potentials = _{{S.type}}Potentials(graph)
        cdef const CHypergraph{{S.type}}Potentials *ptr = \
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
        cdef {{S.ctype}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = {{S.ctype}}(<{{S.vtype}}> bias)

        cdef vector[{{S.ctype}}] potentials = \
             vector[{{S.ctype}}](self.hypergraph.thisptr.edges().size(),
             {{S.type}}_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = {{S.type}}_zero()
            potentials[i] = {{S.ctype}}(<{{S.vtype}}> result)
        self.thisptr =  \
          new CHypergraph{{S.type}}Potentials(self.hypergraph.thisptr,
                                           potentials, my_bias)
        return self

    cdef init(self, const CHypergraph{{S.type}}Potentials *ptr):
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

    def __cinit__(self, val=None):
        if val is not None:
            self.init({{S.ctype}}(<{{S.vtype}}>val))

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
        return _BoolPotentials(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi.wrap))
    {% endif %}

class {{S.type}}:
    Chart = _{{S.type}}Chart
    Marginals = _{{S.type}}Marginals
    Semi = _{{S.ptype}}
    Potentials = _{{S.type}}Potentials

    @staticmethod
    def inside(Hypergraph graph,
               _{{S.type}}Potentials potentials):
        cdef _{{S.type}}Chart chart = _{{S.type}}Chart()
        chart.chart = inside_{{S.type}}(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                _{{S.type}}Potentials potentials,
                _{{S.type}}Chart inside_chart):
        cdef _{{S.type}}Chart out_chart = _{{S.type}}Chart()
        out_chart.chart = outside_{{S.type}}(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    {% if S.viterbi %}
    @staticmethod
    def viterbi(Hypergraph graph,
                _{{S.type}}Potentials potentials):
        cdef CHyperpath *path = \
            viterbi_{{S.type}}(graph.thisptr,
                               deref(potentials.thisptr))
        return Path().init(path)
    {% endif %}

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          _{{S.type}}Potentials potentials):
        cdef const C{{S.type}}Marginals *marginals = \
            {{S.type}}_compute(graph.thisptr, potentials.thisptr)
        return _{{S.type}}Marginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         _{{S.type}}Potentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)
        bool_potentials = marginals.threshold(_{{S.ptype}}().init({{S.ctype}}(<{{S.vtype}}>threshold)))
        projection = Projection(graph, bool_potentials)
        new_graph = projection.project(graph)
        new_potential = potentials.project(new_graph, projection)
        return new_graph, new_potential


def {{S.type}}Potentials(Hypergraph graph):
    return {{S.type}}.Potentials(graph)

{% endfor %}

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

def Potentials(Hypergraph graph, kind=LogViterbi):
    return kind.Potentials(graph)

inside_values = inside
outside_values = outside


####### These are methods that use specific potential ########
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


# cdef extern from "Hypergraph/Algorithms.h":
#     const CHyperpath *best_constrained_path(
#         const CHypergraph *graph,
#         const CHypergraphLogViterbiPotentials theta,
#         const CHypergraphSparseVectorPotentials constraints) except +


def pairwise_dot(_SparseVectorPotentials potentials, vec):
    cdef vector[double] rvec
    for i in vec:
        rvec.push_back(<double>i)
    cdef const CHypergraphLogViterbiPotentials *rpotentials = \
        cpairwise_dot(deref(potentials.thisptr), rvec)
    return _LogViterbiPotentials(potentials.hypergraph).init(rpotentials)

cdef class Projection:
    cdef const CHypergraphProjection *thisptr

    def __init__(self, Hypergraph graph, _BoolPotentials filt):
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
