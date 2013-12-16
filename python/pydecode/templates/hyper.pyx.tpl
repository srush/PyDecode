
#cython: embedsignature=True
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.pair cimport pair
from libcpp cimport bool

include "wrap.pxd"
include "hypergraph.pyx"
include "beam.pyx"



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

    CHyperpath *count_constrained_viterbi_{{S.type}} "count_constrained_viterbi<{{S.ctype}}>"(
        const CHypergraph *graph,
        const CHypergraph{{S.type}}Potentials theta,
        const CHypergraphCountingPotentials counts, int limit) except +

    cdef cppclass C{{S.type}}Marginals "Marginals<{{S.ctype}}>":
        {{S.vtype}} marginal(const CHyperedge *edge)
        {{S.vtype}} marginal(const CHypernode *node)
        CHypergraphBoolPotentials *threshold(
            const {{S.vtype}} &threshold)
        const CHypergraph *hypergraph()

    cdef cppclass C{{S.type}}Chart "Chart<{{S.ctype}}>":
        {{S.vtype}} get(const CHypernode *node)
        void insert(const CHypernode& node, const {{S.vtype}}& val)


cdef extern from "Hypergraph/Algorithms.h" namespace "Marginals<{{S.ctype}}>":
    C{{S.type}}Marginals *{{S.type}}_compute "Marginals<{{S.ctype}}>::compute" (
                           const CHypergraph *hypergraph,
                           const CHypergraph{{S.type}}Potentials *potentials)

cdef extern from "Hypergraph/Semirings.h":
    cdef cppclass {{S.ctype}}:
        pass

    cdef cppclass CHypergraph{{S.type}}Potentials "HypergraphPotentials<{{S.ctype}}>":
        {{S.vtype}} dot(const CHyperpath &path) except +
        {{S.vtype}} score(const CHyperedge *edge)
        CHypergraph{{S.type}}Potentials *times(
            const CHypergraph{{S.type}}Potentials &potentials)
        CHypergraph{{S.type}}Potentials *project_potentials(
            const CHypergraphProjection)
        CHypergraph{{S.type}}Potentials(
            const CHypergraph *hypergraph,
            const vector[{{S.vtype}}] potentials,
            {{S.vtype}} bias) except +
        {{S.vtype}} bias()
        CHypergraph{{S.type}}Potentials *clone() const

cdef extern from "Hypergraph/Semirings.h" namespace "HypergraphVectorPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_potentials_{{S.type}} "HypergraphVectorPotentials<{{S.ctype}}>::make_potentials" (
        const CHypergraph *hypergraph,
        const vector[{{S.vtype}}] potentials,
        {{S.vtype}} bias) except +

cdef extern from "Hypergraph/Semirings.h" namespace "HypergraphProjectedPotentials<{{S.ctype}}>":
    CHypergraph{{S.type}}Potentials *cmake_projected_potentials_{{S.type}} "HypergraphProjectedPotentials<{{S.ctype}}>::make_potentials" (
        CHypergraph{{S.type}}Potentials *base_potentials,
        const CHypergraphProjection *projection) except +


cdef extern from "Hypergraph/Semirings.h" namespace "{{S.ctype}}":
    {{S.vtype}} {{S.type}}_one "{{S.ctype}}::one" ()
    {{S.vtype}} {{S.type}}_zero "{{S.ctype}}::zero" ()
    {{S.vtype}} {{S.type}}_add "{{S.ctype}}::add" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_times "{{S.ctype}}::times" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_safeadd "{{S.ctype}}::safe_add" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_safetimes "{{S.ctype}}::safe_times" ({{S.vtype}}, const {{S.vtype}}&)
    {{S.vtype}} {{S.type}}_normalize "{{S.ctype}}::normalize" ({{S.vtype}}&)



cdef class {{S.type}}Potentials:
    r"""
    Potential vector :math:`\theta \in R^{|{\cal E}|}` associated with a hypergraph.

    Acts as a dictionary::
       >> print potentials[edge]
    """
    cdef Hypergraph hypergraph
    cdef CHypergraph{{S.type}}Potentials *thisptr
    cdef kind

    def __dealloc__(self):
        del self.thisptr

    def __cinit__(self, Hypergraph graph):
        """
        Build the potential vector for a hypergraph.

        :param hypergraph: The underlying hypergraph.
        """
        self.hypergraph = graph
        self.kind = {{S.type}}

    def times(self, {{S.type}}Potentials other):
        cdef CHypergraph{{S.type}}Potentials *new_potentials = \
            self.thisptr.times(deref(other.thisptr))
        return {{S.type}}Potentials(self.hypergraph).init(new_potentials)

    def clone(self):
        return {{S.type}}Potentials(self.hypergraph).init(self.thisptr.clone())

    def project(self, Hypergraph graph, Projection projection):
        cdef {{S.type}}Potentials new_potentials = {{S.type}}Potentials(graph)
        cdef CHypergraph{{S.type}}Potentials *ptr = \
            self.thisptr.project_potentials(deref(projection.thisptr))
        return new_potentials.init(ptr)

    def up_project(self, Hypergraph graph, Projection projection):
        cdef CHypergraph{{S.type}}Potentials *ptr = \
            cmake_projected_potentials_{{S.type}}(self.thisptr, projection.thisptr)
        return {{S.type}}Potentials(graph).init(ptr)

    def show(self, Hypergraph graph):
        return "\n".join(["%20s : %s"%(graph.label(edge), self[edge])
           for edge in graph.edges])

    property kind:
        def __get__(self):
            return self.kind

    property bias:
        def __get__(self):
            return _{{S.ptype}}_from_cpp(self.thisptr.bias())

    def build(self, fn, bias=None):
        """
        build(fn)

        Build the potential vector for a hypergraph.

        :param fn: A function from edge labels to potentials.
        """
        cdef {{S.vtype}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = _{{S.ptype}}_to_cpp(bias)

        cdef vector[{{S.vtype}}] potentials = \
             vector[{{S.vtype}}](self.hypergraph.thisptr.edges().size(),
             {{S.type}}_zero())
        # cdef d result
        for i, ty in enumerate(self.hypergraph.edge_labels):
            result = fn(ty)
            if result is None: potentials[i] = {{S.type}}_zero()
            potentials[i] = _{{S.ptype}}_to_cpp(result)
        self.thisptr =  \
            cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                       potentials, my_bias)
        return self

    def from_potentials(self, other_potentials):
        cdef vector[{{S.vtype}}] potentials = \
             vector[{{S.vtype}}](self.hypergraph.thisptr.edges().size())

        for i, edge in enumerate(self.hypergraph.edges):
            potentials[i] = _{{S.ptype}}_to_cpp(other_potentials[edge])

        self.thisptr =  \
          cmake_potentials_{{S.type}}(
            self.hypergraph.thisptr,
            potentials,
            _{{S.ptype}}_to_cpp(other_potentials.bias))

        return self

    def from_vector(self, in_vec, bias=None):
        cdef {{S.vtype}} my_bias
        if bias is None:
            my_bias = {{S.type}}_one()
        else:
            my_bias = _{{S.ptype}}_to_cpp(bias)

        cdef vector[{{S.vtype}}] potentials = \
             vector[{{S.vtype}}](self.hypergraph.thisptr.edges().size())

        for i, v in enumerate(in_vec):
            potentials[i] = _{{S.ptype}}_to_cpp(v)

        self.thisptr =  \
          cmake_potentials_{{S.type}}(self.hypergraph.thisptr,
                                      potentials, my_bias)
        return self


    cdef init(self, CHypergraph{{S.type}}Potentials *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, Edge edge not None):
        return _{{S.ptype}}_from_cpp(self.thisptr.score(edge.edgeptr))

    def dot(self, Path path not None):
        r"""
        dot(path)

        Take the dot product with `path` :math:`\theta^{\top} y`.
        """

        return _{{S.ptype}}_from_cpp(self.thisptr.dot(deref(path.thisptr)))
        #return _{{S.ptype}}().init(self.thisptr.dot(deref(path.thisptr))).value

cdef class _{{S.ptype}}:
    @staticmethod
    def one():
        return _{{S.ptype}}_from_cpp({{S.type}}_one())

    @staticmethod
    def zero():
        return _{{S.ptype}}_from_cpp({{S.type}}_zero())


cdef {{S.vtype}} _{{S.ptype}}_to_cpp({{S.intype}} val):
    {% if 'to_cpp' in S %}
    return {{S.to_cpp}}
    {% else %}
    return val
    {% endif %}


cdef _{{S.ptype}}_from_cpp({{S.vtype}} val):
    {% if 'from_cpp' in S %}
    return {{S.from_cpp}}
    {% else %}
    return val
    {% endif %}


    # cdef {{S.vtype}} wrap

    # def __cmp__(_{{S.ptype}} self, _{{S.ptype}} other):
    #     return cmp(self.value, other.value)


    # def __cinit__(self, val=None):
    #     if val is not None:
    #         self.init(val)

    # cdef init(self, {{S.vtype}} wrap):
    #     self.wrap = wrap
    #     return self

    # {% if S.float %}
    # def __float__(self):
    #     return <float>self.wrap
    # {% endif %}

    # {% if S.bool %}
    # def __bool__(self):
    #     return <bool>self.wrap
    # {% endif %}

    # property value:
    #     def __get__(self):
    #         {% if S.float %}
    #         return <float>self.wrap
    #         {% elif S.bool %}
    #         return <bool>self.wrap
    #         {% else %}
    #         {{S.conversion}}
    #         {% endif %}

    # def __repr__(self):
    #     return str(self.value)

    # def __add__(_{{S.ptype}} self, _{{S.ptype}} other):
    #     return _{{S.ptype}}().init(
    #         {{S.type}}_add(self.wrap, other.wrap))

    # def __mul__(_{{S.ptype}} self, _{{S.ptype}} other):
    #     return _{{S.ptype}}().init(
    #         {{S.type}}_times(self.wrap, other.wrap))

cdef class _{{S.type}}Chart:
    cdef C{{S.type}}Chart *chart
    cdef kind

    def __init__(self):
        self.kind = {{S.type}}

    def __getitem__(self, Node node):
        return _{{S.ptype}}_from_cpp(self.chart.get(node.nodeptr))

    def __dealloc__(self):
        del self.chart


cdef class _{{S.type}}Marginals:
    cdef const C{{S.type}}Marginals *thisptr

    cdef init(self, const C{{S.type}}Marginals *ptr):
        self.thisptr = ptr
        return self

    def __getitem__(self, obj):
        if isinstance(obj, Edge):
            return _{{S.ptype}}_from_cpp(
                self.thisptr.marginal((<Edge>obj).edgeptr))
        elif isinstance(obj, Node):
            return _{{S.ptype}}_from_cpp(
                self.thisptr.marginal((<Node>obj).nodeptr))
        else:
            raise HypergraphAccessException(
                "Only nodes and edges have {{S.type}} marginal values." + \
                "Passed %s."%obj)
    {% if S.viterbi %}
    def threshold(self, {{S.vtype}} semi):
        return BoolPotentials(Hypergraph().init(self.thisptr.hypergraph())) \
            .init(self.thisptr.threshold(semi))
    {% endif %}

class {{S.type}}:
    Chart = _{{S.type}}Chart
    Marginals = _{{S.type}}Marginals
    #Semi = _{{S.ptype}}
    Potentials = {{S.type}}Potentials

    @staticmethod
    def inside(Hypergraph graph,
               {{S.type}}Potentials potentials):
        cdef _{{S.type}}Chart chart = _{{S.type}}Chart()
        chart.chart = inside_{{S.type}}(graph.thisptr, deref(potentials.thisptr))
        return chart

    @staticmethod
    def outside(Hypergraph graph,
                {{S.type}}Potentials potentials,
                _{{S.type}}Chart inside_chart):
        cdef _{{S.type}}Chart out_chart = _{{S.type}}Chart()
        out_chart.chart = outside_{{S.type}}(graph.thisptr,
                                             deref(potentials.thisptr),
                                             deref(inside_chart.chart))
        return out_chart

    {% if S.viterbi %}
    @staticmethod
    def viterbi(Hypergraph graph,
                {{S.type}}Potentials potentials):
        cdef CHyperpath *path = \
            viterbi_{{S.type}}(graph.thisptr,
                               deref(potentials.thisptr))
        return Path().init(path, graph)

    @staticmethod
    def count_constrained_viterbi(Hypergraph graph,
                                  {{S.type}}Potentials potentials,
                                  CountingPotentials count_potentials,
                                  int limit):
        cdef CHyperpath *path = \
            count_constrained_viterbi_{{S.type}}(graph.thisptr,
                                                 deref(potentials.thisptr),
                                                 deref(count_potentials.thisptr),
                                                 limit)
        return Path().init(path, graph)

    {% endif %}

    @staticmethod
    def compute_marginals(Hypergraph graph,
                          {{S.type}}Potentials potentials):
        cdef const C{{S.type}}Marginals *marginals = \
            {{S.type}}_compute(graph.thisptr, potentials.thisptr)
        return _{{S.type}}Marginals().init(marginals)


    @staticmethod
    def prune_hypergraph(Hypergraph graph,
                         {{S.type}}Potentials potentials,
                         threshold):
        marginals = compute_marginals(graph, potentials)

        bool_potentials = marginals.threshold(
            threshold)
        projection = Projection(graph, bool_potentials)
        new_graph = projection.project(graph)
        new_potential = potentials.project(new_graph, projection)
        return new_graph, new_potential


{% endfor %}

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

def best_path(Hypergraph graph, potentials):
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
    return potentials.kind.viterbi(graph, potentials)

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
        const CHyperedge *project(const CHyperedge *edge)
        const CHypernode *project(const CHypernode *node)
        const CHypergraph *big_graph() const
        const CHypergraph *new_graph() const


    void cpairwise_dot "pairwise_dot"(
        const CHypergraphSparseVectorPotentials sparse_potentials,
        const vector[double] vec,
        CHypergraphLogViterbiPotentials *)

    bool cvalid_binary_vectors "valid_binary_vectors" (cbitset lhs,
                                                       cbitset rhs)


cdef extern from "Hypergraph/Semirings.h" namespace "HypergraphProjection":
    CHypergraphProjection *cproject_hypergraph "HypergraphProjection::project_hypergraph"(
        const CHypergraph *hypergraph,
        const CHypergraphBoolPotentials edge_mask)


    CHypergraphProjection *ccompose_projections "HypergraphProjection::compose_projections" (const CHypergraphProjection *projection1,
                                                                                             bool reverse1,
                                                                                             const CHypergraphProjection *projection2)


cdef extern from "Hypergraph/BeamSearch.h":
    cdef cppclass CBeamHyp "BeamHyp":
        cbitset sig
        double current_score
        double future_score

    cdef cppclass CBeamChart "BeamChart":
        CHyperpath *get_path(int result)
        vector[CBeamHyp *] get_beam(const CHypernode *node)

    cdef cppclass CBeamGroups "BeamGroups":
        CBeamGroups(const CHypergraph *graph,
                    const vector[int] groups,
                    const vector[int] group_limit,
                    int num_groups)

    CBeamChart *cbeam_search "beam_search" (
        const CHypergraph *graph,
        const CHypergraphLogViterbiPotentials &potentials,
        const CHypergraphBinaryVectorPotentials &constraints,
        const CLogViterbiChart &outside,
        double lower_bound,
        const CBeamGroups &groups)


cdef class BeamChart:
    cdef CBeamChart *thisptr
    cdef Hypergraph graph

    cdef init(self, CBeamChart *chart, Hypergraph graph):
        self.thisptr = chart
        self.graph = graph
        return self

    def path(self, int result):
        return Path().init(self.thisptr.get_path(result),
                           self.graph)

    def __getitem__(self, Node node):
        cdef vector[CBeamHyp *] beam = self.thisptr.get_beam(node.nodeptr)
        data = []
        i = 0
        for p in beam:
            data.append((Bitset().init(p.sig),
                         p.current_score,
                         p.future_score))
        return data

def beam_search(Hypergraph graph,
                LogViterbiPotentials potentials,
                BinaryVectorPotentials constraints,
                _LogViterbiChart outside,
                double lower_bound,
                groups,
                group_limits,
                int num_groups):
    cdef vector[int] cgroups = groups
    cdef vector[int] cgroup_limits = group_limits
    cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
                                                    cgroups,
                                                    cgroup_limits,
                                                    num_groups)
    # cgroups.resize(graph.nodes_size())
    # cdef vector[int] cgroup_limits
    # cgroups.resize(graph.nodes_size())

    # for i, group in enumerate(groups):
    #     cgroups[i] = group

    cdef CBeamChart *chart = \
        cbeam_search(graph.thisptr,
                     deref(potentials.thisptr),
                     deref(constraints.thisptr),
                     deref(outside.chart),
                     lower_bound,
                     deref(beam_groups)
)
    return BeamChart().init(chart, graph)


def pairwise_dot(SparseVectorPotentials potentials, vec, LogViterbiPotentials weights):
    cdef vector[double] rvec
    for i in vec:
        rvec.push_back(<double>i)
    cpairwise_dot(deref(potentials.thisptr), rvec, weights.thisptr)


cdef class Projection:
    cdef const CHypergraphProjection *thisptr
    cdef Hypergraph small_graph

    cdef Projection init(self, const CHypergraphProjection *thisptr, Hypergraph small_graph=None):
        self.thisptr = thisptr
        self.small_graph = small_graph
        return self

    def compose(self, Projection other, bool reverse):
        cdef CHypergraphProjection *newptr = \
            ccompose_projections(other.thisptr, reverse, self.thisptr)
        return Projection().init(newptr, self.small_graph)

    def __dealloc__(self):
        del self.thisptr

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
        return Hypergraph().init(graph, [], [])

def make_pruning_projections(Hypergraph graph, BoolPotentials filt):
    cdef const CHypergraphProjection *projection = \
        cproject_hypergraph(graph.thisptr,
                            deref(filt.thisptr))

    return Projection(graph).init(projection)

def valid_binary_vectors(Bitset lhs, Bitset rhs):
    return cvalid_binary_vectors(lhs.data, rhs.data)


cdef extern from "Hypergraph/Algorithms.h":
    CHypergraphProjection *cextend_hypergraph_by_count "extend_hypergraph_by_count" (
        CHypergraph *graph,
        CHypergraphCountingPotentials potentials,
        int lower_limit,
        int upper_limit,
        int goal)

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
    return Projection(graph).init(projection, graph)
