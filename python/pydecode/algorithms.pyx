cdef class BackPointers:
    cdef BackPointers init(self, CBackPointers *ptr,
                           Hypergraph graph):
        self.thisptr = ptr
        self.graph = graph
        return self

    property path:
        def __get__(self):
            return Path().init(construct_path(self.thisptr),
                               self.graph)

    def __getitem__(self, Node node):
        return Edge().init(self.thisptr.get(node.thisptr), graph)

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL


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
    Marginal values :math:`S^{|{\cal E} \times {\cal V}|}` associated with a
    hypergraph ({\cal V}, {\cal E}) and semiring S.

    Acts as a dictionary::
       >> print marginals[edge]
       >> print marginals[node]
    """
    pass

inside_values = inside
outside_values = outside


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
