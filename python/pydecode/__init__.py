from pydecode.potentials import *

def inside(graph, potentials,
           kind=LogViterbi, chart=None):
    r"""
    Compute the inside values for potentials.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    potentials : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    kind : A semiring type.
      The semiring to use. Must agree with potentials.

    chart : Mx1 column vector.
      A chart buffer to reuse.
    Returns
    -------

    chart : Mx1 column vector (type depends on `kind`).
       The inside chart. Type depends on potentials type, i.e.
       for inside potentials this will be the probability paths
       reaching this vertex.
    """
    new_potentials = get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.inside(graph, new_potentials, chart)


def outside(graph, potentials, inside_chart,
            kind=LogViterbi, chart=None):
    r"""
    Compute the outside values for potentials.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    potentials : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    inside_chart : :py:class:`Chart`
       The associated inside chart. Compute by calling
       :py:function:`inside`.  Must be the same type as potentials.

    kind : A semiring type.
      The semiring to use. Must agree with potentials.

    chart : Mx1 column vector.
      A chart buffer to reuse.

    Returns
    ---------

    chart : Mx1 column vector (type depends on `kind`).
       The outside chart. Type depends on potentials type, i.e. for
       inside potentials this will be the probability paths reaching
       this node.

    """
    new_potentials = get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.outside(graph, new_potentials, inside_chart, chart)



def best_path(graph, potentials,
              kind=LogViterbi, chart=None):
    r"""
    Find the best path through a hypergraph for a given set of potentials.

    Formally gives
    :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    potentials : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    kind : A semiring type.
      The semiring to use. Must agree with potentials.

    chart : Mx1 column vector.
      A chart buffer to reuse.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """
    new_potentials = get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.viterbi(graph, new_potentials, chart)

def marginals(graph, potentials,
              inside_chart=None,
              outside_chart=None,
              kind=LogViterbi):
    r"""
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
    my_inside = inside_chart
    if my_inside is None:
        my_inside = inside(graph, potentials, kind=kind)

    my_outside = outside_chart
    if my_outside is None:
        my_outside = outside(graph, potentials, inside_chart=my_inside, kind=kind)

    new_potentials = get_potentials(graph, potentials, kind.Potentials)
    return new_potentials.kind.compute_marginals(graph, new_potentials,
                                                 my_inside, my_outside)



# Higher-level interface.

def _check_output_potentials(dp, out_potentials):
    if dp.outputs.size != out_potentials.size:
        raise ValueError("Potentials do not match output shape: %s != %s"%(out_potentials.shape, dep.outputs.shape))

def path_output(dp, path):
    """
    Convert a hypergraph path to the corresponding
    dynamic programming outputs.

    Parameters
    -----------
    dp : DynamicProgram

    path : Path

    Returns
    --------
    outputs : outputs x output_width matrix
       Matrix of outputs. Output_width is the width of an output
       and outputs is the numbe of non-zero outputs in the path.
    """
    int_output = (dp.output_matrix * path.v).nonzero()[0]
    return np.array(np.unravel_index(int_output,
                                     dp.outputs.shape)).T

def vertex_items(dp):
    """
    Reconstructs the items associated with each
    hypergraph vertex.

    Relatively slow, mainly for debugging.

    Parameters
    -----------
    dp : DynamicProgram

    Returns
    -------
    items : |V| x item_width matrix
       Matrix of items. Item_width is the width of an item
       and |V| is the number of vertices.
    """
    labels = [dp.item_matrix.T[vertex.id].nonzero()[1]
              for vertex in dp.hypergraph.vertices]
    return [zip(*np.unravel_index(l, dp.items.shape))[0]
            for l in labels]

def hyperedge_outputs(dp):
    """
    Reconstructs the outpus associated with each
    hypergraph hyperedge.

    Relatively slow, mainly for debugging.

    Returns
    -------
    items : |E| x output_width matrix
       Matrix of items. Output width is the width of an output
       and |E| is the number of hyperedges.
    """
    return np.unravel_index(dp.output_matrix.nonzero()[0],
                            dp.outputs.shape)


def argmax(dp, out_potentials,
           kind=LogViterbi, chart=None):
    """
    Find the highest scoring output structure in a
    dynamic program.

    Returns
    -------
    outputs : output x output_width matrix
       Matrix of outputs.
    """
    _check_output_potentials(dp, out_potentials)
    potentials = dp.output_matrix.T * out_potentials.ravel()
    path = best_path(dp.hypergraph, potentials,
                     kind, chart)
    return path_output(dp, path)


def fill(dp, out_potentials, kind=LogViterbi, chart=None):
    """
    Fill in a dynamic programming chart based on a set of potentials.

    Parameters
    ----------
    dp : DynamicProgram

    out_potentials : array
       An array in the shape of dp.outputs

    Returns
    -------
    chart : array
       An array in the shape of items.
    """
    _check_output_potentials(dp, out_potentials)
    potentials = dp.output_matrix.T * out_potentials.ravel()
    new_chart = inside(dp.hypergraph, potentials,
                   kind, chart)
    return new_chart.reshape(dp.items.shape)


def output_marginals(dp,
                     out_potentials,
                     kind=LogViterbi):
    """
    Compute marginals for the outputs of a dynamic program.

    Parameters
    ----------
    dp : DynamicProgram

    out_potentials : array

    Returns
    -------
    output_marginals : array
       An array in the shape of dp.outputs with marginal values.
    """
    _check_output_potentials(dp, out_potentials)
    potentials = dp.output_matrix.T * out_potentials.ravel()
    _, edge_marginals = marginals(dp.hypergraph,
                                  potentials, None, None, kind)
    return (dp.output_matrix * edge_marginals).reshape(
        dp.outputs.shape)

def item_marginals(dp,
                   out_potentials,
                   kind=LogViterbi):
    """
    Compute marginals for the outputs of a dynamic program.

    Parameters
    ----------
    dp : DynamicProgram

    out_potentials : array

    Returns
    -------
    item_marginals : array
       An array in the shape of dp.items with marginal values.
    """
    _check_output_potentials(dp, out_potentials)
    potentials = dp.output_matrix.T * out_potentials.ravel()
    node_marginals, _ = marginals(dp.hypergraph,
                                  potentials, None, None, kind)
    return (dp.item_matrix* node_marginals).reshape(
        dp.items.shape)

def score_outputs(dp, outputs, out_potentials,
                  kind=LogViterbi):
    """

    """

    indices = np.ravel_multi_index(outputs.T,
                                   dp.outputs.shape)
    return np.prod(map(kind.Value, out_potentials.ravel()[indices]))
