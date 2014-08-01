from pydecode.potentials import *

def inside(graph, weights, kind=LogViterbi, chart=None):
    r"""
    Compute the inside values for weights.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    weights : |V|-column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    kind : Semiring.
      The semiring to use. Must agree with weights.

    chart : |V|-column vector (optional)
      A chart buffer to reuse.

    Returns
    -------

    chart : |V|x1 column vector (type depends on `kind`).
       The inside chart. Type depends on semiring, i.e.
       for inside this will be the probability paths
       reaching this vertex.
    """

    return kind.inside(graph, weights, chart)


def outside(graph, weights, inside_chart, kind=LogViterbi, chart=None):
    r"""
    Compute the outside values for weights.

    Parameters
    -----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    weights : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    inside_chart : :py:class:`Chart`
       The associated inside chart. Compute by calling
       :py:function:`inside`.  Must be the same type as weights.

    kind : A semiring type.
      The semiring to use. Must agree with weights.

    chart : Mx1 column vector.
      A chart buffer to reuse.

    Returns
    ---------

    chart : Mx1 column vector (type depends on `kind`).
       The outside chart. Type depends on weights type, i.e. for
       inside weights this will be the probability paths reaching
       this node.

    """

    return kind.outside(graph, weights, inside_chart, chart)


def viterbi(graph, weights,
            kind=LogViterbi, chart=None, back_pointers=None, mask=None):
    kind.viterbi(graph, weights, chart,
                 back_pointers, mask, get_path=False)

def best_path(graph, weights,
              kind=LogViterbi, chart=None, back_pointers=None, mask=None):
    r"""
    Find the best path through a hypergraph for a given set of weights.

    Formally gives
    :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
    in the hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The underlying hypergraph :math:`({\cal V}, {\cal E})`.

    weights : Nx1 column vector (type depends on `kind`)
      The potential vector :math:`\theta` for each hyperedge.

    kind : A semiring type.
      The semiring to use. Must agree with weights.

    chart : Mx1 column vector.
      A chart buffer to reuse.

    Returns
    -------
    path : :py:class:`Path`
      The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
    """

    return kind.viterbi(graph, weights,
                        chart=chart,
                        back_pointers=back_pointers,
                        mask=mask)

def marginals(graph, weights,
              inside_chart=None,
              outside_chart=None,
              kind=LogViterbi):
    r"""
    Compute marginals for hypergraph and weights.

    Parameters
    -----------
    graph : :py:class:`Hypergraph`
       The hypergraph to search.

    weights : :py:class:`Weights`
       The weights of the hypergraph.

    Returns
    --------
    marginals : :py:class:`Marginals`
       The node and edge marginals associated with these weights.
    """
    my_inside = inside_chart
    if my_inside is None:
        my_inside = inside(graph, weights, kind=kind)

    my_outside = outside_chart
    if my_outside is None:
        my_outside = \
            outside(graph, weights, inside_chart=my_inside, kind=kind)


    return kind.compute_marginals(graph, weights,
                                  my_inside, my_outside)



# Higher-level interface.

def _check_output_weights(dp, out_weights):
    if dp.outputs.size != out_weights.size:
        raise ValueError("Weights do not match output shape: %s != %s"
                         %(out_weights.shape, dp.outputs.shape))

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
       and outputs is the number of non-zero outputs in the path.
    """
    int_output = dp.output_indices.take(path.edge_indices,
                                        mode="clip")
    return np.array(np.unravel_index(int_output[int_output!=-1],
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
    return np.array(np.unravel_index(dp.item_indices,
                                     dp.items.shape)).T


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
    return np.unravel_index(dp.output_indices,
                            dp.outputs.shape)


def _map_weights(dp, out_weights):
    return out_weights.take(dp.output_indices, mode='clip')

def map_items(dp, out_items):
    return out_items.take(dp.item_indices, mode='clip')

def _map_weights2(dp, out_weights):
    return dp.output_matrix.T * out_weights.ravel()

def argmax(dp, out_weights,
           kind=LogViterbi, chart=None, mask=None):
    """
    Find the highest scoring output structure in a
    dynamic program.

    Returns
    -------
    outputs : output x output_width matrix
       Matrix of outputs.
    """
    _check_output_weights(dp, out_weights)
    weights = _map_weights(dp, out_weights)
    if mask != None:
        new_mask = map_items(dp, mask)
        path = best_path(dp.hypergraph, weights,
                         kind, chart=chart, mask=new_mask)
    else:
        path = best_path(dp.hypergraph, weights,
                         kind, chart=chart)
    return path_output(dp, path)


def fill(dp, out_weights, kind=LogViterbi, chart=None):
    """
    Fill in a dynamic programming chart based on a set of weights.

    Parameters
    ----------
    dp : DynamicProgram

    out_weights : array
       An array in the shape of dp.outputs

    Returns
    -------
    chart : array
       An array in the shape of items.
    """
    _check_output_weights(dp, out_weights)
    weights = _map_weights(dp, out_weights)
    new_chart = inside(dp.hypergraph, weights,
                       kind, chart)
    return new_chart.reshape(dp.items.shape)


def output_marginals(dp,
                     out_weights,
                     kind=LogViterbi):
    """
    Compute marginals for the outputs of a dynamic program.

    Parameters
    ----------
    dp : DynamicProgram

    out_weights : array

    Returns
    -------
    output_marginals : array
       An array in the shape of dp.outputs with marginal values.
    """
    _check_output_weights(dp, out_weights)
    weights = _map_weights(dp, out_weights)
    _, edge_marginals = marginals(dp.hypergraph,
                                  weights, None, None, kind)
    return (dp.output_matrix * edge_marginals).reshape(
        dp.outputs.shape)

def item_marginals(dp,
                   out_weights,
                   kind=LogViterbi):
    """
    Compute marginals for the outputs of a dynamic program.

    Parameters
    ----------
    dp : DynamicProgram

    out_weights : array

    Returns
    -------
    item_marginals : array
       An array in the shape of dp.items with marginal values.
    """
    _check_output_weights(dp, out_weights)
    weights = _map_weights(dp, out_weights)
    node_marginals, _ = marginals(dp.hypergraph,
                                  weights, None, None, kind)
    return (dp.item_matrix * node_marginals).reshape(
        dp.items.shape)

def score_outputs(dp, outputs, out_weights,
                  kind=LogViterbi):
    """

    """
    indices = np.ravel_multi_index(outputs.T,
                                   dp.outputs.shape)
    return np.prod(map(kind.Value,
                       out_weights.ravel()[indices]))
