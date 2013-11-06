from cython.operator cimport dereference as deref
# from wrap cimport *
#from constraints cimport *
#from hypergraph cimport *
#from libcpp.vector cimport vector
# from hypergraph import *

# cdef class Chart:
#     r"""
#     A dynamic programming chart :math:`\pi \in R^{|{\cal V}|}`.

#     Act as a dictionary ::

#        >> print chart[node]

#     """

#     cdef vector[double] chart

#     def __getitem__(self, Node node):
#         r"""
#         __getitem__(self, node)

#         Get the chart score for a node.

#         Parameters
#         ----------
#         node : :py:class:`Node`
#           A node :math:`v \in {\cal V}`.

#         Returns
#         -------
#          : float
#           The score of the node, :math:`\pi[v]`.
#         """
#         return self.chart[node.id]

# cdef class MaxMarginals:
#     r"""
#     The max-marginal scores of a weighted hypergraph.

#     .. math::

#         m(e) =  max_{y \in {\cal X}: y(e) = 1} \theta^{\top} y \\
#         m(v) =  max_{y \in {\cal X}: y(v) = 1} \theta^{\top} y


#     Usage is
#         >> max_marginals = compute_max_marginals(graph, weights)
#         >> m_e = max_marginals[edge]
#         >> m_v = max_marginals[node]

#     """

#     cdef const CMaxMarginals *thisptr

#     cdef init(self, const CMaxMarginals *ptr):
#         self.thisptr = ptr
#         return self

#     def __getitem__(self, obj):
#         """
#         Get the max-marginal value of a node or an edge.

#         :param obj: The node/edge to check..
#         :type obj: A :py:class:`Node` or :py:class:`Edge`

#         :returns: The max-marginal value.
#         :rtype: float

#         """
#         if isinstance(obj, Edge):
#             return self.thisptr.max_marginal((<Edge>obj).edgeptr)
#         elif isinstance(obj, Node):
#             return self.thisptr.max_marginal((<Node>obj).nodeptr)
#         else:
#             raise HypergraphAccessException(
#                 "Only nodes and edges have max-marginal values." + \
#                 "Passed %s."%obj)


# def best_path(Hypergraph graph, Weights weights):
#     r"""
#     Find the highest-scoring path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`
#     in the hypergraph.

#     Parameters
#     ----------

#     graph : :py:class:`Hypergraph`
#       The hypergraph :math:`({\cal V}, {\cal E})` to search.

#     weights : :py:class:`Weights`
#       The weights :math:`\theta` of the hypergraph.

#     Returns
#     -------
#     path : :py:class:`Path`
#       The best path :math:`\arg \max_{y \in {\cal X}} \theta^{\top} x`.
#     """
#     cdef vector[double] chart
#     cdef const CHyperpath *hpath = \
#         viterbi_path(graph.thisptr,
#                      deref(weights.thisptr),
#                      &chart)
#     return Path().init(hpath)


# def inside_values(Hypergraph graph, Weights weights):
#     r"""
#     Find the inside path chart values.

#     Parameters
#     ----------

#     graph : :py:class:`Hypergraph`
#       The hypergraph :math:`({\cal V}, {\cal E})` to search.

#     weights : :py:class:`Weights`
#       The weights :math:`\theta` of the hypergraph.

#     Returns
#     -------

#     : :py:class:`Chart`
#        The inside chart.
#     """
#     cdef Chart chart = Chart()
#     cdef const CHyperpath *hpath = \
#         viterbi_path(graph.thisptr,
#                      deref(weights.thisptr),
#                      &chart.chart)
#     return chart

# def outside_values(Hypergraph graph,
#                    Weights weights,
#                    Chart inside_chart):
#     """
#     Find the outside scores for the hypergraph.

#     Parameters
#     -----------

#     graph : :py:class:`Hypergraph`
#        The hypergraph to search.

#     weights : :py:class:`Weights`
#        The weights of the hypergraph.

#     inside_chart : :py:class:`Chart`
#        The inside chart.

#     Returns
#     ---------

#     : :py:class:`Chart`
#        The outside chart.

#     """
#     cdef Chart out_chart = Chart()
#     outside(graph.thisptr, deref(weights.thisptr),
#             inside_chart.chart, &out_chart.chart)
#     return out_chart

# def best_constrained(Hypergraph graph,
#                      Weights weights,
#                      Constraints constraints):
#     """
#     Find the highest-scoring path satisfying constraints.


#     Parameters
#     -----------

#     graph : :py:class:`Hypergraph`
#        The hypergraph to search.

#     weights : :py:class:`Weights`
#        The weights of the hypergraph.

#     constraints : :py:class:`Constraints`
#         The hyperedge constraints.

#     Returns
#     ---------

#     The best path and the dual values.
#     """
#     cdef vector[CConstrainedResult] results
#     cdef const CHyperpath *cpath = \
#         best_constrained_path(graph.thisptr,
#                               deref(weights.thisptr),
#                               deref(constraints.thisptr),
#                               &results)

#     return Path().init(cpath), convert_results(results)

# def compute_max_marginals(Hypergraph graph,
#                           Weights weights):
#     """
#     Compute the max-marginal value for each vertex and edge in the
#     hypergraph.

#     Parameters
#     -----------

#     graph : :py:class:`Hypergraph`
#        The hypergraph to search.

#     weights : :py:class:`Weights`
#        The weights of the hypergraph.

#     Returns
#     --------

#     :py:class:`MaxMarginals`
#        The max-marginals.
#     """
#     cdef const CMaxMarginals *marginals = \
#         compute(graph.thisptr, weights.thisptr)
#     return MaxMarginals().init(marginals)

# def prune_hypergraph(Hypergraph graph,
#                      Weights weights,
#                      double ratio):
#     """
#     Prune hyperedges with low max-marginal score from the hypergraph.

#     Parameters
#     -----------

#     graph : :py:class:`Hypergraph`
#        The hypergraph to search.

#     weights : :py:class:`Weights`
#        The weights of the hypergraph.

#     Returns
#     --------

#     The new hypergraphs and weights.
#     """
#     cdef const CHypergraphProjection *projection = \
#         prune(graph.thisptr, deref(weights.thisptr), ratio)
#     cdef Hypergraph new_graph = Hypergraph()

#     # Map nodes.
#     node_labels = [None] * projection.new_graph.nodes().size()
#     cdef vector[const CHypernode*] old_nodes = graph.thisptr.nodes()
#     cdef const CHypernode *node
#     for i in range(old_nodes.size()):
#         node = projection.project(old_nodes[i])
#         if node != NULL and node.id() >= 0:
#             node_labels[node.id()] = graph.node_labels[i]

#     # Map edges.
#     edge_labels = [None] * projection.new_graph.edges().size()
#     cdef vector[const CHyperedge *] old_edges = graph.thisptr.edges()
#     cdef const CHyperedge *edge
#     for i in range(old_edges.size()):
#         edge = projection.project(old_edges[i])
#         if edge != NULL and edge.id() >= 0:
#             edge_labels[edge.id()] = graph.edge_labels[i]

#     new_graph.init(projection.new_graph, node_labels, edge_labels)
#     cdef Weights new_weights = Weights(new_graph)
#     new_weights.init(
#         weights.thisptr.project_weights(deref(projection)))
#     return new_graph, new_weights
