// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <vector>

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Constraints.h"
#include "./common.h"


Hyperpath *viterbi_path(const Hypergraph *graph,
                        const HypergraphWeights &theta,
                        vector<double> *chart);

void outside(const Hypergraph *graph,
             const HypergraphWeights &weights,
             const vector<double> &inside_chart,
             vector<double> *chart);


Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights &theta,
    const HypergraphConstraints &constraints,
    vector<double> *duals);



/*   // Run Viterbi Hypergraph algorithm. */
/*   vector<double> chart(graph->nodes().size(), -INF); */
/*   vector<HEdge> back(graph->nodes().size(), NULL); */
/*   foreach (HEdge edge, graph->edges()) { */
/*     double score = theta.score(edge); */
/*     int head_id = edge->head()->id(); */
/*     foreach (HNode node, edge->tail_nodes()) { */
/*       score += chart[node->id()]; */
/*     } */
/*     if (score > chart[head_id]) { */
/*       chart[head_id] = score; */
/*       back[head_id] = edge; */
/*     } */
/*   } */

/*   // Collect backpointers. */
/*   vector<HEdge> path; */
/*   queue<HNode> to_examine; */
/*   to_examine.push_back(graph->root()); */
/*   while(!to_examine.empty()) { */
/*     HEdge edge = back[to_examine.front()->id()]; */
/*     to_examine.pop() */
/*     foreach (HNode node, edge->tail_nodes()) { */
/*       to_examine.push(node); */
/*     } */
/*   } */
/*   return new Hypergraph(graph, path); */
/* } */



/* typedef Cache<Hyperedge, double> EdgeCache; */
/* typedef Cache<Hypernode, double> NodeCache; */
/* typedef Cache<Hypernode, const Hyperedge *> NodeBackCache; */

/* /\* struct HypergraphWeights { *\/ */
/* /\* HypergraphWeights():_edge_weights() *\/ */
/* /\* const EdgeWeights(); *\/ */
/* /\* } *\/ */


/* class HypergraphAlgorithms { */
/*  public: */
/*   HypergraphAlgorithms(const HGraph & hypergraph) */
/*     : _forest(hypergraph) {} */

/*   /\** Associate a weight which each edge in the hypergraph */
/*    *  @param weight_vector A weight vector */
/*    *  @return A cache associated a weight with each edge */
/*    *\/ */
/*   EdgeCache * cache_edge_weights( */
/*       const svector <int, double> & weight_vector) const; */

/*   /\** Combine two weight vectors */
/*    * fix this! */
/*    *\/ */
/*   EdgeCache* combine_edge_weights(const EdgeCache & w1, */
/*                                   const EdgeCache & w2) const; */

/*   /\** Given a hypergraph and back pointers, produces the left-to-right fringe */
/*    *  @param forest The hypergraph */
/*    *  @param back_memo_table The associated back pointers (possibly obtained through best_path) */
/*    *  @return A const iterator of hypernodes in "inorder" order */
/*    *\/ */
/*   HNodes construct_best_fringe(const NodeBackCache & back_memo_table) const; */


/*   /\** Given a hypergraph and back pointers, produces the best edges used in the path */
/*    *  @param back_memo_table The associated back pointers (possibly obtained through best_path) */
/*    *  @return A vector of const edges */
/*    *\/ */
/*   HEdges construct_best_edges(const NodeBackCache & back_memo_table) const; */

/*   /\** Given a hypergraph and back pointers, produces the best nodes used in the path (in inorder order) */
/*    * */
/*    *  @param back_memo_table The associated back pointers (possibly obtained through best_path) */
/*    *  @return A vector of const hypernodes */
/*    *\/ */
/*   HNodes construct_best_node_order( */
/*       const NodeBackCache & back_memo_table) const; */


/*   wvector construct_best_feature_vector( */
/*       const NodeBackCache & back_memo_table) const; */

/*   /\** Find the best path, lowest weight, through a weighted hypergraph */
/*    *  @param edge_weights The cached edge weights associated with the graph */
/*    *  @param score_memo_table The shortest path to each node */
/*    *  @param back_memo_table The back pointers. */
/*    *  @return Weight of shortest path */
/*    *\/ */
/*   double best_path(const EdgeCache & edge_weights, */
/*                    NodeCache & score_memo_table, */
/*                    NodeBackCache & back_memo_table) const; */


/*   double best_outside_path(const EdgeCache & edge_weights, */
/*                            const NodeCache & score_memo_table, */
/*                            NodeCache & outside_score_table) const; */


/*   /\** Topologically sort the given hypergraph (immutable) */
/*    *  @return The ids of the hypergraph in topological order */
/*    *\/ */
/*   HNodes topological_sort() const; */
/*   void reachable(set<int> *reachable_nodes, */
/*                  set<int> *reachable_edges) const; */

/*   HypergraphPrune pretty_good_pruning(const EdgeCache & edge_weights, */
/*                                       const NodeCache & score_memo_table, */
/*                                       const NodeCache & outside_memo_table, */
/*                                       double cutoff); */

/*   // Marginals */
/*   double inside_scores(bool max, */
/*                        const EdgeCache & edge_weights, */
/*                        NodeCache & inside_memo_table) const; */

/*   double outside_scores(bool max, */
/*                         const EdgeCache &edge_weights, */
/*                         const NodeCache &inside_memo_table, */
/*                         NodeCache &outside_memo_table) const; */


/*   void collect_marginals(const NodeCache & inside_memo_table, */
/*                          const NodeCache & outside_memo_table, */
/*                          NodeCache &marginals) const; */

/*   double filter_pruning_threshold(const EdgeCache &edge_weights, */
/*                                   const NodeCache &score_memo_table, */
/*                                   const NodeCache &outside_memo_table, */
/*                                   double best, */
/*                                   double alpha); */


/*  private: */
/*   const HGraph & _forest; */

/*   double outside_score_helper(bool use_max, const Hypernode & node, */
/*                               const EdgeCache &edge_weights, */
/*                               const NodeCache &inside_memo_table, */
/*                               NodeCache &outside_memo_table) const; */

/*   double inside_score_helper(bool use_max, */
/*                              const Hypernode & node, */
/*                              const EdgeCache &edge_weights, */
/*                              NodeCache & inside_memo_table) const; */
/* }; */

#endif  // HYPERGRAPH_ALGORITHMS_H_
