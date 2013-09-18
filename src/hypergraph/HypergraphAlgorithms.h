#ifndef HYPERGRAPHALGORITHMS_H_
#define HYPERGRAPHALGORITHMS_H_

#include "svector.hpp"
#include "EdgeCache.h"
#include "Hypergraph.h"

namespace Scarab {
  namespace HG {


typedef Cache <Hyperedge, double> EdgeCache;
typedef Cache <Hypernode, double> NodeCache;
typedef Cache <Hypernode, const Hyperedge *> NodeBackCache;

 /* struct HypergraphWeights { */
 /* HypergraphWeights():_edge_weights() */
 /* const EdgeWeights(); */
 /* } */



class HypergraphAlgorithms {
 public:
 HypergraphAlgorithms(const HGraph & hypergraph): _forest(hypergraph) {}

/** Associate a weight which each edge in the hypergraph
 *  @param weight_vector A weight vector
 *  @return A cache associated a weight with each edge
 */
EdgeCache * cache_edge_weights(const svector <int, double> & weight_vector ) const;

/** Combine two weight vectors 
 * fix this!
 */
EdgeCache* combine_edge_weights(const EdgeCache & w1, 
                                const EdgeCache & w2 ) const;

/** Given a hypergraph and back pointers, produces the left-to-right fringe
 *  @param forest The hypergraph 
 *  @param back_memo_table The associated back pointers (possibly obtained through best_path) 
 *  @return A const iterator of hypernodes in "inorder" order
 */
HNodes construct_best_fringe(const NodeBackCache & back_memo_table) const ;


/** Given a hypergraph and back pointers, produces the best edges used in the path
 *  @param back_memo_table The associated back pointers (possibly obtained through best_path) 
 *  @return A vector of const edges
 */
HEdges construct_best_edges(const NodeBackCache & back_memo_table) const;

/** Given a hypergraph and back pointers, produces the best nodes used in the path (in inorder order)
 *  
 *  @param back_memo_table The associated back pointers (possibly obtained through best_path) 
 *  @return A vector of const hypernodes
 */
HNodes construct_best_node_order(const NodeBackCache & back_memo_table) const;


 wvector construct_best_feature_vector(const NodeBackCache & back_memo_table) const;

/** Find the best path, lowest weight, through a weighted hypergraph
 *  @param edge_weights The cached edge weights associated with the graph
 *  @param score_memo_table The shortest path to each node
 *  @param back_memo_table The back pointers.
 *  @return Weight of shortest path
 */
double best_path(const EdgeCache & edge_weights, 
                 NodeCache & score_memo_table, 
                 NodeBackCache & back_memo_table) const;


double best_outside_path(const EdgeCache & edge_weights, 
                         const NodeCache & score_memo_table, 
                         NodeCache & outside_score_table) const;


/** Topologically sort the given hypergraph (immutable) 
 *  @return The ids of the hypergraph in topological order
 */
 HNodes topological_sort() const;
 void reachable(set<int> *reachable_nodes, set<int> *reachable_edges) const;

HypergraphPrune pretty_good_pruning(const EdgeCache & edge_weights,
                                    const NodeCache & score_memo_table, 
                                    const NodeCache & outside_memo_table,
                                    double cutoff);

  // Marginals
 double inside_scores(bool max, const EdgeCache & edge_weights,  
                      NodeCache & inside_memo_table) const;

 double outside_scores(bool max, const EdgeCache &edge_weights,  
                       const NodeCache &inside_memo_table, 
                       NodeCache &outside_memo_table) const;

 
 void collect_marginals(const NodeCache & inside_memo_table, 
                        const NodeCache & outside_memo_table,
                        NodeCache & marginals) const;

 double filter_pruning_threshold(const EdgeCache & edge_weights,
                                 const NodeCache & score_memo_table, 
                                 const NodeCache & outside_memo_table,
                                 double best, 
                                 double alpha);


 private:
 const HGraph & _forest;

 double outside_score_helper(bool use_max, const Hypernode & node, 
                             const EdgeCache & edge_weights, 
                             const NodeCache & inside_memo_table, 
                             NodeCache & outside_memo_table) const;

 double inside_score_helper(bool use_max, const Hypernode & node, 
                            const EdgeCache & edge_weights, 
                            NodeCache & inside_memo_table) const;


};

  }}
#endif 
