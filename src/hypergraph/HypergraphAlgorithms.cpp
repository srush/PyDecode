#include "HypergraphAlgorithms.h"

#include <iostream>
#include <iomanip>
#include <set>
#include <queue>
#include <vector>
#include <cmath>
#include <cy_svector.hpp>
#include <algorithm>
#include "../common.h"
using namespace std;

namespace Scarab{
  namespace HG{
    double log_sum( double a, double b) {
      double M = max(a,b);
      double m = min(a,b);
      return M + log(1.0 + exp(m - M));
    }

double best_path_helper(const Hypernode & node, const EdgeCache & edge_weights, NodeCache & score_memo_table, NodeBackCache & back_memo_table);
vector <const Hypernode *> construct_best_fringe_help(const Hypernode & node, const NodeBackCache & back_memo_table);
HEdges construct_best_edges_help(const Hypernode & node, const NodeBackCache & back_memo_table);
wvector construct_best_fv_help(const Hypernode & node, const NodeBackCache & back_memo_table);

    void best_outside_path_helper(const Hypernode & node,
                              const EdgeCache & edge_weights,
                              const NodeCache & score_memo_table,
                              NodeCache & outside_memo_table);


    void  HypergraphAlgorithms::reachable(set<int> *reachable_nodes, set<int> *reachable_edges) const {
      set<int> s;
      s.insert(_forest.root().id());
      while (!s.empty()) {
        int n = (*s.begin());
        s.erase(n);
        if (reachable_nodes->find(n) != reachable_nodes->end()) {
          continue;
        }
        reachable_nodes->insert(n);
        const Hypernode & node = _forest.get_node(n);
        foreach (const Hyperedge *edge, node.edges()) {
          reachable_edges->insert(edge->id());
          foreach (const Hypernode * sub_node, edge->tail_nodes()) {
            s.insert(sub_node->id());
          }
        }
      }
    }

vector <const Hypernode *>  HypergraphAlgorithms::topological_sort() const {
  set<int> reachable_edges, reachable_nodes;
  reachable(&reachable_nodes, &reachable_edges);
  //cerr << reachable_nodes.size() << endl;
  vector <const Hypernode * > top_sort;
  queue<int> s;
  s.push(_forest.root().id());
  top_sort.clear();
  Cache <Hyperedge, bool> removed(_forest.num_edges());
  while (!s.empty()) {
    int n = s.front();
    s.pop();
    top_sort.push_back(&_forest.get_node(n));

    const Hypernode & node = _forest.get_node(n);
    foreach (const Hyperedge *edge, node.edges()) {
      removed.set_value(*edge, 1);
      assert(reachable_edges.find(edge->id()) != reachable_edges.end());
      foreach (const Hypernode *sub_node, edge->tail_nodes()) {
        bool no_edges = true;
        // Have all the above edges been removed.
        foreach (const Hyperedge *in_edge, sub_node->in_edges()) {
          // Is reachable.
          if (reachable_edges.find(in_edge->id()) != reachable_edges.end() ) {
            no_edges &= (removed.has_key(*in_edge) && removed.get_value(*in_edge));
          }
        }
        if (no_edges) {
          s.push(sub_node->id());
        }
      }
    }
  }
  return top_sort;
}


EdgeCache * HypergraphAlgorithms::cache_edge_weights(const svector<int, double> & weight_vector ) const {
  EdgeCache * weights = new EdgeCache(_forest.num_edges());

  foreach (const Hyperedge *edge, _forest.edges()) {
    double dot = edge->fvector().dot(weight_vector);
    //cout << svector_str(weight_vector) << endl;
    //cout << svector_str(edge.fvector()) << endl;
    //cout << dot<< endl;
    //cout << "Norm " << edge.fvector().normsquared() << " " << weight_vector.normsquared() << endl;
    //cout << "Dot " << edge.id() << " " << dot<< endl;
    weights->set_value(*edge, dot);
  }
  return weights;
}


EdgeCache* HypergraphAlgorithms::combine_edge_weights( const EdgeCache & w1, const EdgeCache & w2 )  const {
  EdgeCache * combine = new EdgeCache(_forest.num_edges());
  foreach (const Hyperedge * edge, _forest.edges()) {

    double v1;
    if (w1.has_key(*edge)) {
      v1= w1.get_value(*edge);
    } else {
      v1 =0.0;
    }

    double v2;
    if (w2.has_key(*edge)) {
      v2= w2.get_value(*edge);
    } else {
      v2 = 0.0;
    }
    combine->set_value(*edge, v1 + v2);
  }
  return combine;
}



vector <const Hypernode *> HypergraphAlgorithms::construct_best_fringe( const NodeBackCache & back_memo_table) const {
  return construct_best_fringe_help(_forest.root(), back_memo_table);
}

vector <const Hypernode *> construct_best_fringe_help(const Hypernode & node, const NodeBackCache & back_memo_table) {
  vector <const Hypernode *> best;
  if (node.is_terminal()) {
    best.push_back(&node);
    return best;
  }

  const Hyperedge * edge = back_memo_table.get_value(node);

  foreach(const Hypernode * bnode, edge->tail_nodes()) {
    vector <const Hypernode *> b = construct_best_fringe_help(*bnode, back_memo_table);
    best.insert(best.end(), b.begin(), b.end());

  }
  return best;
}

wvector HypergraphAlgorithms::construct_best_feature_vector(const NodeBackCache & back_memo_table) const {
  return construct_best_fv_help(_forest.root(), back_memo_table);
}

wvector construct_best_fv_help(const Hypernode & node, const NodeBackCache & back_memo_table) {
  wvector best;
  if (node.num_edges() == 0) {
    //assert (node.is_word());
    return best;
  } else {
    const Hyperedge * edge = back_memo_table.get_value(node);
    best += edge->fvector();

    foreach (const Hypernode * bnode, edge->tail_nodes()) {
      best += construct_best_fv_help(*bnode, back_memo_table);
    }
  }
  return best;
}

HEdges HypergraphAlgorithms::construct_best_edges(const NodeBackCache & back_memo_table) const {
  return construct_best_edges_help(_forest.root(), back_memo_table);
}

HEdges construct_best_edges_help(const Hypernode & node, const NodeBackCache & back_memo_table) {
  HEdges best;

  if (node.num_edges() == 0) {
    //assert (node.is_word());
    return best;
  } else {
    const Hyperedge * edge = back_memo_table.get_value(node);
    best.push_back(edge);

    foreach (const Hypernode * bnode, edge->tail_nodes()) {
      vector <const Hyperedge *> b = construct_best_edges_help(*bnode, back_memo_table);
      foreach (const Hyperedge *  in_b, b) {
        best.push_back(in_b);
      }
    }
  }
  return best;
}


vector <const Hypernode *> construct_best_node_order_help(const Hypernode & node, const NodeBackCache & back_memo_table) {
  vector <const Hypernode * > best;

  best.push_back(&node);

  if (node.num_edges() == 0) {
    //assert (node.is_word());
    //cout << "w ";
  } else {
    //cout << node.id() << "D ";
    const Hyperedge * edge = back_memo_table.get_value(node);
    //for (int i =0; i < edge->num_nodes(); i++)  {
    foreach (const Hypernode * bnode, edge->tail_nodes() ) {
      //const Hypernode & bnode = edge->tail_node(i);
      vector <const Hypernode *> b = construct_best_node_order_help(*bnode, back_memo_table);
      foreach (const Hypernode * in_b, b) {
        best.push_back(in_b);
      }
    }
    //cout << node.id() << "U ";
  }

  return best;
}

vector <const Hypernode * > HypergraphAlgorithms::construct_best_node_order(const NodeBackCache & back_memo_table) const {
  return construct_best_node_order_help(_forest.root(), back_memo_table);
}

    double HypergraphAlgorithms::filter_pruning_threshold(const EdgeCache & edge_weights,
                                                          const NodeCache & score_memo_table,
                                                          const NodeCache & outside_memo_table,
                                                          double best,
                                                          double alpha) {

      set<int> reachable_edges, reachable_nodes;
      reachable(&reachable_nodes, &reachable_edges);
      int count = 0;
      double total = 0.0;
      foreach (HNode node, _forest.nodes()) {
        if (reachable_nodes.find(node->id()) == reachable_nodes.end()) continue;
        double node_outside = outside_memo_table.get(*node);
        double node_inside = score_memo_table.get(*node);
        double marginal = node_inside + node_outside;
        total += marginal;
        count++;
      }
      return alpha * best + (1.0 - alpha) * (total / count) + 1e-4;
    }

HypergraphPrune HypergraphAlgorithms::pretty_good_pruning(const EdgeCache & edge_weights,
                                                          const NodeCache & score_memo_table,
                                                          const NodeCache & outside_memo_table,
                                                          double cutoff) {
  HypergraphPrune prune(_forest);
//   vector<const Hypernode *> node_order =  topological_sort();
//   reverse(node_order.begin(), node_order.end());
  set<int> reachable_edges, reachable_nodes;
  reachable(&reachable_nodes, &reachable_edges);
  //vector <const Hypernode *> node_order =  topological_sort();
  const vector<Hypernode *> &nodes = _forest.nodes();

  foreach (HNode node, _forest.nodes()) {
    if (!outside_memo_table.has_key(*node)) {
      prune.nodes.insert(node->id());
      continue;
    }
  }
//   for (int i = nodes.size() - 1; i >= 0; --i) {
//     HNode node = nodes[i];

  //foreach (HNode node, node_order) {
  foreach (HNode node, _forest.nodes()) {
    if (reachable_nodes.find(node->id()) == reachable_nodes.end()) continue;
    double node_outside = outside_memo_table.get(*node);
    double node_inside = score_memo_table.get(*node);
    double marginal = node_inside + node_outside;
    bool node_pruned = marginal > cutoff;
    bool all_pruned = true;
    foreach (HEdge edge, node->edges()) {
      double total = 0.0;
      foreach (HNode sub_node, edge->tail_nodes()) {
        total += score_memo_table.get(*sub_node);
        //cerr << sub_node->label() << " ";
      }

      double edge_weight = edge_weights.get_value(*edge);
      double edge_marginal = total + node_outside + edge_weights.get_value(*edge);
      if (node_pruned || edge_marginal > cutoff) {
        prune.edges.insert(edge->id());
        //cerr << 0 << " " << node->id() << " " << edge->id() << " " << edge_marginal << " " << edge_weight << endl;
      } else {
        //cerr << 1 << " " << node->id() << " " << edge->id() << " " << edge_marginal<< " " << edge_weight << endl;
        all_pruned = false;
      }
    }

    if (node_pruned  || (all_pruned && node->num_edges() > 0) ) {
      prune.nodes.insert(node->id());
      //cerr << 0 << " " << node->id() << " " << node_inside << " " << node_outside << " " << marginal << " " << cutoff << " "
      //   << node->label() << endl;

    } else {
      //cerr << 1 << " " << node->id() << " " << node_inside << " " << node_outside << " " << marginal << " " << cutoff << " "
      //<< node->label() << endl;
    }
  }
  return prune;
}



double HypergraphAlgorithms::best_outside_path(const EdgeCache & edge_weights,
                                               const NodeCache & score_memo_table,
                                               NodeCache & outside_score_table) const {
  vector<const Hypernode *> node_order =
    HypergraphAlgorithms(_forest).topological_sort();

  foreach(HNode node, node_order) {
    outside_score_table.set_value(*node, INF);
  }

  foreach (HNode node, node_order) {
    int id = node->id();
    //if (_out_done.find(id) == _out_done.end()) {
    best_outside_path_helper(*node, edge_weights, score_memo_table, outside_score_table);
      //}
  }
}

void best_outside_path_helper(const Hypernode & node,
                              const EdgeCache & edge_weights,
                              const NodeCache & score_memo_table,
                              NodeCache & outside_memo_table) {
  // when you get to a node it is done already
  assert (outside_memo_table.has_key(node));
  //assert(_out_done.find(node.id()) == _out_done.end());
  double above_score = outside_memo_table.get_value(node);

  foreach (HEdge edge, node.edges()) {
    double edge_value= edge_weights.get_value(*edge);
    double total = 0.0;
    foreach (HNode node, edge->tail_nodes()) {
      double node_inside = score_memo_table.get_value(*node);
      total += node_inside;
    }

    foreach (HNode node, edge->tail_nodes()) {
      double node_inside = score_memo_table.get_value(*node);
      double outside_score = edge_value + above_score + total - node_inside;
      double best_score = outside_memo_table.get(*node);
      if (outside_score < best_score) {
        //best_score = outside_score;
        //best_edge = edge;
        outside_memo_table.set_value(*node, outside_score);
      }
    }
  }
}

    double HypergraphAlgorithms::inside_scores(bool max, const EdgeCache & edge_weights,
                                               NodeCache & inside_memo_table) const {
      return inside_score_helper(max, _forest.root(), edge_weights, inside_memo_table);
    }

    double HypergraphAlgorithms::inside_score_helper(bool use_max, const Hypernode & node,
                                                     const EdgeCache & edge_weights,
                                                     NodeCache & inside_memo_table) const {
      // assume score are log probs
      if (inside_memo_table.has_key(node)) {
        return inside_memo_table.get_value(node);
      }

      double inside_score;

      if (node.num_edges() == 0) {
        inside_score = 0.0; // log(1.0)
      } else {
        inside_score = INF;
        foreach (const Hyperedge * edge, node.edges()) {
          double edge_value = edge_weights.get_value(*edge);
      foreach ( const Hypernode * tail_node, edge->tail_nodes()) {
        // sum
        edge_value += inside_score_helper(use_max, *tail_node, edge_weights, inside_memo_table);
      }
      // log sum
      if (use_max) {
        inside_score = min(inside_score, edge_value);
      } else {
        inside_score = log_sum(inside_score, edge_value);
      }
    }
  }
  //cerr << "inside " << node.id() << " " << node.label() << " " << inside_score << endl;
  inside_memo_table.set_value(node, inside_score);
  return inside_score;
}

    double HypergraphAlgorithms::outside_scores(bool max, const EdgeCache & edge_weights,
                                                const NodeCache & inside_memo_table,
                                                NodeCache & outside_memo_table) const {


      //reverse(node_order.begin(), node_order.end());
      //cerr << node_order.size() << " " << _forest.num_nodes() << endl;
      //assert(node_order.size() == _forest.num_nodes());
      outside_memo_table.set_value(_forest.root(), 0.0);
      //cerr << _forest.root().id() << endl;
      set<int> reachable_edges, reachable_nodes;
      reachable(&reachable_nodes, &reachable_edges);
      //vector <const Hypernode *> node_order =  topological_sort();
      const vector<Hypernode *> &nodes = _forest.nodes();
      //foreach (HNode node, _forest.nodes()) {
      for (int i = nodes.size() - 1; i >= 0; --i) {
        HNode node = nodes[i];
        int id = node->id();
        if (reachable_nodes.find(id) != reachable_nodes.end()) {
          outside_score_helper(max, *node, edge_weights, inside_memo_table, outside_memo_table);
        }
      }
    }

    double HypergraphAlgorithms::outside_score_helper(bool use_max,
                                                      const Hypernode & node,
                                                      const EdgeCache & edge_weights,
                                                      const NodeCache & inside_memo_table,
                                                      NodeCache & outside_memo_table) const {
      if (!outside_memo_table.has_key(node)) return 0.0;
      assert(outside_memo_table.has_key(node));
      double above_score = outside_memo_table.get_value(node);

      foreach (HEdge edge, node.edges()) {
        double edge_value= edge_weights.get_value(*edge);
        double total = 0.0;
        foreach (HNode node, edge->tail_nodes()) {
          double node_inside = inside_memo_table.get_value(*node);
          total += node_inside;
        }

        foreach (HNode node, edge->tail_nodes()) {
          double node_inside = inside_memo_table.get_value(*node);
          double outside_score = edge_value + above_score + total - node_inside;
          if (outside_memo_table.has_key(*node)) {
            double cur_score = outside_memo_table.get(*node);
            if (!use_max) {
              outside_memo_table.set_value(*node, log_sum(cur_score, outside_score));
            } else {
              outside_memo_table.set_value(*node, min(cur_score, outside_score));
            }
          } else {
            outside_memo_table.set_value(*node, outside_score);
          }
        }
      }
    }


double HypergraphAlgorithms::best_path( const EdgeCache & edge_weights, NodeCache & score_memo_table,
                                        NodeBackCache & back_memo_table) const {
  return  best_path_helper(_forest.root(), edge_weights, score_memo_table, back_memo_table);
}

// find the best path through a hypergraph
double best_path_helper(const Hypernode & node, const EdgeCache & edge_weights,
                        NodeCache & score_memo_table, NodeBackCache & back_memo_table) {
  double best_score = INF;
  //int id = node.id();

  const Hyperedge * best_edge = NULL;
  if (score_memo_table.has_key(node)) {
    return score_memo_table.get_value(node);
  }

  //cout << "EDGES: "<< node.num_edges() <<endl;
  if (node.num_edges() == 0) {
    //assert (node.is_word());
    best_score = score_memo_table.get_default(node, 0.0);
    // best_score = 0.0;
    best_edge = NULL;
  } else {
    foreach (const Hyperedge * edge, node.edges()) {
      double edge_value= edge_weights.get_value(*edge);
      foreach ( const Hypernode * tail_node, edge->tail_nodes()) {
        edge_value += best_path_helper(*tail_node, edge_weights, score_memo_table, back_memo_table);
      }
      //cout << edge_value << endl;
      if (edge_value < best_score) {
        best_score = edge_value;
        best_edge = edge;
      }
    }
  }

  assert (best_score != INF);

  score_memo_table.set_value(node, best_score);
  back_memo_table.set_value(node, best_edge);
  return best_score;
}

void HypergraphAlgorithms::collect_marginals(const NodeCache & inside_memo_table,
                                             const NodeCache & outside_memo_table,
                                             NodeCache & marginals ) const {
  double normalize = inside_memo_table.get_value(_forest.root());
  foreach (HNode node, _forest.nodes()) {
    if (inside_memo_table.has_key(*node) ) {
      double inside = inside_memo_table.get(*node);
      double outside = outside_memo_table.get(*node);
      double marginal = exp((inside + outside) - normalize);
      //assert (marginal >= 0.0 && marginal <= 1.0);
      marginals.set_value(*node, marginal);
    }
  }
}




  }}



