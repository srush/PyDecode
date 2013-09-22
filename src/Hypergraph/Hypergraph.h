// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_HYPERGRAPH_H_
#define HYPERGRAPH_HYPERGRAPH_H_

#include <set>
#include <string>
#include <vector>

#include "Hypergraph/Weights.h"
#include "./common.h"
using namespace std;

class Hypernode;
class Hyperedge;
typedef const Hypernode *HNode;
typedef vector <const Hypernode *> HNodes;
typedef const Hyperedge *HEdge;
typedef vector<const Hyperedge *> HEdges;

// Base class for weighted hyperedge.
class Hyperedge {
 public:
  Hyperedge(int id, string label, HNode head,
            const vector<HNode> &tails)
    : id_(id), label_(label),
      head_(head), tail_nodes_(tails) {}

  // Get the id of the edge.
  unsigned int id() const { return id_; }

  // Get the label of the edge.
  string label() const { return label_; }

  // Get the head node of the edge.
  HNode head_node() const { return head_; }

  // Get the tail nodes of the hyperedge in order.
  const vector<HNode> &tail_nodes() const { return tail_nodes_; }

 private:
  int id_;
  string label_;
  vector<HNode> tail_nodes_;
  HNode head_;
};


/**
 * Hypernode - Constant representation of a hypernode in a hypergraph.
 * Accessors for edges above and below.
 */
class Hypernode {
  public:
  explicit Hypernode(int id) : id_(id) {}

  unsigned int id() const { return id_; }

  void add_edge(const Hyperedge *edge) {
    edges_.push_back(edge);
  }

  // Get all hyperedges with this hypernode as head.
  const vector<HEdge> &edges() const { return edges_; }

  bool terminal() const { return (edges_.size() == 0); }

  /**
   * Get all hyperedges with this hypernode as a tail.
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to edges.
   */
  //virtual const vector<Hyperedge *> &in_edges() const = 0;
 private:
  int id_;
  vector<HEdge> edges_;
};

class Hypergraph {
 public:
  /**
   * Get the root of the hypergraph
   *
   * @return Hypernode at root
   */
  HNode root() const { return nodes_[root_id_]; }

  // Switching to iterator interface
  /**
   * Get all hypernodes in the hypergraph. (Assume unordered)
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to hypernodes in hypergraph.
   */
  const vector <HNode> &nodes() const {
    return nodes_;
  }

  /**
   * Get all hyperedges in the hypergraph. (Assume unordered)
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to edges in hypergraph .
   */
  const vector <HEdge> & edges() const {
    return edges_;
  }

  // Construction Code.

  // Create a new node and begin adding edges.
  HNode start_node();

  HEdge add_edge(const vector<HNode> &nodes, string label);


  void end_node() { lock_ = false; }

  // Add a hyperedge to the current hypernode in focus.
//HEdge add_edge(const vector<HNode> &nodes, string label);

  // Complete the hypergraph.
  void finish() {
    root_id_ = nodes_.size() - 1;
    // TODO(srush) Run checks to make sure we are complete.
  }

 private:
  // For construction.

  // The hypergraph is adding an edge. It is locked.
  bool lock_;

  // The current node being created.
  Hypernode *creating_node_;

  // List of nodes guarenteed to be in topological order.
  vector<HNode> nodes_;

  // List of edges guarenteed to be in topological order.
  vector<HEdge> edges_;

  // The id of the root.
  int root_id_;
};

class Hyperpath {
 public:
  Hyperpath(const Hypergraph *graph,
            const vector<HEdge> &edges)
      : edges_(edges) {
    foreach (HEdge edge, edges) {
      edges_set_.insert(edge->id());
    }
  }

  const vector<HEdge> &edges() const {
    return edges_;
  }

  bool has_edge(HEdge edge) const {
    return edges_set_.find(edge->id())
        != edges_set_.end();
  }

 private:
  set<int> edges_set_;
  const vector<HEdge> edges_;
};

class HypergraphWeights {
 public:
  HypergraphWeights(const Hypergraph *hypergraph,
                    const vector<double> &weights,
                    double bias)
  : weights_(weights),
      hypergraph_(hypergraph),
      bias_(bias)

  {
    assert(weights.size() == hypergraph->edges().size());
  }

  double dot(const Hyperpath &path) const;

  double score(HEdge edge) const { return weights_[edge->id()]; }

  double bias() const { return bias_; }

  HypergraphWeights *modify(const SparseVec &, double) const;



 private:
  const Hypergraph *hypergraph_;
  vector<double> weights_;
  double bias_;
};


#endif  // HYPERGRAPH_HYPERGRAPH_H_
