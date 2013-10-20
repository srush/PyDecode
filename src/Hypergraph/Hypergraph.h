// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_HYPERGRAPH_H_
#define HYPERGRAPH_HYPERGRAPH_H_

#include <cassert>
#include <exception>
#include <set>
#include <string>
#include <vector>

#include "./common.h"
#include "Hypergraph/Semirings.h"
using namespace std;

class Hypernode;
class Hyperedge;
typedef const Hypernode *HNode;
typedef vector <const Hypernode *> HNodes;
typedef const Hyperedge *HEdge;
typedef vector<const Hyperedge *> HEdges;

struct HypergraphException : public exception {
  string s;
  HypergraphException(string ss) : s(ss) {}
  ~HypergraphException() throw () {}
  const char* what() const throw() { return s.c_str(); }
};

class HypergraphAccessException : public HypergraphException {
};

class HypergraphMatchException : public HypergraphException {

};

class HypergraphConstructionException : public HypergraphException {

};

// Base class for weighted hyperedge.
class Hyperedge {
 public:
  Hyperedge(string label,
            HNode head,
            const vector<HNode> &tails)
    : id_(-1),
      label_(label),
      head_(head),
      tail_nodes_(tails) {}

  // Get the id of the edge.
  int id() const { return id_; }

  void set_id(int id) { id_ = id; }

  // Get the label of the edge.
  string label() const { return label_; }

  // Get the head node of the edge.
  HNode head_node() const { return head_; }

  // Get the tail nodes of the hyperedge in order.
  const vector<HNode> &tail_nodes() const { return tail_nodes_; }

  bool operator<(const Hyperedge *edge2) const {
    return id() < edge2->id();
  }

 private:
  int id_;
  string label_;
  HNode head_;
  vector<HNode> tail_nodes_;
};


/**
 * Hypernode - Constant representation of a hypernode in a hypergraph.
 * Accessors for edges above and below.
 */
class Hypernode {
  public:
  explicit Hypernode(string label)
    : id_(-1), label_(label) {}

  string label() const { return label_; }

  int id() const { return id_; }

  void set_id(int id) { id_ = id; }

  void add_edge(const Hyperedge *edge) {
    edges_.push_back(edge);
  }

  // Get all hyperedges with this hypernode as head.
  const vector<HEdge> &edges() const { return edges_; }

  bool terminal() const { return edges_.size() == 0; }

 private:
  int id_;
  string label_;
  vector<HEdge> edges_;
};

class Hypergraph {
 public:
  Hypergraph()
      : terminal_lock_(true), lock_(false),
      temp_nodes_(0), temp_edges_(0),
      id_(ID++) {}

  /**
   * Get the root of the hypergraph
   *
   * @return Hypernode at root
   */
  HNode root() const { return root_; }

  // Switching to iterator interface
  /**
   * Get all hypernodes in the hypergraph. (Assume unordered)
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to hypernodes in hypergraph.
   */
  const vector<HNode> &nodes() const {
    return nodes_;
  }

  /**
   * Get all hyperedges in the hypergraph. (Assume unordered)
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to edges in hypergraph .
   */
  const vector<HEdge> & edges() const {
    return edges_;
  }

  // Construction Code.

  // Create a new node and begin adding edges.
  HNode add_terminal_node(string label);

  HNode start_node(string label);

  HEdge add_edge(const vector<HNode> &nodes, string label);

  // Returns true if the node was created.
  // Returns false if the node was removed (no children).
  bool end_node();

  // Complete the hypergraph.
  void finish() {
    if (temp_nodes_.size() == 0) {
      throw HypergraphException("Hypergraph has size 0.");
    }
    root_ = temp_nodes_[temp_nodes_.size() - 1];
    fill();
    // TODO(srush) Run checks to make sure we are complete.
  }

  // Remove paths that do not reach the root.
  void fill();


  bool same(const Hypergraph &other) const { return other.id_ == id_; }

 private:

  // For construction.
  bool terminal_lock_;

  // The hypergraph is adding an edge. It is locked.
  bool lock_;

  // The current node being created.
  Hypernode *creating_node_;

  // List of nodes guarenteed to be in topological order.
  vector<Hypernode *> temp_nodes_;

  // List of edges guarenteed to be in topological order.
  vector<Hyperedge *> temp_edges_;

  // The true interface.

  // List of nodes guarenteed to be in topological order.
  vector<HNode> nodes_;


  // List of edges guarenteed to be in topological order.
  vector<HEdge> edges_;

  HNode root_;

  int id_;

  static int ID;
};

class Hyperpath {
 public:
  Hyperpath(const Hypergraph *graph,
            const vector<HEdge> &edges)
      : graph_(graph), edges_(edges) {
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

  void check(const Hypergraph &graph) const {
    if (!graph.same(*graph_)) {
      throw HypergraphException("Hypergraph does not match path.");
    }
  }

 private:
  const Hypergraph *graph_;
  set<int> edges_set_;
  const vector<HEdge> edges_;

};

class HypergraphProjection;

template<typename SemiringType = DoubleWeight>
class HypergraphWeights {
 public:
  HypergraphWeights(const Hypergraph *hypergraph,
                    const vector<SemiringType> &weights,
                    SemiringType bias)
  : hypergraph_(hypergraph),
    weights_(weights),
    bias_(bias) {
      assert(weights.size() == hypergraph->edges().size());
  }
  // DEPRECATED:
  HypergraphWeights(const Hypergraph *hypergraph,
                    const vector<double> &weights,
                    double bias)
  : hypergraph_(hypergraph),
    bias_(bias) {
      foreach(double weight, weights) {
        weights_.push_back(weight);
      }
      assert(weights.size() == hypergraph->edges().size());
  }

  SemiringType dot(const Hyperpath &path) const;

  SemiringType score(HEdge edge) const { return weights_[edge->id()]; }

  SemiringType bias() const { return bias_; }

  HypergraphWeights<SemiringType> *modify(const vector<SemiringType> &, SemiringType) const;
  // DEPRECATED:
  HypergraphWeights<SemiringType> *modify(const vector<double> &, double) const;

  HypergraphWeights<SemiringType> *project_weights(
      const HypergraphProjection &projection) const;

  void check(const Hypergraph &graph) const {
    if (!graph.same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match weights.");
    }
  }

 private:
  const Hypergraph *hypergraph_;
  vector<SemiringType> weights_;
  SemiringType bias_;
};

class HypergraphProjection {
 public:
  HypergraphProjection(const Hypergraph *original,
                       const Hypergraph *_new_graph,
                       const vector<HNode> *node_map,
                       const vector<HEdge> *edge_map)
      : original_graph(original),
      new_graph(_new_graph),
      node_map_(node_map),
      edge_map_(edge_map) {
        assert(node_map->size() == original_graph->nodes().size());
        assert(edge_map->size() == original_graph->edges().size());
        foreach (HNode node, *node_map) {
          assert(node == NULL ||
                 node->id() < (int)_new_graph->nodes().size());
        }
        foreach (HEdge edge, *edge_map) {
          assert(edge == NULL ||
                 edge->id() < (int)_new_graph->edges().size());
        }
      }

  ~HypergraphProjection() {
    delete node_map_;
    delete edge_map_;
  }

  static HypergraphProjection *project_hypergraph(
      const Hypergraph *hypergraph,
      vector<bool> edge_mask);

  HEdge project(HEdge original) const {
    return (*edge_map_)[original->id()];
  }

  HNode project(HNode original) const {
    return (*node_map_)[original->id()];
  }

  const Hypergraph *original_graph;
  const Hypergraph *new_graph;

 private:

  // Owned.
  const vector<HNode> *node_map_;
  const vector<HEdge> *edge_map_;
};


// Given an original hypergraph, builds a pruned hypergraph with only
// edges in the mask.
Hypergraph *build_pruned_hypergraph(
    const Hypergraph *hypergraph,
    vector<bool> edge_mask,
    vector<HEdge *> edge_map);

#endif  // HYPERGRAPH_HYPERGRAPH_H_
