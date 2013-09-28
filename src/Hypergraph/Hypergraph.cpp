// Copyright [2013] Alexander Rush
#include "Hypergraph/Hypergraph.h"


HEdge Hypergraph::add_edge(const vector<HNode> &nodes, string label)  {
  assert(lock_);
  Hyperedge *edge = new Hyperedge(label, creating_node_, nodes);
  creating_node_->add_edge(edge);
  temp_edges_.push_back(edge);
  return edge;
}

HNode Hypergraph::start_node(string label) {
  terminal_lock_ = false;
  lock_ = true;
  creating_node_ = new Hypernode(label);
  creating_node_->set_id(temp_nodes_.size());
  temp_nodes_.push_back(creating_node_);
  return creating_node_;
}

HNode Hypergraph::add_terminal_node(string label) {
  assert(terminal_lock_);
  Hypernode *node = new Hypernode(label);
  node->set_id(temp_nodes_.size());
  temp_nodes_.push_back(node);
  return temp_nodes_[temp_nodes_.size() - 1];
}

double HypergraphWeights::dot(const Hyperpath &path) const {
  double score = 0.0;
  foreach (HEdge edge, path.edges()) {
    score += weights_[edge->id()];
  }
  return score + bias_;
}

HypergraphWeights *HypergraphWeights::modify(const SparseVec &edge_duals,
                                             double bias_dual) const {
  vector<double> new_weights(weights_);
  for(SparseVec::const_iterator i = edge_duals.begin();
      i != edge_duals.end(); ++i ) {
    new_weights[i.index()] += edge_duals[i.index()];
  }
  return new HypergraphWeights(hypergraph_, 
                               new_weights, bias_ + bias_dual);
}


void Hypergraph::fill() {
  vector<bool> reachable_nodes(temp_nodes_.size(), false);
  vector<bool> reachable_edges(temp_edges_.size(), false);  

  // Outside order.
  for (int i = temp_edges_.size() - 1; i >= 0; --i) {
    HEdge edge = temp_edges_[i];
    if (edge->head_node()->id() == root()->id()) {
      reachable_nodes[edge->head_node()->id()] = true;
    }
    if (reachable_nodes[edge->head_node()->id()]) {
      reachable_edges[i] = true;
      foreach (HNode node, edge->tail_nodes()) {
        reachable_nodes[node->id()] = true;
      }
    }
  }
  int node_count = 0;
  for (int i = 0; i < reachable_nodes.size(); ++i) {
    if (reachable_nodes[i]) {
      temp_nodes_[i]->set_id(node_count);
      nodes_.push_back(temp_nodes_[i]);
      node_count++;
    }
  }
  int edge_count = 0;
  for (int i = 0; i < reachable_edges.size(); ++i) {
    if (reachable_edges[i]) {
      temp_edges_[i]->set_id(edge_count);
      edges_.push_back(temp_edges_[i]);
      edge_count++;
    }
  }
}
