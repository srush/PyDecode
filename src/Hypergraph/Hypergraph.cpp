// Copyright [2013] Alexander Rush
#include "Hypergraph/Hypergraph.h"


HEdge Hypergraph::add_edge(const vector<HNode> &nodes, string label)  {
  assert(lock_);
  int id = edges_.size();
  Hyperedge *edge = new Hyperedge(id, label, creating_node_, nodes);
  creating_node_->add_edge(edge);
  edges_.push_back(edge);
  return edge;
}

HNode Hypergraph::start_node() {
  lock_ = true;
  int id = nodes_.size();
  creating_node_ = new Hypernode(id);
  nodes_.push_back(creating_node_);
  return creating_node_;
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
  return new HypergraphWeights(hypergraph_, new_weights, bias_ + bias_dual);
}
