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

bool Hypergraph::end_node() {
  assert(lock_);
  lock_ = false;

  // Remove this node if it has no edges.
  if (creating_node_->edges().size() == 0) {
    creating_node_->set_id(-1);
    temp_nodes_.pop_back();
    return false;
  } else {
    return true;
  }
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
  for (HEdge edge : path.edges()) {
    score += weights_[edge->id()];
  }
  return score + bias_;
}

HypergraphWeights *HypergraphWeights::modify(
    const vector<double> &edge_duals,
    double bias_dual) const {
  vector<double> new_weights(weights_);
  for (uint i = 0; i < edge_duals.size(); ++i) {
    new_weights[i] += edge_duals[i];
  }
  return new HypergraphWeights(hypergraph_,
                               new_weights,
                               bias_ + bias_dual);
}

HypergraphWeights *HypergraphWeights::project_weights(
    const HypergraphProjection &projection) const {
  vector<double> weights(projection.new_graph->edges().size());
  for (HEdge edge : projection.original_graph->edges()) {
    HEdge new_edge = projection.project(edge);
    if (new_edge != NULL && new_edge->id() >= 0) {
      assert(new_edge->id() < projection.new_graph->edges().size());
      weights[new_edge->id()] = score(edge);
    }
  }
  return new HypergraphWeights(projection.new_graph, weights, bias_);
}


void Hypergraph::fill() {
  vector<bool> reachable_nodes(temp_nodes_.size(), false);
  vector<bool> reachable_edges(temp_edges_.size(), false);

  // Mark the reachable temp edges and nodes.
  for (int i = temp_edges_.size() - 1; i >= 0; --i) {
    HEdge edge = temp_edges_[i];
    if (edge->head_node()->id() == root()->id()) {
      reachable_nodes[edge->head_node()->id()] = true;
    }
    if (reachable_nodes[edge->head_node()->id()]) {
      reachable_edges[i] = true;
      for (HNode node : edge->tail_nodes()) {
        reachable_nodes[node->id()] = true;
      }
    }
  }

  // Relabel edges and nodes.
  int node_count = 0;
  for (uint i = 0; i < reachable_nodes.size(); ++i) {
    if (reachable_nodes[i]) {
      temp_nodes_[i]->set_id(node_count);
      nodes_.push_back(temp_nodes_[i]);
      node_count++;
    } else {
      temp_nodes_[i]->set_id(-1);
    }
  }
  int edge_count = 0;
  for (uint i = 0; i < reachable_edges.size(); ++i) {
    if (reachable_edges[i]) {
      temp_edges_[i]->set_id(edge_count);
      edges_.push_back(temp_edges_[i]);
      edge_count++;
    } else {
      temp_edges_[i]->set_id(-1);
    }
  }
}

HypergraphProjection *HypergraphProjection::project_hypergraph(
    const Hypergraph *hypergraph,
    vector<bool> edge_mask) {
  vector<HNode> *node_map =
      new vector<HNode>(hypergraph->nodes().size(), NULL);
  vector<HEdge> *edge_map =
      new vector<HEdge>(hypergraph->edges().size(), NULL);

  Hypergraph *new_graph = new Hypergraph();
  for (HNode node : hypergraph->nodes()) {
    if (node->terminal()) {
      // The node is a terminal, so just add it.
      (*node_map)[node->id()] =
          new_graph->add_terminal_node(node->label());
    } else {
      (*node_map)[node->id()] = new_graph->start_node(node->label());

      // Try to add each of the edges of the node.
      for (HEdge edge : node->edges()) {
        if (!edge_mask[edge->id()]) break;
        vector<HNode> tails;
        bool all_tails_exist = true;
        for (HNode tail_node : edge->tail_nodes()) {
          HNode new_tail_node = (*node_map)[tail_node->id()];
          if (new_tail_node == NULL) {
            // The tail node was pruned.
            all_tails_exist = false;
            break;
          } else {
            tails.push_back(new_tail_node);
          }
        }
        if (all_tails_exist) {
          HEdge new_edge = new_graph->add_edge(tails, edge->label());
          (*edge_map)[edge->id()] = new_edge;
        }
      }
      if (!new_graph->end_node()) {
        (*node_map)[node->id()] = NULL;
      }
    }
  }
  new_graph->finish();
  return new HypergraphProjection(hypergraph, new_graph,
                                  node_map, edge_map);
}
