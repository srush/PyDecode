// Copyright [2013] Alexander Rush

#include <typeinfo>
#include <vector>

#include "Hypergraph/Hypergraph.h"

int Hypergraph::ID = 0;

HEdge  Hypergraph::add_edge(const vector<HNode> &nodes)  {
    assert(lock_);
    //Hyperedge *edge = new Hyperedge(creating_node_, nodes);
    HEdge edge = temp_edges_.size();
    //creating_node_->add_edge(edge);
    temp_edges_.push_back(-1);
    temp_edge_tails_.push_back(nodes);
    temp_edge_heads_.push_back(creating_node_);
    return edge;
}

HNode Hypergraph::start_node() {
    terminal_lock_ = false;
    lock_ = true;
    creating_node_ = new Hypernode();
    creating_node_->set_id(temp_nodes_.size());
    temp_nodes_.push_back(creating_node_);
    return creating_node_;
}

bool Hypergraph::end_node() {
    assert(lock_);
    lock_ = false;

    // Remove this node if it has no edges.
    if (temp_edges_.size() > 0 && temp_edge_heads_.back() == creating_node_) {
        return true;
    } else {
        creating_node_->set_id(-1);
        temp_nodes_.pop_back();
        return false;
    }
}

HNode Hypergraph::add_terminal_node() {
    assert(terminal_lock_);
    Hypernode *node = new Hypernode();
    node->set_id(temp_nodes_.size());
    temp_nodes_.push_back(node);
    return temp_nodes_[temp_nodes_.size() - 1];
}

void Hypergraph::fill() {
    vector<bool> reachable_nodes(temp_nodes_.size(), false);
    vector<bool> reachable_edges(temp_edges_.size(), false);

    // Mark the reachable temp edges and nodes.
    for (int i = temp_edges_.size() - 1; i >= 0; --i) {
        vector<HNode> &edge = temp_edge_tails_[i];
        HNode head = temp_edge_heads_[i];
        if (head->id() == root()->id()) {
            reachable_nodes[head->id()] = true;
        }
        if (reachable_nodes[head->id()]) {
            reachable_edges[i] = true;
            foreach (HNode node, edge) {
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
            temp_edges_[i] = edge_count;
            edges_.push_back(edge_count);
            edge_tails_.push_back(temp_edge_tails_[i]);
            edge_heads_.push_back(temp_edge_heads_[i]);
            temp_edge_heads_[i]->add_edge(edge_count);
            edge_count++;
        } else {
            temp_edges_[i] = -1;
        }
    }
    temp_edge_tails_.clear();
    temp_edge_heads_.clear();
}
