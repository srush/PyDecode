// Copyright [2013] Alexander Rush

#include <algorithm>
#include <vector>
#include <queue>

#include "Hypergraph/Hypergraph.hh"

int Hypergraph::ID = 0;

HEdge HypergraphBuilder::add_edge(const vector<HNode> &nodes,
                                  int label)  {
    if (unary_) {
        return add_edge(nodes[0]);
    }
    assert(lock_);
    assert(!unary_);

    HEdge edge = temp_structure_->edges_.size();
    temp_structure_->edges_.push_back(edge);
    temp_structure_->edge_labels_.push_back(label);
    temp_structure_->edge_tails_.push_back(nodes);
    temp_structure_->edge_heads_.push_back(creating_node_);
    foreach (HNode node, nodes) {
        if (node == NODE_NULL) {
            throw HypergraphException("Hypernode is NULL.");
        }
    }
    return edge;
}

HEdge HypergraphBuilder::add_edge(HNode node, int label)  {
    assert(lock_);
    assert(unary_);

    HEdge edge = temp_structure_->edges_.size();
    temp_structure_->edges_.push_back(edge);
    temp_structure_->edge_labels_.push_back(label);
    temp_structure_->edge_tails_unary_.push_back(node);
    temp_structure_->edge_heads_.push_back(creating_node_);
    return edge;
}

HNode HypergraphBuilder::start_node(int label) {
    terminal_lock_ = false;
    lock_ = true;
    // creating_node_ = new Hypernode();
    // creating_node_->set_id(temp_structure_->nodes_.size());
    creating_node_ = temp_structure_->nodes_.size();
    temp_structure_->node_labels_.push_back(label);
    temp_structure_->nodes_.push_back(creating_node_);
    return creating_node_;
}

bool HypergraphBuilder::end_node() {
    assert(lock_);
    lock_ = false;

    // Remove this node if it has no edges.
    if (temp_structure_->edges_.size() > 0 &&
        temp_structure_->edge_heads_.back() == creating_node_) {
        return true;
    } else {
        // creating_node_->set_id(-1);
        temp_structure_->nodes_.pop_back();
        temp_structure_->node_labels_.pop_back();
        return false;
    }
}

HNode HypergraphBuilder::add_terminal_node(int label) {
    assert(terminal_lock_);
    // Hypernode *node = new Hypernode();
    // node->set_id(temp_structure_->nodes_.size());

    HNode node = temp_structure_->nodes_.size();
    temp_structure_->node_labels_.push_back(label);
    temp_structure_->nodes_.push_back(node);
    return temp_structure_->nodes_[temp_structure_->nodes_.size() - 1];
}

Hypergraph *HypergraphBuilder::finish(bool reconstruct) {
    if (temp_structure_->nodes_.size() == 0) {
        throw HypergraphException("Hypergraph has size 0.");
    }
    HNode root = temp_structure_->nodes_[temp_structure_->nodes_.size() - 1];
    /* if (!(root == NULL || root_ == root)) { */
    /*   throw HypergraphException("Root is not expected root."); */
    /* } */

    _HypergraphStructure *structure;
    if (reconstruct) {
        structure = new _HypergraphStructure();
        root = fill(structure, root);
        delete temp_structure_;
    } else {
        structure = temp_structure_;
        structure->node_edges_.resize(structure->nodes_.size());
        for (uint i = 0; i < structure->edges_.size(); ++i) {
            structure->node_edges_[structure->edge_heads_[i]]
                    .push_back(i);
        }
    }

    if (structure->nodes_.size() == 0) {
        throw HypergraphException("Final hypergraph has node size 0.");
    }
    if (structure->edges_.size() == 0) {
        throw HypergraphException("Final hypergraph has edge size 0.");
    }

    return new Hypergraph(structure, root, unary_);
    // TODO(srush) Run checks to make sure we are complete.
}

HNode HypergraphBuilder::fill(_HypergraphStructure *structure,
                             HNode root) {
    vector<bool> reachable_nodes(temp_structure_->nodes_.size(), false);
    vector<bool> reachable_edges(temp_structure_->edges_.size(), false);

    // Mark the reachable temp edges and nodes.
    for (int i = temp_structure_->edges_.size() - 1; i >= 0; --i) {
        HNode head = temp_structure_->edge_heads_[i];
        if (head == root) {
            reachable_nodes[head] = true;
        }
        if (reachable_nodes[head]) {
            reachable_edges[i] = true;
            if (!unary_) {
                vector<HNode> &edge = temp_structure_->edge_tails_[i];
                foreach (HNode node, edge) {
                    reachable_nodes[node] = true;
                }
            } else {
                reachable_nodes[temp_structure_->edge_tails_unary_[i]] = true;
            }
        }
    }

    // Relabel edges and nodes.
    int node_count = 0;
    vector<int> node_mapping(temp_structure_->nodes_.size(),
                             NODE_NULL);
    for (uint i = 0; i < reachable_nodes.size(); ++i) {
        if (reachable_nodes[i]) {
            structure->nodes_.push_back(node_count);
            structure->node_labels_.push_back(temp_structure_->node_labels_[i]);
            node_mapping[i] = node_count;
            node_count++;
        }
    }
    structure->node_edges_.resize(structure->nodes_.size());
    int edge_count = 0;
    for (uint i = 0; i < reachable_edges.size(); ++i) {
        if (reachable_edges[i]) {
            temp_structure_->edges_[i] = edge_count;
            structure->edges_.push_back(edge_count);
            if (!unary_) {
                vector<HNode> tail_nodes;
                foreach (HNode node, temp_structure_->edge_tails_[i]) {
                    tail_nodes.push_back(node_mapping[node]);
                }
                structure->edge_tails_.push_back(tail_nodes);
            } else {
                structure->edge_tails_unary_.push_back(
                    node_mapping[temp_structure_->edge_tails_unary_[i]]);
            }
            HNode head = node_mapping[temp_structure_->edge_heads_[i]];
            structure->edge_heads_.push_back(head);
            structure->edge_labels_.push_back(
                temp_structure_->edge_labels_[i]);
            structure->node_edges_[head].push_back(edge_count);
            edge_count++;
        } else {
            temp_structure_->edges_[i] = EDGE_NULL;
        }
    }
    temp_structure_->edge_tails_.clear();
    temp_structure_->edge_heads_.clear();
    return node_mapping[root];
}

Hyperpath *construct_path(const Hypergraph *graph,
                          int *back) {
    // Collect backpointers.
    bool unary = graph->is_unary();
    vector<HEdge> path;
    vector<HNode> node_path;
    if (unary) {
        HNode cur = graph->root();
        node_path.push_back(cur);
        while (!graph->terminal(cur)) {
            HEdge edge = back[cur];
            path.push_back(edge);
            cur = graph->tail_node(edge);
            node_path.push_back(cur);
        }
        reverse(path.begin(), path.end());
        reverse(node_path.begin(), node_path.end());
    } else {
        queue<HNode> to_examine;
        to_examine.push(graph->root());
        while (!to_examine.empty()) {
            HNode node = to_examine.front();
            node_path.push_back(node);
            HEdge edge = back[node];
            to_examine.pop();
            if (edge == -1) {
                assert(graph->terminal(node));
                continue;
            }
            path.push_back(edge);
            for (int i = 0; i < graph->tail_nodes(edge); ++i) {
                to_examine.push(graph->tail_node(edge, i));
            }
        }
        sort(path.begin(), path.end(), IdComparator());
        sort(node_path.begin(), node_path.end(), IdComparator());
    }
    return new Hyperpath(graph, node_path, path);
}
