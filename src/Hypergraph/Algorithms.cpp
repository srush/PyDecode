// Copyright [2013] Alexander Rush


#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "Hypergraph/Algorithms.hh"
#include "Hypergraph/Potentials.hh"

using namespace std;

template<class Set1, class Set2>
bool is_disjoint(const Set1 &set1, const Set2 &set2) {
    if (set1.empty() || set2.empty()) return true;

    typename Set1::const_iterator
            it1 = set1.begin(),
            it1End = set1.end();
    typename Set2::const_iterator
            it2 = set2.begin(),
            it2End = set2.end();

    if (*it1 > *set2.rbegin() || *it2 > *set1.rbegin()) return true;

    while (it1 != it1End && it2 != it2End) {
        if (*it1 == *it2) return false;
        if (*it1 < *it2) {
            it1++;
        } else {
            it2++;
        }
    }

    return true;
}

struct EdgeGroup {
    EdgeGroup(HEdge _edge, int back_left, int back_right)
            : edge(_edge), back(2) {
        back[0] = back_left;
        back[1] = back_right;
    }
    HEdge edge;
    vector<int> back;
};


// Implementation of the Bar-Hillel algorithm.
HypergraphMap *extend_with_dfa(
    Hypergraph *hypergraph,
    const HypergraphPotentials<CountingPotential> &potentials,
    const DFA &dfa,
    vector<DFANode> *labels) {

    Hypergraph *new_graph = new Hypergraph(hypergraph->is_unary());
    vector<vector<DFANode> > new_nodes(
        hypergraph->nodes().size());
    vector<vector<vector<DFANode> > > new_indexed(
        hypergraph->nodes().size());

    vector<DFANode> possible_labels;

    foreach (HNode node, hypergraph->nodes()) {
        new_indexed[node].resize(dfa.states().size());
    }

    vector<vector<HNode> > reverse_node_map(
        hypergraph->nodes().size());
    vector<vector<HEdge> > reverse_edge_map(
        hypergraph->edges().size());

    foreach (HNode node, hypergraph->nodes()) {
        if (hypergraph->terminal(node)) {
            foreach (int state, dfa.states()) {
                HNode tmp_node = new_graph->add_terminal_node();
                reverse_node_map[node].push_back(tmp_node);
                int id = new_nodes[node].size();
                DFANode label(state, state, tmp_node, id);
                possible_labels.push_back(label);
                new_nodes[node].push_back(label);
                new_indexed[node][state].push_back(label);
            }
        } else {
            set<int> lefts;
            set<int> rights;
            vector<vector<vector<EdgeGroup> > > hyps(dfa.states().size());
            foreach (int state, dfa.states()) {
                hyps[state].resize(dfa.states().size());
            }
            foreach (HEdge edge, hypergraph->edges(node)) {
                bool unary = hypergraph->tail_nodes(edge) == 1;
                HNode left_node = hypergraph->tail_node(edge, 0);
                vector<DFANode> &base = new_nodes[left_node];
                int symbol = potentials.score(edge);
                for (int i = 0; i < base.size(); ++i) {
                    if (unary) {
                        if (!dfa.valid_transition(base[i].right_state,
                                                  symbol)) {
                            continue;
                        }

                        int right_state =
                                dfa.transition(base[i].right_state,
                                               symbol);

                        hyps[base[i].left_state][right_state].push_back(
                            EdgeGroup(edge, i, 0));
                        lefts.insert(base[i].left_state);
                        rights.insert(right_state);

                    } else {
                        HNode right_node = hypergraph->tail_node(edge, 1);
                        vector<DFANode> &base_right =
                                new_indexed[right_node][base[i].right_state];
                        for (int j = 0; j < base_right.size(); ++j) {
                            if (!dfa.valid_transition(base_right[j].right_state,
                                                      symbol)) {
                                continue;
                            }
                            int right_state =
                                    dfa.transition(base_right[j].right_state,
                                                   symbol);
                            hyps[base[i].left_state][right_state].push_back(
                                EdgeGroup(edge, i, base_right[j].id));

                            lefts.insert(base[i].left_state);
                            rights.insert(right_state);
                        }
                    }
                }
            }

            // Make new decorated nodes.
            for (set<int>::iterator iter = lefts.begin();
                 iter != lefts.end(); ++iter) {
                int state_left = *iter;
                for (set<int>::iterator iter2 = rights.begin();
                     iter2 != rights.end(); ++iter2) {
                    int state_right = *iter2;
                    if (hyps[state_left][state_right].size() == 0) continue;
                    if (hypergraph->root() == node &&
                        !(state_left == 0 &&
                          dfa.final(state_right))) continue;

                    HNode tmp_node = new_graph->start_node();
                    int id = new_nodes[node].size();
                    DFANode label(state_left, state_right,
                                  tmp_node, id);
                    new_nodes[node].push_back(label);
                    new_indexed[node][state_left].push_back(label);

                    foreach (EdgeGroup edge_group, hyps[state_left][state_right]) {
                        HEdge edge = edge_group.edge;
                        vector<HNode> tails;
                        for (int i = 0; i < hypergraph->tail_nodes(edge); ++i) {
                            tails.push_back(new_nodes[hypergraph->tail_node(edge, i)]
                                            [edge_group.back[i]].node);
                        }
                        HEdge new_edge = new_graph->add_edge(tails);
                        reverse_edge_map[hypergraph->id(edge)].push_back(new_edge);
                    }
                    if (new_graph->end_node()) {
                        reverse_node_map[node].push_back(
                            new_nodes[node].back().node);
                        possible_labels.push_back(label);
                    }
                }
            }
        }
    }
    if (reverse_node_map[hypergraph->root()].size() != 1) {
        throw HypergraphException("New hypergraph has root size is not 1.");
    }
    new_graph->finish();


    foreach (DFANode label, possible_labels) {
        if (label.node != NODE_NULL && label.node >= 0) {
            labels->push_back(label);
        }
    }

    return HypergraphMap::make_reverse_map(
        new_graph, hypergraph,
        reverse_node_map, reverse_edge_map);
}


struct NodeCount {
    NodeCount(int _count, HNode _node) {
        count = _count;
        node = _node;
    }
    int count;
    HNode node;
};

HypergraphMap *extend_hypergraph_by_count(
    Hypergraph *hypergraph,
    const HypergraphPotentials<CountingPotential> &potentials,
    int lower_limit,
    int upper_limit,
    int goal) {

    int limit = upper_limit - lower_limit;
    int modified_goal = goal - lower_limit;

    Hypergraph *new_graph = new Hypergraph(hypergraph->is_unary());
    vector<vector<HNode> > reverse_node_map(hypergraph->nodes().size());
    vector<vector<HEdge> > reverse_edge_map(hypergraph->edges().size());
    vector<vector<NodeCount > > new_nodes(hypergraph->nodes().size());

    foreach (HNode node, hypergraph->nodes()) {
        if (hypergraph->terminal(node)) {
            // The node is a terminal, so just add it.
            new_nodes[node].push_back(
                NodeCount(0,
                          new_graph->add_terminal_node()));
        } else {
            // Bucket edges.
            vector<vector<EdgeGroup> > counts(limit + 1);
            foreach (HEdge edge, hypergraph->edges(node)) {
                bool unary = hypergraph->tail_nodes(edge) == 1;
                assert(hypergraph->tail_nodes(edge) <= 2);
                HNode left_node = hypergraph->tail_node(edge, 0);

                int score = potentials.score(edge);
                vector<NodeCount> &base = new_nodes[left_node];

                for (int i = 0; i < base.size(); ++i) {
                    int total = score + base[i].count;
                    if (total < lower_limit || total > upper_limit) continue;

                    if (unary) {
                        counts[total - lower_limit].push_back(
                            EdgeGroup(edge, i, 0));
                    } else {
                        HNode right_node = hypergraph->tail_node(edge, 1);
                        int right_size = new_nodes[right_node].size();
                        for (int j = 0; j < right_size; ++j) {
                            int total =
                                    score + new_nodes[left_node][i].count
                                    + new_nodes[right_node][j].count;
                            if (total < lower_limit ||
                                total > upper_limit) continue;
                            counts[total - lower_limit].push_back(
                                EdgeGroup(edge, i, j));
                        }
                    }
                }
            }

            // Make new nodes.
            for (int count = 0; count <= limit; ++count) {
                if (counts[count].size() == 0) continue;
                if (hypergraph->root() == node &&
                    count != modified_goal) continue;

                new_nodes[node].push_back(
                    NodeCount(count + lower_limit,
                              new_graph->start_node()));

                foreach (EdgeGroup edge_group, counts[count]) {
                    HEdge edge = edge_group.edge;
                    vector<HNode> tails;
                    for (int i = 0; i < hypergraph->tail_nodes(edge); ++i) {
                        tails.push_back(new_nodes[hypergraph->tail_node(edge, i)]
                                        [edge_group.back[i]].node);
                    }
                    HEdge new_edge = new_graph->add_edge(tails);
                    reverse_edge_map[hypergraph->id(edge)].push_back(new_edge);
                }
                if (new_graph->end_node()) {
                    reverse_node_map[node].push_back(
                        new_nodes[node].back().node);
                }
            }
        }
    }
    new_graph->finish();

    return HypergraphMap::make_reverse_map(
        new_graph, hypergraph,
        reverse_node_map, reverse_edge_map);
}


// Chart<SetPotential> *edge_domination(const Hypergraph &graph) {
//     vector<set<int> > *v = new vector<set<int> >(graph.edges().size());
//     foreach (HEdge edge, graph.edges()) {
//         (*v)[graph.id(edge)].insert(graph.id(edge));
//     }

//     HypergraphVectorPotentials<SetPotential> potentials(
//         &graph, v, false /* copy */);
//     Chart<SetPotential> *chart = general_inside(&graph, potentials);
//     return chart;
// }

// Chart<SetPotential> *node_domination(const Hypergraph &graph) {
//     vector<set<int> > *v = new vector<set<int> >(graph.edges().size());
//     foreach (HEdge edge, graph.edges()) {
//         for (int i = 0; i < graph.tail_nodes(edge); ++i) {
//             HNode node = graph.tail_node(edge, i);
//             (*v)[graph.id(edge)].insert(node);
//         }
//     }

//     HypergraphVectorPotentials<SetPotential> potentials(
//         &graph, v, false /* copy */);
//     Chart<SetPotential> *chart = general_inside(&graph, potentials);
//     return chart;
// }

vector<set<int> > *children_sparse(
    const Hypergraph *graph,
    const HypergraphPotentials<SparseVectorPotential> &potentials) {
    vector<set<int> > *children = new vector<set<int> >(graph->nodes().size());
    foreach (HNode node, graph->nodes()) {
        set<int> &vec = (*children)[node];
        foreach (HEdge edge, graph->edges(node)) {
            const SparseVector &v = potentials.score(edge);
            foreach (const SparsePair &pair, v) {
                vec.insert(pair.first);
            }
        }
    }
    return children;
}

set<int> *updated_nodes(
    const Hypergraph *graph,
    const vector<set<int> > &children,
    const set<int> &updated) {
    set<int> *updated_nodes = new set<int>();
    foreach (HNode node, graph->nodes()) {
        if (!is_disjoint<set<int>, set<int> >(updated,
                                             children[node])) {
            updated_nodes->insert(node);
        }
    }
    return updated_nodes;
}

vector<set<int> > *children_nodes(const Hypergraph &graph) {
    vector<set<int> > *children =
            new vector<set<int> >(graph.nodes().size());
    foreach (HNode node, graph.nodes()) {
        foreach (HEdge edge, graph.edges(node)) {
            for (int i = 0; i < graph.tail_nodes(edge); ++i) {
                HNode tail = graph.tail_node(edge, i);
                (*children)[node].insert(tail);
            }
        }
        (*children)[node].insert(node);
    }
    return children;
}


HypergraphMap *project_hypergraph(
    const Hypergraph *hypergraph,
    const HypergraphPotentials<BoolPotential> &edge_mask) {
    vector<HNode> *node_map =
            new vector<HNode>(hypergraph->nodes().size(), NODE_NULL);
    vector<HEdge> *edge_map =
            new vector<HEdge>(hypergraph->edges().size(), EDGE_NULL);

    Hypergraph *new_graph = new Hypergraph();
    foreach (HNode node, hypergraph->nodes()) {
        if (hypergraph->terminal(node)) {
            // The node is a terminal, so just add it.
            (*node_map)[node] =
                    new_graph->add_terminal_node();
        } else {
            (*node_map)[node] = new_graph->start_node();

            // Try to add each of the edges of the node.
            foreach (HEdge edge, hypergraph->edges(node)) {
                if (!static_cast<bool>(edge_mask[edge])) continue;
                vector<HNode> tails;
                bool all_tails_exist = true;
                for (int i = 0; i < hypergraph->tail_nodes(edge); ++i) {
                    HNode tail_node =  hypergraph->tail_node(edge, i );
                    HNode new_tail_node = (*node_map)[tail_node];
                    if (new_tail_node == NODE_NULL) {
                        // The tail node was pruned.
                        all_tails_exist = false;
                        break;
                    } else {
                        tails.push_back(new_tail_node);
                    }
                }
                if (all_tails_exist) {
                    HEdge new_edge = new_graph->add_edge(tails);
                    (*edge_map)[hypergraph->id(edge)] = new_edge;
                }
            }
            bool success = true;
            if (!new_graph->end_node()) {
                (*node_map)[node] = NODE_NULL;
                success = false;
            }
            if (hypergraph->root() == node) {
                assert(success);
            }
        }
    }
    new_graph->finish();
    return new HypergraphMap(hypergraph, new_graph,
                             node_map, edge_map, true);
}

HypergraphMap *binarize(const Hypergraph *hypergraph) {
    Hypergraph *range_hyper = new Hypergraph();
    vector<HNode> *nodes =
            new vector<HNode>(hypergraph->nodes().size(), NODE_NULL);
    vector<HEdge> *edges =
            new vector<HEdge>(hypergraph->edges().size(), EDGE_NULL);

    foreach (HNode node, hypergraph->nodes()) {
        if (hypergraph->terminal(node)) {
            (*nodes)[node] = range_hyper->add_terminal_node();
            continue;
        }
        vector<vector<HNode> > new_edges;
        foreach (HEdge edge, hypergraph->edges(node)) {
            vector<HNode> new_edge(2);
            if (hypergraph->tail_nodes(edge) <= 2) {
                vector<HNode> new_tail;
                for (int i = 0; i < hypergraph->tail_nodes(edge); ++i) {
                    HNode tnode = hypergraph->tail_node(edge, i);
                    new_tail.push_back((*nodes)[tnode]);
                }
                new_edges.push_back(new_tail);
                continue;
            }
            new_edge[0] = (*nodes)[hypergraph->tail_node(edge, 0)];
            new_edge[1] = (*nodes)[hypergraph->tail_node(edge, 1)];
            HNode new_node = range_hyper->start_node();
            range_hyper->add_edge(new_edge);
            range_hyper->end_node();

            for (int i = 2; i < hypergraph->tail_nodes(edge) - 1; ++i) {
                HNode tnode = hypergraph->tail_node(edge, i);
                HNode tmp = range_hyper->start_node();
                new_edge[0] = new_node;
                new_edge[1] = (*nodes)[tnode];
                range_hyper->add_edge(new_edge);
                range_hyper->end_node();
                new_node = tmp;
            }
            new_edge[0] = new_node;
            new_edge[1] = (*nodes)[hypergraph->tail_node(edge, hypergraph->tail_nodes(edge) - 1)];
            new_edges.push_back(new_edge);
        }
        (*nodes)[node] = range_hyper->start_node();

         // assert(new_edges.size() == node->edges().size());
        for (uint i = 0; i < new_edges.size(); ++i) {
            (*edges)[i] = range_hyper->add_edge(new_edges[i]);
        }
        range_hyper->end_node();
    }
    return new HypergraphMap(hypergraph, range_hyper,
                             nodes, edges, true);
}

// HypergraphMap *to_lattice(const Hypergraph *hypergraph) {
//     stack<int> vertices;
//     vertices.push(hypergraph->root().id());
//     Hypergraph *new_hypergraph;
//     while (!vertices.empty()) {
//         int node_id = vertices.top();
//         vertices.pop();
//         HNode node = hypergraph->nodes()[node_id];
//         new_hypergraph->start_node();
//         foreach(HEdge edge, node->edges()) {
//             // Start Node
//             vector<HNode> &tails = edge->tail_nodes();

//             foreach (HNode tail_node, tails) {
//                 tail_node->id();
//             }
//             // End Node

//         }
//     }
// }

//     graph->start_node();
//     labels->push_back(LatticeLabel(width + 1, 0));
//     for (int j = 0; j < height; ++j) {
//         graph->add_edge(old_nodes[j]);
//     }
//     graph->end_node();

//     graph->finish();
//     return graph;
// }


// vector<bool> active_nodes(const Hypergraph *graph, set<int> on_edges) {
//     Chart<SetPotential> *edges = edge_domination;

// }
