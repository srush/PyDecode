// Copyright [2013] Alexander Rush


#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "Hypergraph/Algorithms.h"
#include "Hypergraph/Potentials.h"


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


template<typename StatSem>
struct NodeScore {
    NodeScore()
            :
            count(-1),
            edge(NULL),
            back(0),
            score(StatSem::zero()) {}

    NodeScore(int _count, HEdge _edge,
              typename StatSem::ValType _score)
            :
            count(_count),
            edge(_edge),
            back(0),
            score(_score) {}

    NodeScore(int _count, HEdge _edge, int i, int j,
              typename StatSem::ValType _score)
            :
            count(_count),
            edge(_edge),
            back(2),
            score(_score) {
        back[0] = i;
        back[1] = j;
    }

    int count;
    HEdge edge;
    vector<int> back;
    typename StatSem::ValType score;
};


struct EdgeGroup {
    EdgeGroup(HEdge _edge, int back_left, int back_right)
            : edge(_edge), back(2) {
        back[0] = back_left;
        back[1] = back_right;
    }
    HEdge edge;
    vector<int> back;
};

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

    Hypergraph *new_graph = new Hypergraph();
    vector<vector<NodeCount > > new_nodes(hypergraph->nodes().size());

    vector<vector<HNode> > reverse_node_map(hypergraph->nodes().size());;
    vector<vector<HEdge> > reverse_edge_map(hypergraph->edges().size());;

    foreach (HNode node, hypergraph->nodes()) {
        if (node->terminal()) {
            // The node is a terminal, so just add it.
            new_nodes[node->id()].push_back(
                NodeCount(0,
                          new_graph->add_terminal_node()));
        } else {
            // Bucket edges.
            vector<vector<EdgeGroup> > counts(limit + 1);
            foreach (HEdge edge, node->edges()) {
                bool unary = edge->tail_nodes().size() == 1;
                assert(edge->tail_nodes().size() <= 2);
                HNode left_node = edge->tail_nodes()[0];

                int score = potentials.score(edge);
                vector<NodeCount> &base = new_nodes[left_node->id()];

                for (int i = 0; i < base.size(); ++i) {
                    int total = score + base[i].count;
                    if (total < lower_limit || total > upper_limit) continue;

                    if (unary) {
                        counts[total - lower_limit].push_back(
                            EdgeGroup(edge, i, 0));
                    } else {
                        HNode right_node = edge->tail_nodes()[1];
                        int right_size = new_nodes[right_node->id()].size();
                        for (int j = 0; j < right_size; ++j) {
                            int total =
                                    score + new_nodes[left_node->id()][i].count
                                    + new_nodes[right_node->id()][j].count;
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
                if (hypergraph->root()->id() == node->id() &&
                    count != modified_goal) continue;

                new_nodes[node->id()].push_back(
                    NodeCount(count + lower_limit,
                              new_graph->start_node()));

                foreach (EdgeGroup edge_group, counts[count]) {
                    HEdge edge = edge_group.edge;
                    vector<HNode> tails;
                    for (int i = 0; i < edge->tail_nodes().size(); ++i) {
                        tails.push_back(new_nodes[edge->tail_nodes()[i]->id()]
                                        [edge_group.back[i]].node);
                    }
                    HEdge new_edge = new_graph->add_edge(tails);
                    reverse_edge_map[edge->id()].push_back(new_edge);
                }
                if (new_graph->end_node()) {
                    reverse_node_map[node->id()].push_back(
                        new_nodes[node->id()].back().node);
                }
            }
        }
    }
    new_graph->finish();

    // Create node maps.
    vector<HNode> *node_map =
            new vector<HNode>(new_graph->nodes().size(), NULL);
    vector<HEdge> *edge_map =
            new vector<HEdge>(new_graph->edges().size(), NULL);


    foreach (HNode node, hypergraph->nodes()) {
        foreach (HNode new_node, reverse_node_map[node->id()]) {
            if (new_node->id() == -1) continue;
            (*node_map)[new_node->id()] = node;
        }
    }
    foreach (HEdge edge, hypergraph->edges()) {
        foreach (HEdge new_edge, reverse_edge_map[edge->id()]) {
            if (new_edge->id() == -1) continue;
            (*edge_map)[new_edge->id()] = edge;
        }
    }
    return new HypergraphMap(new_graph, hypergraph,
                                    node_map, edge_map, false);
}

Chart<SetPotential> *edge_domination(const Hypergraph &graph) {
    vector<set<int> > *v = new vector<set<int> >(graph.edges().size());
    foreach (HEdge edge, graph.edges()) {
        (*v)[edge->id()].insert(edge->id());
    }

    HypergraphVectorPotentials<SetPotential> potentials(
        &graph, v, set<int>(), false /* copy */);
    Chart<SetPotential> *chart = general_inside(&graph, potentials);
    return chart;
}

Chart<SetPotential> *node_domination(const Hypergraph &graph) {
    vector<set<int> > *v = new vector<set<int> >(graph.edges().size());
    foreach (HEdge edge, graph.edges()) {
        foreach (HNode node, edge->tail_nodes()) {
            (*v)[edge->id()].insert(node->id());
        }
    }

    HypergraphVectorPotentials<SetPotential> potentials(
        &graph, v, set<int>(), false /* copy */);
    Chart<SetPotential> *chart = general_inside(&graph, potentials);
    return chart;
}

vector<set<int> > *children_sparse(
    const Hypergraph *graph,
    const HypergraphPotentials<SparseVectorPotential> &potentials) {
    vector<set<int> > *children = new vector<set<int> >(graph->nodes().size());
    foreach (HNode node, graph->nodes()) {
        set<int> &vec = (*children)[node->id()];
        foreach (HEdge edge, node->edges()) {
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
                                             children[node->id()])) {
            updated_nodes->insert(node->id());
        }
    }
    return updated_nodes;
}

vector<set<int> > *children_nodes(const Hypergraph &graph) {
    vector<set<int> > *children =
            new vector<set<int> >(graph.nodes().size());
    foreach (HNode node, graph.nodes()) {
        foreach (HEdge edge, node->edges()) {
            foreach (HNode tail, edge->tail_nodes()) {
                (*children)[node->id()].insert(tail->id());
            }
        }
        (*children)[node->id()].insert(node->id());
    }
    return children;
}


HypergraphMap *project_hypergraph(
    const Hypergraph *hypergraph,
    const HypergraphPotentials<BoolPotential> &edge_mask) {
    vector<HNode> *node_map =
            new vector<HNode>(hypergraph->nodes().size(), NULL);
    vector<HEdge> *edge_map =
            new vector<HEdge>(hypergraph->edges().size(), NULL);

    Hypergraph *new_graph = new Hypergraph();
    foreach (HNode node, hypergraph->nodes()) {
        if (node->terminal()) {
            // The node is a terminal, so just add it.
            (*node_map)[node->id()] =
                    new_graph->add_terminal_node();
        } else {
            (*node_map)[node->id()] = new_graph->start_node();

            // Try to add each of the edges of the node.
            foreach (HEdge edge, node->edges()) {
                if (!static_cast<bool>(edge_mask[edge])) continue;
                vector<HNode> tails;
                bool all_tails_exist = true;
                foreach (HNode tail_node, edge->tail_nodes()) {
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
                    HEdge new_edge = new_graph->add_edge(tails);
                    (*edge_map)[edge->id()] = new_edge;
                }
            }
            bool success = true;
            if (!new_graph->end_node()) {
                (*node_map)[node->id()] = NULL;
                success = false;
            }
            if (hypergraph->root()->id() == node->id()) {
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
            new vector<HNode>(hypergraph->nodes().size(), NULL);
    vector<HEdge> *edges =
            new vector<HEdge>(hypergraph->edges().size(), NULL);

    foreach (HNode node, hypergraph->nodes()) {
        if (node->terminal()) {
            (*nodes)[node->id()] = range_hyper->add_terminal_node();
            continue;
        }
        vector<vector<HNode> > new_edges;
        foreach (HEdge edge, node->edges()) {
            vector<HNode> new_edge(2);
            if (edge->tail_nodes().size() <= 2) {
                vector<HNode> new_tail;
                foreach (HNode tnode, edge->tail_nodes()) {
                    new_tail.push_back((*nodes)[tnode->id()]);
                }
                new_edges.push_back(new_tail);
                continue;
            }
            new_edge[0] = (*nodes)[edge->tail_nodes()[0]->id()];
            new_edge[1] = (*nodes)[edge->tail_nodes()[1]->id()];
            HNode new_node = range_hyper->start_node();
            range_hyper->add_edge(new_edge);
            range_hyper->end_node();

            for (int i = 2; i < edge->tail_nodes().size() - 1; ++i) {
                HNode tnode = edge->tail_nodes()[i];
                HNode tmp = range_hyper->start_node();
                new_edge[0] = new_node;
                new_edge[1] = (*nodes)[tnode->id()];
                range_hyper->add_edge(new_edge);
                range_hyper->end_node();
                new_node = tmp;
            }
            new_edge[0] = new_node;
            new_edge[1] = (*nodes)[edge->tail_nodes().back()->id()];
            new_edges.push_back(new_edge);
        }
        (*nodes)[node->id()] = range_hyper->start_node();

        assert(new_edges.size() == node->edges().size());
        for (int i = 0; i < new_edges.size(); ++i) {
            (*edges)[node->edges()[i]->id()] =
                    range_hyper->add_edge(new_edges[i]);
        }
        range_hyper->end_node();
    }
    return new HypergraphMap(hypergraph, range_hyper,
                             nodes, edges, true);
}


Hypergraph *make_lattice(int width, int height,
                         const vector<vector<int> > &transitions,
                         vector<LatticeLabel> *labels) {
    Hypergraph *graph = new Hypergraph();
    HNode source = graph->add_terminal_node();
    labels->push_back(LatticeLabel(0, 0));

    vector<HNode> old_nodes(height), new_nodes(height);
    for (int j = 0; j < height; ++j) {
        old_nodes[j] = graph->start_node();
        labels->push_back(LatticeLabel(1, j));
        vector<HNode> tails;
        tails.push_back(source);
        graph->add_edge(tails);
        graph->end_node();
    }

    for (int i = 1; i <= width; ++i) {
        for (int j = 0; j < height; ++j) {
            new_nodes[j] = graph->start_node();
            labels->push_back(LatticeLabel(i, j));
            foreach (int k, transitions[j]) {
                vector<HNode> tails;
                tails.push_back(old_nodes[k]);
                graph->add_edge(tails);
            }
            graph->end_node();
        }
        old_nodes = new_nodes;
    }

    graph->start_node();
    labels->push_back(LatticeLabel(width + 1, 0));
    for (int j = 0; j < height; ++j) {
        vector<HNode> tails;
        tails.push_back(old_nodes[j]);
        graph->add_edge(tails);
    }
    graph->end_node();

    graph->finish();
    return graph;
}


// vector<bool> active_nodes(const Hypergraph *graph, set<int> on_edges) {
//     Chart<SetPotential> *edges = edge_domination;

// }
