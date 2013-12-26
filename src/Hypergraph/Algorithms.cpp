// Copyright [2013] Alexander Rush


#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#include "Hypergraph/Algorithms.h"
#include "Hypergraph/Potentials.h"

#define SPECIALIZE_ALGORITHMS_FOR_SEMI(X)\
  template class Chart<X>;\
  template class Marginals<X>;\
  template void general_viterbi<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials, Chart<X> *chart, BackPointers *back); \
  template Hyperpath *count_constrained_viterbi<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials, const HypergraphPotentials<CountingPotential> &count_potentials, int limit);

#define SPECIALIZE_FOR_SEMI_MIN(X)\
  template class Chart<X>;\
  template Chart<X> *general_inside<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials);\
  template Chart<X> *general_outside<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials, const Chart<X> &);

using namespace std;

template<class Set1, class Set2>
bool is_disjoint(const Set1 &set1, const Set2 &set2)
{
    if(set1.empty() || set2.empty()) return true;

    typename Set1::const_iterator
            it1 = set1.begin(),
            it1End = set1.end();
    typename Set2::const_iterator
            it2 = set2.begin(),
            it2End = set2.end();

    if(*it1 > *set2.rbegin() || *it2 > *set1.rbegin()) return true;

    while(it1 != it1End && it2 != it2End)
    {
        if(*it1 == *it2) return false;
        if(*it1 < *it2) { it1++; }
        else { it2++; }
    }

    return true;
}




// General code.

template<typename S>
Chart<S> *
general_inside(const Hypergraph *graph,
               const HypergraphPotentials<S> &potentials) {
  potentials.check(*graph);


  // Run Viterbi Hypergraph algorithm.
  Chart<S> *chart = new Chart<S>(graph);
  chart->initialize_inside();

  foreach (HEdge edge, graph->edges()) {
    typename S::ValType score =
            chart->compute_edge_score(edge, potentials.score(edge));
    chart->insert(edge->head_node(),
                  S::add((*chart)[edge->head_node()], score));
  }
  chart->insert(graph->root(),
                S::times((*chart)[graph->root()], potentials.bias()));
  return chart;
}

template<typename S>
Chart<S> *
general_outside(const Hypergraph *graph,
                const HypergraphPotentials<S> &potentials,
                const Chart<S> &inside_chart) {
  potentials.check(*graph);
  inside_chart.check(graph);
  Chart<S> *chart = new Chart<S>(graph);
  const vector<HEdge> &edges = graph->edges();
  chart->insert(graph->root(), potentials.bias());

  for (int i = edges.size() - 1; i >= 0; --i) {
    HEdge edge = edges[i];
    typename S::ValType head_score = (*chart)[edge->head_node()];
    // if (edge->head_node()->id() == graph->root()->id()) {
    //     head_score = potentials.bias();
    // }
    foreach (HNode node, edge->tail_nodes()) {
      typename S::ValType other_score = S::one();
      foreach (HNode other_node, edge->tail_nodes()) {
        if (other_node->id() == node->id()) continue;
        other_score = S::times(other_score, inside_chart[other_node]);
      }
      chart->insert(node, S::add((*chart)[node],
                                 S::times(head_score,
                                          S::times(other_score,
                                                   potentials.score(edge)))));
    }
  }
  return chart;
}

template<typename S>
void general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<S> &potentials,
    Chart<S> *chart,
    BackPointers *back) {

    potentials.check(*graph);
    chart->check(graph);
    back->check(graph);
    chart->clear();

    chart->initialize_inside();
    foreach (HNode node, graph->nodes()) {
        typename S::ValType best = (*chart)[node];
        foreach (HEdge edge, node->edges()) {
            typename S::ValType score =
                    chart->compute_edge_score(edge,
                                              potentials.score(edge));
            if (score > best) {
                chart->insert(node, score);
                back->insert(node, edge);
                best = score;
            }
        }
    }
}

template<typename S>
void DynamicViterbi<S>::update(
    const HypergraphPotentials<S> &potentials,
    set<int> *updated) {

    potentials.check(*graph_);
    chart_ = new Chart<S>(graph_);
    bp_ = new BackPointers(graph_);
    chart_->initialize_inside();
    //int count = 0;
    foreach (HNode node, graph_->nodes()) {
        // If the node does not need to be updated,
        // use last weights.
        if (is_disjoint<set<int>, set<int> >(
                (*children_sets_)[node->id()],
                *updated)) {
            //count += 1;
            chart_->insert(node, (*last_chart_)[node]);
            bp_->insert(node, (*last_bp_)[node]);
            continue;
        }


        typename S::ValType best = (*chart_)[node];
        foreach (HEdge edge, node->edges()) {
            typename S::ValType score =
                    chart_->compute_edge_score(edge,
                                               potentials.score(edge));
            if (score > best) {
                chart_->insert(node, score);
                bp_->insert(node, edge);
                best = score;
            }
        }
        if ((*bp_)[node] != (*last_bp_)[node] ||
            (*chart_)[node] != (*last_chart_)[node]) {
            updated->insert(node->id());
        }
    }
    //cout << count << endl;

    update_pointers();
}


Hyperpath *BackPointers::construct_path() const {
    // Collect backpointers.
    vector<HEdge> path;
    queue<HNode> to_examine;
    to_examine.push(graph_->root());
    while (!to_examine.empty()) {
        HNode node = to_examine.front();
        HEdge edge = chart_[node->id()];
        to_examine.pop();
        if (edge == NULL) {
            assert(node->terminal());
            continue;
        }
        path.push_back(edge);
        foreach (HNode node, edge->tail_nodes()) {
            to_examine.push(node);
        }
    }
    sort(path.begin(), path.end(), IdComparator());
    return new Hyperpath(graph_, path);
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


template<typename S>
Hyperpath *count_constrained_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<S> &weight_potentials,
    const HypergraphPotentials<CountingPotential> &count_potentials,
    int limit) {

  weight_potentials.check(*graph);
  count_potentials.check(*graph);

  vector<vector<NodeScore<S> > > chart(graph->nodes().size());

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
        chart[node->id()].push_back(
            NodeScore<S>(0, NULL, S::one()));
    }
    // Bucket edges.
    vector<NodeScore<S> > counts(limit + 1);
    foreach (HEdge edge, node->edges()) {
        bool unary = edge->tail_nodes().size() == 1;
        HNode left_node = edge->tail_nodes()[0];

        int start_count = count_potentials.score(edge);
        typename S::ValType start_score = weight_potentials.score(edge);
        for (int i = 0; i < chart[left_node->id()].size(); ++i) {
            int total = start_count + chart[left_node->id()][i].count;
            typename S::ValType total_score =
                    S::times(start_score,
                                   chart[left_node->id()][i].score);
            if (total > limit) continue;
            if (unary) {
                if (total_score > counts[total].score) {
                    counts[total] =
                            NodeScore<S>(total, edge, i, -1, total_score);
                }
            } else {
                HNode right_node = edge->tail_nodes()[1];
                for (int j = 0; j < chart[right_node->id()].size(); ++j) {
                    int total = start_count + chart[left_node->id()][i].count
                            + chart[right_node->id()][j].count;
                    typename S::ValType final_score =
                            S::times(total_score,
                                           chart[right_node->id()][j].score);

                    if (total > limit) continue;
                    if (final_score > counts[total].score) {
                        counts[total] =
                                NodeScore<S>(total,
                                             edge,
                                             i,
                                             j,
                                             final_score);
                    }
                }
            }
        }
    }

    // Compute scores.
    for (int count = 0; count <= limit; ++count) {
        if (counts[count].edge == NULL) continue;
        chart[node->id()].push_back(counts[count]);
    }
  }

  // Collect backpointers.
  vector<HEdge> path;
  queue<pair<HNode, int> > to_examine;
  int result = -1;
  int i = -1;
  foreach (NodeScore<S> score, chart[graph->root()->id()]) {
      ++i;
      if (score.count == limit) {
          result = i;
      }
  }

  to_examine.push(pair<HNode, int>(graph->root(), result));
  while (!to_examine.empty()) {
      if (result == -1) break;
      pair<HNode, int> p = to_examine.front();
      HNode node = p.first;
      int position = p.second;

      NodeScore<S> &score = chart[node->id()][position];
      HEdge edge = score.edge;

      to_examine.pop();
      if (edge == NULL) {
          assert(node->terminal());
          continue;
      }
      path.push_back(edge);
      for (int i = 0; i < edge->tail_nodes().size(); ++i) {
          HNode node = edge->tail_nodes()[i];
          to_examine.push(pair<HNode, int>(node,
                                           score.back[i]));
      }
  }
  sort(path.begin(), path.end(), IdComparator());
  return new Hyperpath(graph, path);
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

struct NodeCount {
    NodeCount(int _count, HNode _node) {
        count = _count;
        node = _node;
    }
    int count;
    HNode node;
};


HypergraphProjection *extend_hypergraph_by_count(
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
                          new_graph->add_terminal_node(node->label())));
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
                        counts[total - lower_limit].push_back(EdgeGroup(edge, i, 0));
                    } else {
                        HNode right_node = edge->tail_nodes()[1];
                        int right_size = new_nodes[right_node->id()].size();
                        for (int j = 0; j < right_size; ++j) {
                            int total =
                                    score + new_nodes[left_node->id()][i].count
                                    + new_nodes[right_node->id()][j].count;
                            if (total < lower_limit ||
                                total > upper_limit) continue;
                            counts[total - lower_limit].push_back(EdgeGroup(edge, i, j));
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
                              new_graph->start_node(node->label())));

                foreach (EdgeGroup edge_group, counts[count]) {
                    HEdge edge = edge_group.edge;
                    vector<HNode> tails;
                    for (int i = 0; i < edge->tail_nodes().size(); ++i) {
                        tails.push_back(new_nodes[edge->tail_nodes()[i]->id()]
                                        [edge_group.back[i]].node);
                    }
                    HEdge new_edge = new_graph->add_edge(tails, edge->label());
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
    return new HypergraphProjection(new_graph, hypergraph,
                                    node_map, edge_map, false);
}

Chart<SetPotential> *edge_domination(const Hypergraph &graph) {
    vector<set<int> > v(graph.edges().size());
    foreach (HEdge edge, graph.edges()) {
        v[edge->id()].insert(edge->id());
    }

    HypergraphVectorPotentials<SetPotential> potentials(
        &graph, v, set<int>());
    Chart<SetPotential> *chart = general_inside(&graph, potentials);
    return chart;
}

Chart<SetPotential> *node_domination(const Hypergraph &graph) {
    vector<set<int> > v(graph.edges().size());
    foreach (HEdge edge, graph.edges()) {
        foreach (HNode node, edge->tail_nodes()) {
            v[edge->id()].insert(node->id());
        }
    }

    HypergraphVectorPotentials<SetPotential> potentials(
        &graph, v, set<int>());
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


// vector<bool> active_nodes(const Hypergraph *graph, set<int> on_edges) {
//     Chart<SetPotential> *edges = edge_domination;

// }

SPECIALIZE_ALGORITHMS_FOR_SEMI(ViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(LogViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(InsidePotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(BoolPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(CountingPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(SetPotential)
SPECIALIZE_FOR_SEMI_MIN(SparseVectorPotential)
SPECIALIZE_FOR_SEMI_MIN(MinSparseVectorPotential)
SPECIALIZE_FOR_SEMI_MIN(MaxSparseVectorPotential)
SPECIALIZE_FOR_SEMI_MIN(BinaryVectorPotential)

template class DynamicViterbi<LogViterbiPotential>;

// End General code.
