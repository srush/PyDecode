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

#define SPECIALIZE_ALGORITHMS_FOR_SEMI(X)       \
    SPECIALIZE_FOR_SEMI_MIN(X);                 \
  template void general_viterbi<X>(const Hypergraph *graph, \
    const HypergraphPotentials<X> &potentials, \
    Chart<X> *chart, BackPointers *back, bool *mask);

#define SPECIALIZE_FOR_SEMI_MIN(X)\
  template class Chart<X>;\
  template void general_inside<X>(const Hypergraph *graph, \
                                  const HypergraphPotentials<X> &potentials, \
                                  Chart<X> *chart);                     \
  template void general_outside<X>(const Hypergraph *graph, \
                                   const HypergraphPotentials<X> &potentials, \
                                   const Chart<X> &,                    \
                                   Chart<X> *chart);                    \
  template void node_marginals(const Hypergraph *hypergraph, \
                               const Chart<X> &in_chart,     \
                               const Chart<X> &out_chart,   \
                               Chart<X> *node_marginals);   \
  template void edge_marginals(const Hypergraph *hypergraph,            \
                               const HypergraphPotentials<X> &potentials, \
                               const Chart<X> &in_chart,                \
                               const Chart<X> &out_chart,               \
                               typename X::ValType *edge_marginals);    \

using namespace std;


// General code.
template<typename S>
void general_inside(const Hypergraph *graph,
                    const HypergraphPotentials<S> &potentials,
                    Chart<S> *chart) {
    potentials.check(*graph);
    bool unary = graph->is_unary();

    // Run Viterbi Hypergraph algorithm.
    chart->clear();
    chart->initialize_inside();
    typename S::ValType *inner_chart = chart->chart();
    const typename S::ValType *potential = potentials.potentials();
    int edge = 0;
    foreach (HNode node, graph->nodes()) {
        if (graph->terminal(node)) continue;
        typename S::ValType cur = inner_chart[node];
        int end = graph->edge_end(node);
        if (unary) {
            for (; edge <= end; ++edge) {
                typename S::ValType score =
                        score = S::times(potential[edge],
                                         inner_chart[graph->tail_node(edge)]);
                cur = S::add(cur, score);
            }
        } else {
            for (; edge <= end; ++edge) {
                typename S::ValType score =
                        chart->compute_edge_score(edge,
                                                  potential[edge]);
                cur = S::add(cur, score);
            }
        }
        inner_chart[node] = cur;
    }
}

template<typename S>
void general_outside(const Hypergraph *graph,
                     const HypergraphPotentials<S> &potentials,
                     const Chart<S> &inside_chart,
                     Chart<S> *chart) {
    potentials.check(*graph);
    inside_chart.check(graph);
    chart->clear();

    const vector<HEdge> &edges = graph->edges();
    chart->insert(graph->root(), S::one());

    for (int i = edges.size() - 1; i >= 0; --i) {
        HEdge edge = edges[i];
        typename S::ValType head_score = (*chart)[graph->head(edge)];
        for (int j = 0; j < graph->tail_nodes(edge); ++j) {
            HNode node = graph->tail_node(edge, j);
            typename S::ValType other_score = S::one();
            for (int k = 0; k < graph->tail_nodes(edge); ++k) {
                HNode other_node = graph->tail_node(edge, k);
                if (other_node == node) continue;
                other_score = S::times(other_score, inside_chart[other_node]);
            }
            chart->insert(node, S::add((*chart)[node],
                                       S::times(head_score,
                                                S::times(other_score,
                                                         potentials.score(edge)))));
        }
    }
}

template<typename S>
void general_viterbi(const Hypergraph *graph,
                     const HypergraphPotentials<S> &potentials,
                     Chart<S> *chart,
                     BackPointers *back,
                     bool *mask) {
    potentials.check(*graph);
    chart->check(graph);
    back->check(graph);
    chart->clear();

    chart->initialize_inside();
    const bool unary = graph->is_unary();
    const bool use_mask = (mask != NULL);
    typename S::ValType *inner_chart = chart->chart();
    const typename S::ValType *potential = potentials.potentials();
    HEdge *back_chart = back->chart();
    int edge = 0;
    foreach (HNode node, graph->nodes()) {
        if (graph->terminal(node)) continue;
        if (use_mask && !mask[node]) continue;
        typename S::ValType best = inner_chart[node];
        int end = graph->edge_end(node);
        // if (use_mask && !mask[graph->tail_node(edge)]) continue;
        if (unary) {
            for (; edge <= end; ++edge) {
                typename S::ValType score = S::times(
                    potential[edge],
                    inner_chart[graph->tail_node(edge)]);

                if (score > best) {
                    inner_chart[node] = score;
                    back_chart[node] = edge;
                    best = score;
                }
            }
        } else {
            for (; edge <= end; ++edge) {
                typename S::ValType score = potential[edge];
                bool fail = false;
                for (int j = 0; j < graph->tail_nodes(edge); ++j) {
                    HNode tail = graph->tail_node(edge, j);
                    if (use_mask && !mask[tail]) {
                        fail = true;
                        break;
                    }
                    score = S::times(score,
                                     inner_chart[tail]);
                }
                if (use_mask && fail) continue;
                if (score > best) {
                    inner_chart[node] = score;
                    back_chart[node] = edge;
                    best = score;
                }
            }
        }
        if (use_mask && inner_chart[node] <= S::zero()) {
            mask[node] = false;
        }
    }
}

// template<>
// void general_viterbi(const Hypergraph *graph,
//                      const HypergraphPotentials<LogViterbiPotential> &potentials,
//                      Chart<LogViterbiPotential> *chart,
//                      BackPointers *back,
//                      bool *mask) {
//     potentials.check(*graph);
//     chart->check(graph);
//     back->check(graph);
//     chart->clear();
//     const double *potential = potentials.potentials();
//     HEdge *back_chart = back->chart();
//     chart->initialize_inside();
//     bool unary = graph->is_unary();
//     double zero = -1e9;
//     double *inner_chart = chart->chart();
//     int edge = 0;
//     foreach (HNode node, graph->nodes()) {
//         if (graph->terminal(node)) continue;
//         double best = inner_chart[node];
//         int end = graph->edge_end(node);
//         for (; edge <= end; ++edge) {
//             double score;
//             if (true) {
//                 score = potential[edge] +
//                         inner_chart[graph->tail_node(edge)];
//             } else {
//                 score = potential[edge];
//                 for (int j = 0; j < graph->tail_nodes(edge); ++j) {
//                     HNode tail = graph->tail_node(edge, j);
//                     score = score + inner_chart[tail];
//                 }
//             }

//             if (score > best) {
//                 inner_chart[node] = score;
//                 back_chart[node] = edge;
//                 best = score;
//             }
//         }
//     }
// }


// template<typename S>
// struct Hypothesis {
//     vector<int> vec;
//     HEdge edge;
//     V score;
// };



// template<typename S>
// void general_kbest(
//     const Hypergraph *graph,
//     const HypergraphPotentials<S> &potentials,
//     KBackPointers *back,
//     int K) {

//     potentials.check(*graph);
//     back->check(graph);

//     foreach (HNode node, graph->nodes()) {
//         for (int k = 0; k < K; ++k) {
//             vector<Hypothesis> edge_hyps(edges().size());
//             foreach (HEdge edge, node->edges()) {

//             }

//             int edge_num = 0;
//             foreach (HEdge edge, node->edges()) {
//                 vector<int> children(edge.tail->size(), 0);
//                 typename S::ValType score = potentials.score(edge);
//                 foreach(HNode node, edge->tail()) {
//                     score = S::times(score, chart_[tail->id()]);
//                 }
//                 Hypothesis hypothesis();

//                 edge_num++;
//             }
//             children[best_edge]++;
//         }

//     }
// }

Hyperpath *BackPointers::construct_path() const {
    // Collect backpointers.
    bool unary = graph_->is_unary();
    vector<HEdge> path;
    vector<HNode> node_path;
    if (unary) {
        HNode cur = graph_->root();
        node_path.push_back(cur);
        while (!graph_->terminal(cur)) {
            HEdge edge = chart_[cur];
            path.push_back(edge);
            cur = graph_->tail_node(edge);
            node_path.push_back(cur);
        }
        reverse(path.begin(), path.end());
        reverse(node_path.begin(), node_path.end());
    } else {
        queue<HNode> to_examine;
        to_examine.push(graph_->root());
        while (!to_examine.empty()) {
            HNode node = to_examine.front();
            node_path.push_back(node);
            HEdge edge = chart_[node];
            to_examine.pop();
            if (edge == -1) {
                assert(graph_->terminal(node));
                continue;
            }
            path.push_back(edge);
            for (int i = 0; i < graph_->tail_nodes(edge); ++i) {
                to_examine.push(graph_->tail_node(edge, i));
            }
        }
        sort(path.begin(), path.end(), IdComparator());
        sort(node_path.begin(), node_path.end(), IdComparator());
    }
    return new Hyperpath(graph_, node_path, path);
}

template<typename StatSem>
struct NodeScore {
    NodeScore()
            :
            count(-1),
            edge(-1),
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
      if (graph->terminal(node)) {
          chart[node].push_back(
            NodeScore<S>(0, -1, S::one()));
    }
    // Bucket edges.
    vector<NodeScore<S> > counts(limit + 1);
    foreach (HEdge edge, graph->edges(node)) {
        bool unary = graph->tail_nodes(edge) == 1;
        HNode left_node = graph->tail_node(edge, 0);

        int start_count = count_potentials.score(edge);
        typename S::ValType start_score = weight_potentials.score(edge);
        for (int i = 0; i < chart[left_node].size(); ++i) {
            int total = start_count + chart[left_node][i].count;
            typename S::ValType total_score =
                    S::times(start_score,
                                   chart[left_node][i].score);
            if (total > limit) continue;
            if (unary) {
                if (total_score > counts[total].score) {
                    counts[total] =
                            NodeScore<S>(total, edge, i, -1, total_score);
                }
            } else {
                HNode right_node = graph->tail_node(edge, 1);
                for (int j = 0; j < chart[right_node].size(); ++j) {
                    int total = start_count + chart[left_node][i].count
                            + chart[right_node][j].count;
                    typename S::ValType final_score =
                            S::times(total_score,
                                           chart[right_node][j].score);

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
        if (counts[count].edge == -1) continue;
        chart[node].push_back(counts[count]);
    }
  }

  // Collect backpointers.
  vector<HEdge> path;
  vector<HNode> node_path;
  queue<pair<HNode, int> > to_examine;
  int result = -1;
  int i = -1;
  foreach (NodeScore<S> score, chart[graph->root()]) {
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
      node_path.push_back(node);
      int position = p.second;

      NodeScore<S> &score = chart[node][position];
      HEdge edge = score.edge;

      to_examine.pop();
      if (edge == -1) {
          assert(graph->terminal(node));
          continue;
      }
      path.push_back(edge);
      for (int i = 0; i < graph->tail_nodes(edge); ++i) {
          HNode node = graph->tail_node(edge, i);
          to_examine.push(pair<HNode, int>(node,
                                           score.back[i]));
      }
  }
  sort(path.begin(), path.end());
  sort(node_path.begin(), node_path.end());
  return new Hyperpath(graph, node_path, path);
}


SPECIALIZE_ALGORITHMS_FOR_SEMI(ViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(LogViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(InsidePotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(BoolPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(CountingPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(SetPotential)
SPECIALIZE_FOR_SEMI_MIN(LogProbPotential)
SPECIALIZE_FOR_SEMI_MIN(SparseVectorPotential)
// SPECIALIZE_FOR_SEMI_MIN(MinSparseVectorPotential)
// SPECIALIZE_FOR_SEMI_MIN(MaxSparseVectorPotential)
SPECIALIZE_FOR_SEMI_MIN(MinMaxPotential)


template Hyperpath *count_constrained_viterbi<LogViterbiPotential>(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &weight_potentials,
    const HypergraphPotentials<CountingPotential> &count_potentials,
    int limit);


// End General code.
