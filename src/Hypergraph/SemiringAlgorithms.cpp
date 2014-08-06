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
#include "Hypergraph/SemiringAlgorithms.hh"

#define SPECIALIZE_ALGORITHMS_FOR_SEMI(X)       \
    SPECIALIZE_FOR_SEMI_MIN(X);                 \
  template void general_viterbi<X>(const Hypergraph *graph, \
             const typename X::ValType *weights,      \
             typename X::ValType *chart, int *back, bool *mask); \
  template void general_kbest<X>(    \
          const Hypergraph *graph, \
          const typename X::ValType *weights,   \
          int K,  \
          vector<Hyperpath *> *paths);


#define SPECIALIZE_FOR_SEMI_MIN(X)\
  template void general_inside<X>(const Hypergraph *graph, \
                                  const typename X::ValType *weights, \
                                  typename X::ValType *chart);         \
  template void general_outside<X>(const Hypergraph *graph, \
                                   const typename X::ValType *weights, \
                                   const typename X::ValType *,        \
                                   typename X::ValType *chart);        \
  template void node_marginals<X>(const Hypergraph *hypergraph,         \
                               const typename X::ValType *in_chart,     \
                               const typename X::ValType *out_chart,   \
                               typename X::ValType *node_marginals);   \
  template void edge_marginals<X>(const Hypergraph *hypergraph,        \
                               const typename X::ValType *weights, \
                               const typename X::ValType *in_chart,  \
                               const typename X::ValType *out_chart,  \
                               typename X::ValType *edge_marginals);  \
  template void transform<X>(const Hypergraph *hypergraph,        \
                             const typename X::ValType *weights,     \
                             const int *labeling,                  \
                             typename X::ValType *label_weights, int);   \

using namespace std;

// General code.
template<typename S>
void general_inside(const Hypergraph *graph,
                    const typename S::ValType *weights,
                    typename S::ValType *chart) {
    bool unary = graph->is_unary();
    fill(chart, chart + graph->nodes().size(), S::zero());

    // Run Viterbi Hypergraph algorithm.
    int edge = 0;
    foreach (HNode node, graph->nodes()) {
        if (graph->terminal(node)) {
            chart[node] = S::one();
            continue;
        }

        typename S::ValType cur = chart[node];
        int end = graph->edge_end(node);
        if (unary) {
            for (; edge <= end; ++edge) {
                typename S::ValType score =
                        S::times(weights[edge],
                                 chart[graph->tail_node(edge)]);
                cur = S::add(cur, score);
            }
        } else {
            for (; edge <= end; ++edge) {
                typename S::ValType score = weights[edge];
                for (int j = 0; j < graph->tail_nodes(edge); ++j) {
                    HNode tail = graph->tail_node(edge, j);
                    score = S::times(score, chart[tail]);
                }
                cur = S::add(cur, score);
            }
        }
        chart[node] = cur;
    }
}

template<typename S>
void general_outside(const Hypergraph *graph,
                     const typename S::ValType * weights,
                     const typename S::ValType *inside_chart,
                     typename S::ValType *chart) {
    fill(chart, chart + graph->nodes().size(), S::zero());
    const vector<HEdge> &edges = graph->edges();
    chart[graph->root()] = S::one();

    for (int i = edges.size() - 1; i >= 0; --i) {
        HEdge edge = edges[i];
        typename S::ValType head_score = chart[graph->head(edge)];
        for (int j = 0; j < graph->tail_nodes(edge); ++j) {
            HNode node = graph->tail_node(edge, j);
            typename S::ValType other_score = S::one();
            for (int k = 0; k < graph->tail_nodes(edge); ++k) {
                HNode other_node = graph->tail_node(edge, k);
                if (other_node == node) continue;
                other_score = S::times(other_score, inside_chart[other_node]);
            }
            chart[node] = S::add(chart[node],
                                 S::times(head_score,
                                          S::times(other_score,
                                                   weights[edge])));
        }
    }
}

template<typename S>
void general_viterbi(const Hypergraph *graph,
                     const typename S::ValType *weights,
                     typename S::ValType *chart,
                     int *back_chart,
                     bool *mask) {
    const bool unary = graph->is_unary();
    const bool use_mask = (mask != NULL);

    fill(back_chart, back_chart + graph->nodes().size(), -1);
    fill(chart, chart + graph->nodes().size(), S::zero());

    int edge = 0;
    foreach (HNode node, graph->nodes()) {
        if (graph->terminal(node)) {
            chart[node] = S::one();
            continue;
        }
        if (use_mask && !mask[node]) continue;
        typename S::ValType best = chart[node];
        int end = graph->edge_end(node);
        // if (use_mask && !mask[graph->tail_node(edge)]) continue;
        if (unary) {
            for (; edge <= end; ++edge) {
                typename S::ValType score = S::times(
                    weights[edge],
                    chart[graph->tail_node(edge)]);

                if (score > best) {
                    chart[node] = score;
                    back_chart[node] = edge;
                    best = score;
                }
            }
        } else {
            for (; edge <= end; ++edge) {
                typename S::ValType score = weights[edge];
                bool fail = false;
                for (int j = 0; j < graph->tail_nodes(edge); ++j) {
                    HNode tail = graph->tail_node(edge, j);
                    if (use_mask && !mask[tail]) {
                        fail = true;
                        break;
                    }
                    score = S::times(score,
                                     chart[tail]);
                }
                if (use_mask && fail) continue;
                if (score > best) {
                    chart[node] = score;
                    back_chart[node] = edge;
                    best = score;
                }
            }
        }
        if (use_mask && chart[node] <= S::zero()) {
            mask[node] = false;
        }
    }
}


template<typename S>
void node_marginals(const Hypergraph *hypergraph,
                    const typename S::ValType *in_chart,
                    const typename S::ValType *out_chart,
                    typename S::ValType *node_marginals) {
    foreach (HNode node, hypergraph->nodes()) {
        node_marginals[node] = \
                S::times(in_chart[node], out_chart[node]);
    }
}

template<typename S>
void edge_marginals(const Hypergraph *hypergraph,
                    const typename S::ValType *weights,
                    const typename S::ValType *in_chart,
                    const typename S::ValType *out_chart,
                    typename S::ValType *edge_marginals) {
    foreach (HEdge edge, hypergraph->edges()) {
        typename S::ValType score = out_chart[hypergraph->head(edge)];

        score = S::times(score, weights[edge]);
        for (int j = 0; j < hypergraph->tail_nodes(edge); ++j) {
            HNode node = hypergraph->tail_node(edge, j);
            score = S::times(score, in_chart[node]);
        }
        edge_marginals[edge] = score;
    }
}

template<typename S>
void transform(const Hypergraph *hypergraph,
               const typename S::ValType *weights,
               const int *labeling,
               typename S::ValType *label_weights,
               int label_size) {
    fill(label_weights, label_weights + label_size, S::zero());
    foreach (HEdge edge, hypergraph->edges()) {
        int label = labeling[edge];
        if (label != -1) {
            label_weights[label] = S::add(label_weights[label],
                                          weights[edge]);
        }
    }
}

template<typename SemiringType>
struct KBestHypothesis {
    KBestHypothesis(typename SemiringType::ValType score_,
                    HEdge last_edge_,
                    vector<int> last_best_)
            : score(score_), last_edge(last_edge_), last_best(last_best_) {}

    KBestHypothesis(HEdge last_edge_, int size)
            : last_edge(last_edge_), last_best(size, 0) {}


    typename SemiringType::ValType score;
    HEdge last_edge;
    vector<int> last_best;

    double rescore(const Hypergraph *graph,
                 const typename SemiringType::ValType *weights,
                 const vector<vector<KBestHypothesis<SemiringType> > > &chart) {
        score = weights[last_edge];
        for (int i = 0; i < graph->tail_nodes(last_edge); ++i) {
            HNode tail_node = graph->tail_node(last_edge, i);
            score = SemiringType::times(score,
                                        chart[tail_node][last_best[i]].score);
        }
    }

    KBestHypothesis<SemiringType> advance(int i) const {
        vector<int> new_best = last_best;
        new_best[i] += 1;
        return KBestHypothesis(score, last_edge, new_best);
    }

    bool operator<(const KBestHypothesis<SemiringType> &hyp) const{
        return score > hyp.score;
    }
};



template<typename SemiringType>
void general_kbest(
    const Hypergraph *graph,
    const typename SemiringType::ValType *weights,
    int K,
    vector<Hyperpath *> *paths) {
    typedef KBestHypothesis<SemiringType> KHyp;
    vector<vector<KHyp> > chart(graph->nodes().size());

    foreach (HNode node, graph->nodes()) {
        if (graph->terminal(node)) {
            KHyp hyp(SemiringType::one(),
                     EDGE_NULL);
            chart[node].push_back(hyp);
            continue;
        }

        // Initialize priority queue.
        priority_queue<KHyp> queue;
        foreach (HEdge edge, graph->edges(node)) {
            KHyp hypothesis(edge, graph->tail_nodes(edge));
            hypothesis.rescore(graph, weights, chart);
            queue.push(hypothesis);
        }


        // Pull up on k at a time.
        for (int k_round = 0; k_round < K; k_round++) {
            // Pull up.
            KHyp hyp = queue.top();
            queue.pop();
            chart[node].push_back(hyp);

            // Add new hypotheses.
            for (int i = 0; i < graph->tail_nodes(hyp.last_edge); ++i) {
                HNode node = graph->tail_node(hyp.last_edge, i);
                if (hyp.last_best[i] + 1 >= chart[node].size())
                    continue;
                KHyp new_hyp(hyp.advance(i));
                new_hyp.rescore(graph, weights, chart);
                queue.push(new_hyp);
            }
        }
    }

    // Find the k-best paths.
    HNode root = graph->root();
    for (int i = 0; i < chart[root].size(); ++i) {
        vector<HEdge> path;
        vector<HNode> node_path;
        queue<pair<HNode, int> > to_examine;
        to_examine.push(pair<HNode, int>(root, i));
        // if (result >= get_beam(hypergraph_->root()).size()) {
        //     return NULL;
        // }

        while (!to_examine.empty()) {
            pair<HNode, int> p = to_examine.front();
            HNode node = p.first;
            node_path.push_back(node);
            int position = p.second;

            const KHyp &hyp = chart[node][position];
            HEdge edge = hyp.last_edge;

            to_examine.pop();
            if (edge == EDGE_NULL) {
                assert(graph->terminal(node));
                continue;
            }
            path.push_back(edge);
            for (int i = 0; i < graph->tail_nodes(edge); ++i) {
                HNode node = graph->tail_node(edge, i);
                to_examine.push(pair<HNode, int>(node,
                                                 hyp.last_best[i]));

            }
        }
        sort(node_path.begin(), node_path.end(), IdComparator());
        sort(path.begin(), path.end(), IdComparator());
        paths->push_back(new Hyperpath(graph, node_path, path));
   }
}


SPECIALIZE_ALGORITHMS_FOR_SEMI(Viterbi)
SPECIALIZE_ALGORITHMS_FOR_SEMI(LogViterbi)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Real)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Log)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Boolean)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Counting)
SPECIALIZE_ALGORITHMS_FOR_SEMI(MinMax)

// End General code.
