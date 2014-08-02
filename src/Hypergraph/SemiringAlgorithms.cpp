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
             typename X::ValType *chart, int *back, bool *mask);

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

SPECIALIZE_ALGORITHMS_FOR_SEMI(Viterbi)
SPECIALIZE_ALGORITHMS_FOR_SEMI(LogViterbi)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Real)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Log)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Boolean)
SPECIALIZE_ALGORITHMS_FOR_SEMI(Counting)
SPECIALIZE_ALGORITHMS_FOR_SEMI(MinMax)

// End General code.
