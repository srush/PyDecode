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

#define SPECIALIZE_ALGORITHMS_FOR_SEMI(X)       \
  template class Chart<X>;\
  template class Marginals<X>;\
  template void general_viterbi<X>(const Hypergraph *graph, \
    const HypergraphPotentials<X> &potentials, \
    Chart<X> *chart, BackPointers *back);

#define SPECIALIZE_FOR_SEMI_MIN(X)\
  template class Chart<X>;\
  template Chart<X> *general_inside<X>(const Hypergraph *graph, \
               const HypergraphPotentials<X> &potentials); \
  template Chart<X> *general_outside<X>(const Hypergraph *graph, \
               const HypergraphPotentials<X> &potentials, \
               const Chart<X> &);

using namespace std;



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

// End General code.
