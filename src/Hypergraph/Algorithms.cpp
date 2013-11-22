// Copyright [2013] Alexander Rush


#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>

#include "Hypergraph/Algorithms.h"
//#include "Hypergraph/Subgradient.h"

#define SPECIALIZE_ALGORITHMS_FOR_SEMI(X)\
  template class Chart<X>;\
  template class HypergraphPotentials<X>;\
  template class Marginals<X>;\
  template Hyperpath *general_viterbi<X>(const Hypergraph *graph,const HypergraphPotentials<X> &potentials);

#define SPECIALIZE_FOR_SEMI_MIN(X)\
  template class Chart<X>;\
  template class HypergraphPotentials<X>;\
  template Chart<X> *general_inside<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials);\
  template Chart<X> *general_outside<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials, const Chart<X> &);

using namespace std;

// General code.

struct IdComparator {
  bool operator()(HEdge edge1, HEdge edge2) const {
    return edge1->id() < edge2->id();
  }
};

template<typename S>
Chart<S> *
general_inside(const Hypergraph *graph,
               const HypergraphPotentials<S> &potentials) {
  potentials.check(*graph);

  // Run Viterbi Hypergraph algorithm.
  Chart<S> *chart = new Chart<S>(graph);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      chart->insert(node, S::one());
    }
  }
  foreach (HEdge edge, graph->edges()) {
    typename S::ValType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      score = S::times(score, (*chart)[node]);
    }
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
    foreach (HNode node, edge->tail_nodes()) {
      typename S::ValType other_score = S::one();
      foreach (HNode other_node, edge->tail_nodes()) {
        if (other_node->id() == node->id()) continue;
        other_score = S::times(other_score, inside_chart[other_node]);
      }
      chart->insert(node, S::add((*chart)[node], S::times(head_score, S::times(other_score, potentials.score(edge)))));
    }
  }
  return chart;
}

template<typename S>
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<S> &potentials) {

  potentials.check(*graph);
  Chart<S> *chart = new Chart<S>(graph);
  vector<HEdge> back(graph->nodes().size(), NULL);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      chart->insert(node, S::one());
    }
  }
  foreach (HEdge edge, graph->edges()) {
    typename S::ValType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      score *= (*chart)[node];
    }
    if (score > (*chart)[edge->head_node()]) {
      chart->insert(edge->head_node(), score);
      back[edge->head_node()->id()] = edge;
    }
  }

  // Collect backpointers.
  vector<HEdge> path;
  queue<HNode> to_examine;
  to_examine.push(graph->root());
  while (!to_examine.empty()) {
    HNode node = to_examine.front();
    HEdge edge = back[node->id()];
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
  delete chart;
  return new Hyperpath(graph, path);
}

template<typename StatSem>
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<typename StatSem::ValType> &potentials) {

  potentials.check(*graph);
  Chart<typename StatSem::ValType> *chart = new Chart<typename StatSem::ValType>(graph);
  vector<HEdge> back(graph->nodes().size(), NULL);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      chart->insert(node, StatSem::one());
    }
  }
  foreach (HEdge edge, graph->edges()) {
    typename StatSem::ValType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      typename StatSem::ValType(score, (*chart)[node]);
    }
    if (score > (*chart)[edge->head_node()]) {
      chart->insert(edge->head_node(), score);
      back[edge->head_node()->id()] = edge;
    }
  }

  // Collect backpointers.
  vector<HEdge> path;
  queue<HNode> to_examine;
  to_examine.push(graph->root());
  while (!to_examine.empty()) {
    HNode node = to_examine.front();
    HEdge edge = back[node->id()];
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
  delete chart;
  return new Hyperpath(graph, path);
}


SPECIALIZE_ALGORITHMS_FOR_SEMI(ViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(LogViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(InsidePotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(BoolPotential)
SPECIALIZE_FOR_SEMI_MIN(SparseVectorPotential)

// End General code.
