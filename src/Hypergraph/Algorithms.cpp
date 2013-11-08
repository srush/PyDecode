// Copyright [2013] Alexander Rush


#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>

#include "Hypergraph/Algorithms.h"
//#include "Hypergraph/Subgradient.h"

#define SPECIALIZE_FOR_SEMI(X)\
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

template<typename SemiringType>
Chart<SemiringType> *
general_inside(const Hypergraph *graph,
               const HypergraphPotentials<SemiringType> &potentials) {
  potentials.check(*graph);

  // Run Viterbi Hypergraph algorithm.
  Chart<SemiringType> *chart = new Chart<SemiringType>(graph);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      (*chart)[node] = SemiringType::one();
    }
  }
  foreach (HEdge edge, graph->edges()) {
    SemiringType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      score *= (*chart)[node];
    }
    (*chart)[edge->head_node()] += score;
  }
  (*chart)[graph->root()] *= potentials.bias();
  return chart;
}

template<typename SemiringType>
Chart<SemiringType> *
general_outside(const Hypergraph *graph,
                const HypergraphPotentials<SemiringType> &potentials,
                const Chart<SemiringType> &inside_chart) {
  potentials.check(*graph);
  inside_chart.check(graph);
  Chart<SemiringType> *chart = new Chart<SemiringType>(graph);
  const vector<HEdge> &edges = graph->edges();
  (*chart)[graph->root()] = potentials.bias();

  for (int i = edges.size() - 1; i >= 0; --i) {
    HEdge edge = edges[i];
    SemiringType head_score = (*chart)[edge->head_node()];
    foreach (HNode node, edge->tail_nodes()) {
      SemiringType other_score = SemiringType::one();
      foreach (HNode other_node, edge->tail_nodes()) {
        if (other_node->id() == node->id()) continue;
        other_score *= inside_chart[other_node];
      }
      (*chart)[node] += head_score * other_score * potentials.score(edge);
    }
  }
  return chart;
}

template<typename SemiringType>
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<SemiringType> &potentials) {

  potentials.check(*graph);
  Chart<SemiringType> *chart = new Chart<SemiringType>(graph);
  vector<HEdge> back(graph->nodes().size(), NULL);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      (*chart)[node] = SemiringType::one();
    }
  }
  foreach (HEdge edge, graph->edges()) {
    SemiringType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      score *= (*chart)[node];
    }
    if (score > (*chart)[edge->head_node()]) {
      (*chart)[edge->head_node()] = score;
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


SPECIALIZE_FOR_SEMI(ViterbiPotential)
SPECIALIZE_FOR_SEMI(LogViterbiPotential)
SPECIALIZE_FOR_SEMI(InsidePotential)
SPECIALIZE_FOR_SEMI(BoolPotential)
SPECIALIZE_FOR_SEMI_MIN(SparseVectorPotential)

// End General code.
