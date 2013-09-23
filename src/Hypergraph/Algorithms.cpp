// Copyright [2013] Alexander Rush

#include "Hypergraph/Algorithms.h"

#include <queue>
#include <vector>

#include "Hypergraph/Constraints.h"
#include "Hypergraph/Subgradient.h"

#include "./common.h"
using namespace std;

Hyperpath *viterbi_path(const Hypergraph *graph,
                        const HypergraphWeights &theta,
                        vector<double> *chart) {

  // Run Viterbi Hypergraph algorithm.
  chart->clear();
  chart->resize(graph->nodes().size(), -INF);
  vector<HEdge> back(graph->nodes().size(), NULL);
  foreach (HEdge edge, graph->edges()) {
    double score = theta.score(edge);
    int head_id = edge->head_node()->id();
    foreach (HNode node, edge->tail_nodes()) {
      score += (*chart)[node->id()];
    }
    if (score > (*chart)[head_id]) {
      (*chart)[head_id] = score;
      back[head_id] = edge;
    }
  }

  // Collect backpointers.
  vector<HEdge> path;
  queue<HNode> to_examine;
  to_examine.push(graph->root());
  while(!to_examine.empty()) {
    HEdge edge = back[to_examine.front()->id()];
    to_examine.pop();
    if (edge == NULL) continue;
    path.push_back(edge);
    foreach (HNode node, edge->tail_nodes()) {
      to_examine.push(node);

    }
  }
  return new Hyperpath(graph, path);
}

void outside(const Hypergraph *graph,
             const HypergraphWeights &weights,
             const vector<double> &inside_chart,
             vector<double> *chart) {
  chart->resize(graph->nodes().size(), -INF);
  const vector<HEdge> &edges = graph->edges();

  (*chart)[graph->root()->id()] = 0;
  for (int i = edges.size() - 1; i >= 0; --i) {
    HEdge edge = edges[i];
    double full_score = weights.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      full_score += inside_chart[node->id()];
    }
    double head_score = (*chart)[edge->head_node()->id()];
    foreach (HNode node, edge->tail_nodes()) {
      double score = head_score + full_score - inside_chart[node->id()];
      if (score > (*chart)[node->id()]){
        (*chart)[node->id()] = score;
      }
    }
  }

  // Add in the bias.
  double bias = weights.bias();
  for (int i = 0; i < chart->size(); ++i) {
    (*chart)[i] += bias;
  }
}


class ConstrainedProducer : public SubgradientProducer {
 public:
  ConstrainedProducer(
      const Hypergraph *graph,
      const HypergraphWeights *weights,
      const HypergraphConstraints *constraints)
      : graph_(graph), weights_(weights), constraints_(constraints)
  {}

  void solve(const SubgradState &cur_state,
             SubgradResult *result) const {
    SparseVec edge_duals;
    double bias_dual;
    constraints_->convert(*cur_state.duals, &edge_duals, &bias_dual);
    HypergraphWeights *dual_weights =
        weights_->modify(edge_duals, bias_dual);
    vector<double> chart;
    Hyperpath *path = viterbi_path(graph_, *dual_weights, &chart);
    result->dual = dual_weights->dot(*path);
    result->subgrad = constraints_->subgradient(*path);
    delete dual_weights, path;
  }

 private:
  const Hypergraph *graph_;
  const HypergraphWeights *weights_;
  const HypergraphConstraints *constraints_;

};

Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights &theta,
    const HypergraphConstraints &constraints) {
ConstrainedProducer producer(graph, &theta, &constraints);
  Subgradient subgradient(&producer, NULL);
}
