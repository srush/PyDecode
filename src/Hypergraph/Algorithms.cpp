// Copyright [2013] Alexander Rush

#include "Hypergraph/Algorithms.h"

#include <queue>
#include <vector>
#include <assert.h>

#include "Hypergraph/Constraints.h"
#include "Hypergraph/Subgradient.h"

#include "./common.h"
using namespace std;

struct IdComparator {
  bool operator()(HEdge edge1, HEdge edge2) const {
    return edge1->id() < edge2->id();
  }
};

Hyperpath *viterbi_path(const Hypergraph *graph,
                        const HypergraphWeights &theta,
                        vector<double> *chart) {

  // Run Viterbi Hypergraph algorithm.
  chart->clear();
  chart->resize(graph->nodes().size(), -INF);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      (*chart)[node->id()] = 0;
    }
  }
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
    SparseVec edge_duals(10000);
    double bias_dual;
    constraints_->convert(*cur_state.duals, 
                          &edge_duals, 
                          &bias_dual);
    HypergraphWeights *dual_weights =
        weights_->modify(edge_duals, bias_dual);
    vector<double> chart;
    Hyperpath *path = viterbi_path(graph_, 
                                   *dual_weights, 
                                   &chart);
    result->dual = dual_weights->dot(*path);
    constraints_->subgradient(*path, &result->subgrad);

    vector<const Constraint *> failed_constraints;
    vector<int> counts;
    constraints_->check_constraints(*path,
                                    &failed_constraints, 
                                    &counts);
    for (int i = 0; i < failed_constraints.size(); ++i) {
      cerr << "Dual " << result->dual << endl;
      cerr << "Missed " << failed_constraints[i]->label << endl;
    }
    delete dual_weights;
    path_ = path;
  }

  mutable Hyperpath *path_;

 private:
  const Hypergraph *graph_;
  const HypergraphWeights *weights_;
  const HypergraphConstraints *constraints_;

};

Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights &theta,
    const HypergraphConstraints &constraints) {
  DecreasingRate rate;
  cerr << "decreasing" << endl;
  ConstrainedProducer producer(graph, &theta, &constraints);
  Subgradient subgradient(&producer, &rate);
  subgradient.set_debug();
  subgradient.solve();
  return producer.path_;
}
