// Copyright [2013] Alexander Rush

#include "Hypergraph/Algorithms.h"

#include <assert.h>
#include <algorithm>
#include <queue>
#include <vector>


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
      if (score > (*chart)[node->id()]) {
        (*chart)[node->id()] = score;
      }
    }
  }

  // Add in the bias.
  double bias = weights.bias();
  for (uint i = 0; i < chart->size(); ++i) {
    (*chart)[i] += bias;
  }
}

const MaxMarginals *MaxMarginals::compute(
    const Hypergraph *hypergraph,
    const HypergraphWeights *weights) {
  vector<double> *in_chart = new vector<double>();
  vector<double> *out_chart = new vector<double>();

  Hyperpath *path = viterbi_path(hypergraph,
                                 *weights,
                                 in_chart);
  outside(hypergraph, *weights, *in_chart, out_chart);
  delete path;
  return new MaxMarginals(hypergraph, weights, in_chart, out_chart);
}

double MaxMarginals::max_marginal(HEdge edge) const {
  double score = (*out_chart_)[edge->head_node()->id()];
  score += weights_->score(edge);
  foreach (HNode node, edge->tail_nodes()) {
    score += (*in_chart_)[node->id()];
  }
  return score;
}

double MaxMarginals::max_marginal(HNode node) const {
  return (*in_chart_)[node->id()] + (*out_chart_)[node->id()];
}

const HypergraphProjection *prune(const Hypergraph *original,
                                  const HypergraphWeights &weights,
                                  double ratio) {
  const MaxMarginals *max_marginals =
      MaxMarginals::compute(original, &weights);
  double best = max_marginals->max_marginal(original->root());
  double total_score = 0.0;
  vector<bool> edge_mask(original->edges().size(), true);
  foreach (HEdge edge, original->edges()) {
    total_score += max_marginals->max_marginal(edge);
  }
  int prune = 0;
  double average_score = total_score / original->edges().size();
  foreach (HEdge edge, original->edges()) {
    double score = max_marginals->max_marginal(edge);
    if (score < (ratio * best  +  (1.0 - ratio) * average_score)) {
      edge_mask[edge->id()] = false;
      prune += 1;
    }
  }
  //cerr << average_score << " " << total_score << " " << (ratio * best  +  (1.0 - ratio) * average_score) << " " << prune << endl;
  delete max_marginals;
  return HypergraphProjection::project_hypergraph(original, edge_mask);
}

class ConstrainedProducer : public SubgradientProducer {
 public:
  ConstrainedProducer(
      const Hypergraph *graph,
      const HypergraphWeights *weights,
      const HypergraphConstraints *constraints)
      : graph_(graph),
        weights_(weights),
        constraints_(constraints) {}

  void solve(const SubgradState &cur_state,
             SubgradResult *result) {
    vector<double> edge_duals(graph_->edges().size());
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


    for (uint i = 0; i < failed_constraints.size(); ++i) {
      // cerr << "Dual " << result->dual << endl;
      // cerr << "Missed " << failed_constraints[i]->label << endl;
    }
    constrained_results_.push_back(
        ConstrainedResult(path, result->dual, 0.0,
                          failed_constraints));
    delete dual_weights;
  }

  vector<ConstrainedResult> results() const {
    return constrained_results_;
  }

 private:
  const Hypergraph *graph_;
  const HypergraphWeights *weights_;
  const HypergraphConstraints *constraints_;
  vector<ConstrainedResult> constrained_results_;
};

const Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights &theta,
    const HypergraphConstraints &constraints,
    vector<ConstrainedResult> *result) {
  DecreasingRate rate;
  //cerr << "decreasing" << endl;
  ConstrainedProducer producer(graph, &theta, &constraints);
  Subgradient subgradient(&producer, &rate,
                          constraints.constraints().size());
  subgradient.set_debug();
  subgradient.solve();
  *result = producer.results();
  return (*result)[result->size() - 1].path;
}
