// Copyright [2013] Alexander Rush


#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>

#include "Hypergraph/Algorithms.h"
#include "Hypergraph/Subgradient.h"

#define SPECIALIZE_FOR_SEMI(X)\
  template class Chart<X>;\
  template class HypergraphWeights<X>;\
  template class Marginals<X>;\
  template Hyperpath *general_viterbi<X>(const Hypergraph *graph,const HypergraphWeights<X> &weights);


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
               const HypergraphWeights<SemiringType> &weights) {
  weights.check(*graph);

  // Run Viterbi Hypergraph algorithm.
  Chart<SemiringType> *chart = new Chart<SemiringType>(graph);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      (*chart)[node] = SemiringType::one();
    }
  }
  foreach (HEdge edge, graph->edges()) {
    SemiringType score = weights.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      score *= (*chart)[node];
    }
    (*chart)[edge->head_node()] += score;
  }
  (*chart)[graph->root()] *= weights.bias();
  return chart;
}

template<typename SemiringType>
Chart<SemiringType> *
general_outside(const Hypergraph *graph,
                const HypergraphWeights<SemiringType> &weights,
                const Chart<SemiringType> &inside_chart) {
  weights.check(*graph);
  inside_chart.check(graph);
  Chart<SemiringType> *chart = new Chart<SemiringType>(graph);
  const vector<HEdge> &edges = graph->edges();
  (*chart)[graph->root()] = weights.bias();

  for (int i = edges.size() - 1; i >= 0; --i) {
    HEdge edge = edges[i];
    SemiringType head_score = (*chart)[edge->head_node()];
    foreach (HNode node, edge->tail_nodes()) {
      SemiringType other_score = SemiringType::one();
      foreach (HNode other_node, edge->tail_nodes()) {
        if (other_node->id() == node->id()) continue;
        other_score *= inside_chart[other_node];
      }
      (*chart)[node] += head_score * other_score * weights.score(edge);
    }
  }
  return chart;
}

template<typename SemiringType>
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphWeights<SemiringType> &weights) {

  weights.check(*graph);
  Chart<SemiringType> *chart = new Chart<SemiringType>(graph);
  vector<HEdge> back(graph->nodes().size(), NULL);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      (*chart)[node] = SemiringType::one();
    }
  }
  foreach (HEdge edge, graph->edges()) {
    SemiringType score = weights.score(edge);
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

template<typename SemiringType>
SemiringType general_subgradient(
    const Hypergraph &graph,
    const HypergraphWeights<SemiringType> &weights,
    const HypergraphWeights<SemiringType> &values)  {
  Hyperpath *path = general_viterbi(graph, weights);
  SemiringType subgradient = values.dot(path);
  delete path;
  return subgradient;
}

template<typename SemiringType>
SemiringType general_gradient(
    const Hypergraph &graph,
    const HypergraphWeights<SemiringType> &weights,
    const HypergraphWeights<SemiringType> &values) {

  Marginals<SemiringType> *marginals =
      Marginals<SemiringType>::compute(graph, weights);
  SemiringType gradient = marginals->dot(values);
  delete marginals;
  return gradient;
}

SPECIALIZE_FOR_SEMI(ViterbiWeight)
SPECIALIZE_FOR_SEMI(LogViterbiWeight)
SPECIALIZE_FOR_SEMI(InsideWeight)
SPECIALIZE_FOR_SEMI(BoolWeight)

// End General code.

class ConstrainedProducer : public SubgradientProducer {
 public:
  ConstrainedProducer(
      const Hypergraph *graph,
      const HypergraphWeights<LogViterbiWeight> *weights,
      const HypergraphWeights<SparseVectorWeight> *constraints)
      : graph_(graph),
        weights_(weights),
        constraints_(constraints) {}

  void solve(const SubgradState &cur_state, SubgradResult *result) {
    HypergraphWeights<LogViterbiWeight> weights(graph_);
    foreach (HEdge edge, graph_->edges()) {
      SparseVector edge_constraints =
          static_cast<SparseVector>(constraints_->score(edge));
      weights[edge] *= weights_->score(edge);
      foreach (SparsePair pair, edge_constraints) {
        weights[edge] *=
            LogViterbiWeight(pair.second * (*cur_state.duals)[pair.first]);
      }
    }
    SparseVector bias_constraints =
        static_cast<SparseVector>(constraints_->bias());
    weights.bias() = weights_->bias();
    foreach (SparsePair pair, bias_constraints) {
      weights.bias() *=
          LogViterbiWeight(pair.second *
                           (*cur_state.duals)[pair.first]);
    }

    HypergraphWeights<LogViterbiWeight> *dual_weights =
        weights_->times(weights);


    Hyperpath *path =
        general_viterbi<LogViterbiWeight>(graph_,
                                          *dual_weights);
    result->dual = (double)dual_weights->dot(*path);
    SparseVector final_constraints = static_cast<SparseVector>(constraints_->dot(*path));
    foreach (SparsePair pair, final_constraints) {
      result->subgrad[pair.first] = pair.second;
    }
    vector<const Constraint *> failed_constraints;
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
  const HypergraphWeights<LogViterbiWeight> *weights_;
  const HypergraphWeights<SparseVectorWeight> *constraints_;
  vector<ConstrainedResult> constrained_results_;
};

const Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights<LogViterbiWeight> &theta,
    const HypergraphWeights<SparseVectorWeight> &constraints,
    int number_of_constraints,
    vector<ConstrainedResult> *result) {
  theta.check(*graph);
  constraints.check(*graph);

  DecreasingRate rate;
  ConstrainedProducer producer(graph, &theta, &constraints);
  Subgradient subgradient(&producer, &rate, number_of_constraints);
  subgradient.set_debug();
  subgradient.solve();
  *result = producer.results();
  return (*result)[result->size() - 1].path;
}



// Hyperpath *viterbi_path(const Hypergraph *graph,
//                         const HypergraphWeights<double> &theta,
//                         vector<double> *chart) {
//   theta.check(*graph);

//   // Run Viterbi Hypergraph algorithm.
//   chart->clear();
//   chart->resize(graph->nodes().size(), -INF);

//   foreach (HNode node, graph->nodes()) {
//     if (node->terminal()) {
//       (*chart)[node->id()] = 0;
//     }
//   }
//   vector<HEdge> back(graph->nodes().size(), NULL);
//   foreach (HEdge edge, graph->edges()) {
//     double score = theta.score(edge);
//     int head_id = edge->head_node()->id();
//     foreach (HNode node, edge->tail_nodes()) {
//       score += (*chart)[node->id()];
//     }
//     if (score > (*chart)[head_id]) {
//       (*chart)[head_id] = score;
//       back[head_id] = edge;
//     }
//   }

//   // Collect backpointers.
//   vector<HEdge> path;
//   queue<HNode> to_examine;
//   to_examine.push(graph->root());
//   while (!to_examine.empty()) {
//     HNode node = to_examine.front();
//     HEdge edge = back[node->id()];
//     to_examine.pop();
//     if (edge == NULL) {
//       assert(node->terminal());
//       continue;
//     }
//     path.push_back(edge);
//     foreach (HNode node, edge->tail_nodes()) {
//       to_examine.push(node);
//     }
//   }
//   sort(path.begin(), path.end(), IdComparator());
//   return new Hyperpath(graph, path);
// }

// void outside(const Hypergraph *graph,
//              const HypergraphWeights<double> &weights,
//              const vector<double> &inside_chart,
//              vector<double> *chart) {
//   weights.check(*graph);
//   if (inside_chart.size() != graph->nodes().size()) {
//     throw HypergraphException("Chart size doesn't match graph");
//   }

//   chart->resize(graph->nodes().size(), -INF);
//   const vector<HEdge> &edges = graph->edges();

//   (*chart)[graph->root()->id()] = 0;
//   for (int i = edges.size() - 1; i >= 0; --i) {
//     HEdge edge = edges[i];
//     double full_score = weights.score(edge);
//     foreach (HNode node, edge->tail_nodes()) {
//       full_score += inside_chart[node->id()];
//     }
//     double head_score = (*chart)[edge->head_node()->id()];
//     foreach (HNode node, edge->tail_nodes()) {
//       double score = head_score + full_score - inside_chart[node->id()];
//       if (score > (*chart)[node->id()]) {
//         (*chart)[node->id()] = score;
//       }
//     }
//   }

//   // Add in the bias.
//   double bias = weights.bias();
//   for (uint i = 0; i < chart->size(); ++i) {
//     (*chart)[i] += bias;
//   }
// }

// const MaxMarginals *MaxMarginals::compute(
//     const Hypergraph *hypergraph,
//     const HypergraphWeights<double> *weights) {
//   weights->check(*hypergraph);
//   vector<double> *in_chart = new vector<double>();
//   vector<double> *out_chart = new vector<double>();

//   Hyperpath *path = viterbi_path(hypergraph,
//                                  *weights,
//                                  in_chart);
//   outside(hypergraph, *weights, *in_chart, out_chart);
//   delete path;
//   return new MaxMarginals(hypergraph, weights, in_chart, out_chart);
// }

// double MaxMarginals::max_marginal(HEdge edge) const {
//   double score = (*out_chart_)[edge->head_node()->id()];
//   score += weights_->score(edge);
//   foreach (HNode node, edge->tail_nodes()) {
//     score += (*in_chart_)[node->id()];
//   }
//   return score;
// }

// double MaxMarginals::max_marginal(HNode node) const {
//   return (*in_chart_)[node->id()] + (*out_chart_)[node->id()];
// }

// const HypergraphProjection *prune(const Hypergraph *original,
//                                   const HypergraphWeights<double> &weights,
//                                   double ratio) {
//   weights.check(*original);
//   const MaxMarginals *max_marginals =
//       MaxMarginals::compute(original, &weights);
//   double best = max_marginals->max_marginal(original->root());
//   double total_score = 0.0;
//   vector<bool> edge_mask(original->edges().size(), true);
//   foreach (HEdge edge, original->edges()) {
//     total_score += max_marginals->max_marginal(edge);
//   }
//   int prune = 0;
//   double average_score =
//       total_score / (float)original->edges().size();
//   assert(average_score - 1e-4 <= best);
//   foreach (HEdge edge, original->edges()) {
//     double score = max_marginals->max_marginal(edge);
//     if (score + 1e-4 <
//         (ratio * best  +  (1.0 - ratio) * average_score)) {
//       edge_mask[edge->id()] = false;
//       prune += 1;
//     }
//   }
//   cerr << average_score << " " << total_score << " " << (ratio * best  +  (1.0 - ratio) * average_score) << " " << best << " " << prune << endl;
//   delete max_marginals;
//   return HypergraphProjection::project_hypergraph(original, edge_mask);
// }


// const HypergraphProjection *prune(const Hypergraph *original,
//                                   const HypergraphWeights<BoolWeight> &weights) {

//   weights.check(*original);


//   const MaxMarginals *max_marginals =
//       MaxMarginals::compute(original, &weights);
//   double best = max_marginals->max_marginal(original->root());
//   double total_score = 0.0;
//   vector<bool> edge_mask(original->edges().size(), true);
//   foreach (HEdge edge, original->edges()) {
//     total_score += max_marginals->max_marginal(edge);
//   }
//   int prune = 0;
//   double average_score =
//       total_score / (float)original->edges().size();
//   assert(average_score - 1e-4 <= best);
//   foreach (HEdge edge, original->edges()) {
//     double score = max_marginals->max_marginal(edge);
//     if (score + 1e-4 <
//         (ratio * best  +  (1.0 - ratio) * average_score)) {
//       edge_mask[edge->id()] = false;
//       prune += 1;
//     }
//   }
//   cerr << average_score << " " << total_score << " " << (ratio * best  +  (1.0 - ratio) * average_score) << " " << best << " " << prune << endl;
//   delete max_marginals;
//   return HypergraphProjection::project_hypergraph(original, edge_mask);
// }
