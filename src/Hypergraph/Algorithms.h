// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Constraints.h"
#include "Hypergraph/Semirings.h"

// General Code.

template<typename SemiringType>
class Chart {
public:
  Chart<SemiringType>(const Hypergraph *hypergraph)
      : hypergraph_(hypergraph),
      chart_(hypergraph->nodes().size(), SemiringType::zero()) {}

  void check(const Hypergraph *hypergraph) const {
    if (!hypergraph->same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match chart.");
    }
  }

  SemiringType& operator[] (HNode node) {
    return chart_[node->id()];
  }

  const SemiringType& operator[] (HNode node) const {
    return chart_[node->id()];
  }

  const SemiringType& get(HNode node) const {
    return chart_[node->id()];
  }

protected:
  const Hypergraph *hypergraph_;
  vector<SemiringType> chart_;
};

template<typename SemiringType>
Chart<SemiringType> *general_inside(
    const Hypergraph *graph,
    const HypergraphWeights<SemiringType> &weights);

template<typename SemiringType>
Chart<SemiringType> *general_outside(
    const Hypergraph *graph,
    const HypergraphWeights<SemiringType> &weights,
    const Chart<SemiringType> &inside_chart);

template<typename SemiringType>
class Marginals {
 public:

  Marginals(const Hypergraph *hypergraph,
            const HypergraphWeights<SemiringType> *weights,
            const Chart<SemiringType> *in_chart,
            const Chart<SemiringType> *out_chart)
      : hypergraph_(hypergraph),
      weights_(weights),
      in_chart_(in_chart),
      out_chart_(out_chart) {
        weights->check(*hypergraph);
        in_chart->check(hypergraph);
        out_chart->check(hypergraph);
      }

  ~Marginals() {
    delete in_chart_;
    delete out_chart_;
  }

  // Compute the max-marginals for the weighted hypergraph.
  static const Marginals *compute(
      const Hypergraph *hypergraph,
      const HypergraphWeights<SemiringType> *weights) {

    Chart<SemiringType> *in_chart =
        general_inside<SemiringType>(hypergraph, *weights);
    Chart<SemiringType> *out_chart =
        general_outside<SemiringType>(hypergraph, *weights,
                                      *in_chart);
    // TODO: fix me!
    return new Marginals<SemiringType>(hypergraph, weights, in_chart, out_chart);
  }

  // Get max-marginal for edge or node.
  SemiringType marginal(HEdge edge) const {
      SemiringType score = (*out_chart_)[edge->head_node()];
      score *= weights_->score(edge);
      foreach (HNode node, edge->tail_nodes()) {
        score *= (*in_chart_)[node];
      }
      return score;
  }

  SemiringType marginal(HNode node) const {
    return (*in_chart_)[node] * (*out_chart_)[node];
  }

 private:
  const Hypergraph *hypergraph_;
  const HypergraphWeights<SemiringType> *weights_;

  // Pointer to inside and outside charts.
  // Note these are owned by the object.
  const Chart<SemiringType> *in_chart_;
  const Chart<SemiringType> *out_chart_;
};

// TODO(srush): deprecate / specialize
// Viterbi Specific code

Hyperpath *viterbi_path(const Hypergraph *graph,
                        const HypergraphWeights<double> &theta,
                        vector<double> *chart);


void outside(const Hypergraph *graph,
             const HypergraphWeights<double> &weights,
             const vector<double> &inside_chart,
             vector<double> *chart);

class ConstrainedResult {
 public:
  ConstrainedResult() {}
  ConstrainedResult(
      const Hyperpath *path_,
      double dual_,
      double primal_,
      const vector<const Constraint *> &constraints_)
      : path(path_),
        dual(dual_),
        primal(primal_),
        constraints(constraints_) {}

  const Hyperpath *path;
  double dual;
  double primal;
  vector<const Constraint *> constraints;
};

class MaxMarginals {
 public:

  MaxMarginals(const Hypergraph *hypergraph,
               const HypergraphWeights<double> *weights,
               const vector<double> *in_chart,
               const vector<double> *out_chart)
      : weights_(weights),
        in_chart_(in_chart),
        out_chart_(out_chart) {
        assert(in_chart->size() == out_chart->size());
        assert(hypergraph->nodes().size() == out_chart->size());
      }

  ~MaxMarginals() {
    delete in_chart_;
    delete out_chart_;
  }

  // Compute the max-marginals for the weighted hypergraph.
  static const MaxMarginals *compute(const Hypergraph *hypergraph,
                                     const HypergraphWeights<double> *weights);

  // Get max-marginal for edge or node.
  double max_marginal(HEdge edge) const;
  double max_marginal(HNode node) const;

 private:
  const HypergraphWeights<double> *weights_;

  // Pointer to inside and outside charts.
  // Note these are owned by the object.
  const vector<double> *in_chart_;
  const vector<double> *out_chart_;
};


const Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights<double> &theta,
    const HypergraphConstraints &constraints,
    vector<ConstrainedResult> *duals);

const HypergraphProjection *prune(const Hypergraph *original,
                                  const HypergraphWeights<double> &weights,
                                  double ratio);


#endif  // HYPERGRAPH_ALGORITHMS_H_
