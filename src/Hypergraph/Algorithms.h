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

/**
 * A dynamic programming chart for SemiringType.
 * A templated vector over hypergraph nodes.
 */
template<typename SemiringType>
class Chart {
public:
  typedef SemiringType S;
  Chart<S>(const Hypergraph *hypergraph)
      : hypergraph_(hypergraph),
      chart_(hypergraph->nodes().size(), S::zero()) {}


  S& operator[] (HNode node) { return chart_[node->id()]; }
  const S& operator[] (HNode node) const {
    return chart_[node->id()];
  }

  const S& get(HNode node) const { return chart_[node->id()]; }


  void check(const Hypergraph *hypergraph) const {
    if (!hypergraph->same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match chart.");
    }
  }

protected:
  const Hypergraph *hypergraph_;
  vector<S> chart_;
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
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphWeights<SemiringType> &weights);

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

  HypergraphWeights<BoolWeight> *threshold(
      const SemiringType &threshold) const {
    HypergraphWeights<BoolWeight> *weights =
        new HypergraphWeights<BoolWeight>(hypergraph_);
    foreach (HEdge edge, hypergraph_->edges()) {
      (*weights)[edge] = BoolWeight(threshold < marginal(edge));
    }
    return weights;
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
    return new Marginals<SemiringType>(hypergraph, weights,
                                       in_chart, out_chart);
  }


  // Get max-marginal for edge or node.
  SemiringType marginal(HEdge edge) const {
      SemiringType score = (*out_chart_)[edge->head_node()];
      score *= weights_->score(edge);
      foreach (HNode node, edge->tail_nodes()) {
        score *= (*in_chart_)[node];
      }
      return score; /// (*in_chart_)[hypergraph_->root()];
  }

  SemiringType marginal(HNode node) const {
    return (*in_chart_)[node] * (*out_chart_)[node];
  }

  template<typename OtherSemi>
  OtherSemi dot(HypergraphWeights<OtherSemi> other) const {
    OtherSemi out_score = OtherSemi::one();
    foreach (HEdge edge, hypergraph_->edges()) {
      out_score += marginal(edge) * other.score(edge);
    }
    return out_score;
  }

  const Hypergraph *hypergraph() const {
    return hypergraph_;
  }

 private:
  const Hypergraph *hypergraph_;
  const HypergraphWeights<SemiringType> *weights_;

  // Pointer to inside and outside charts.
  // Note these are owned by the object.
  const Chart<SemiringType> *in_chart_;
  const Chart<SemiringType> *out_chart_;
};

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

const Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights<LogViterbiWeight> &theta,
    const HypergraphWeights<SparseVectorWeight> &constraints);


#endif  // HYPERGRAPH_ALGORITHMS_H_
