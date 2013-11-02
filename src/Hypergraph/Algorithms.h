// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Constraints.h"
#include "Hypergraph/Semirings.h"


Hyperpath *viterbi_path(const Hypergraph *graph,
                        const HypergraphWeights<double> &theta,
                        vector<double> *chart);

template<typename SemiringType>
void general_inside(const Hypergraph *graph,
                    const HypergraphWeights<SemiringType> &theta,
                    vector<SemiringType> *chart);

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

template<typename SemiringType>
class Marginals {
 public:

  Marginals(const Hypergraph *hypergraph,
            const HypergraphWeights<SemiringType> *weights,
            const vector<SemiringType> *in_chart,
            const vector<SemiringType> *out_chart)
      : hypergraph_(hypergraph),
      weights_(weights),
      in_chart_(in_chart),
      out_chart_(out_chart) {
        assert(in_chart->size() == out_chart->size());
        assert(hypergraph->nodes().size() == out_chart->size());
      }

  ~Marginals() {
    delete in_chart_;
    delete out_chart_;
  }

  // Compute the max-marginals for the weighted hypergraph.
  static const Marginals *compute(const Hypergraph *hypergraph,
                                  const HypergraphWeights<SemiringType> *weights) {
    weights->check(*hypergraph);
    vector<SemiringType> *in_chart = new vector<SemiringType>();
    vector<SemiringType> *out_chart = new vector<SemiringType>();

    general_inside<SemiringType>(hypergraph, *weights, in_chart);
    //outside(hypergraph, *weights, *in_chart, out_chart);
    // TODO: fix me!
    return new Marginals<SemiringType>(hypergraph, weights, in_chart, out_chart);

  }

  // Get max-marginal for edge or node.
  SemiringType marginal(HEdge edge) const {
      SemiringType score = (*out_chart_)[edge->head_node()->id()];
      score *= weights_->score(edge);
      foreach (HNode node, edge->tail_nodes()) {
        score *= (*in_chart_)[node->id()];
      }
      return score;
  }

  SemiringType marginal(HNode node) const {
    return (*in_chart_)[node->id()] * (*out_chart_)[node->id()];
  }

 private:
  const Hypergraph *hypergraph_;
  const HypergraphWeights<SemiringType> *weights_;

  // Pointer to inside and outside charts.
  // Note these are owned by the object.
  const vector<SemiringType> *in_chart_;
  const vector<SemiringType> *out_chart_;
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
