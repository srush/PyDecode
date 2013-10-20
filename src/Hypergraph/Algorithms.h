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
                        const HypergraphWeights<DoubleWeight> &theta,
                        vector<double> *chart);

void outside(const Hypergraph *graph,
             const HypergraphWeights<DoubleWeight> &weights,
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
               const HypergraphWeights<DoubleWeight> *weights,
               const vector<double> *in_chart,
               const vector<double> *out_chart)
      : hypergraph_(hypergraph),
      weights_(weights),
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
                                     const HypergraphWeights<DoubleWeight> *weights);

  // Get max-marginal for edge or node.
  double max_marginal(HEdge edge) const;
  double max_marginal(HNode node) const;

 private:
  const Hypergraph *hypergraph_;
  const HypergraphWeights<DoubleWeight> *weights_;

  // Pointer to inside and outside charts.
  // Note these are owned by the object.
  const vector<double> *in_chart_;
  const vector<double> *out_chart_;
};


const Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights<DoubleWeight> &theta,
    const HypergraphConstraints &constraints,
    vector<ConstrainedResult> *duals);


const HypergraphProjection *prune(const Hypergraph *original,
                                  const HypergraphWeights<DoubleWeight> &weights,
                                  double ratio);


#endif  // HYPERGRAPH_ALGORITHMS_H_
