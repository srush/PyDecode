// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <vector>

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Constraints.h"
#include "./common.h"


Hyperpath *viterbi_path(const Hypergraph *graph,
                        const HypergraphWeights &theta,
                        vector<double> *chart);

void outside(const Hypergraph *graph,
             const HypergraphWeights &weights,
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

Hyperpath *best_constrained_path(
    const Hypergraph *graph,
    const HypergraphWeights &theta,
    const HypergraphConstraints &constraints,
    vector<ConstrainedResult> *duals);

#endif  // HYPERGRAPH_ALGORITHMS_H_
