// Copyright [2013] Alexander Rush

#include "Hypergraph/Constraints.h"

// bool HypergraphConstraints::check_constraints(
//     const Hyperpath &path,
//     vector<const Constraint *> *failed_constraints,
//     vector<int> *counts) const {
//   foreach (const Constraint *cons, constraints_) {
//     int count = cons->bias;
//     for (uint i = 0; i < cons->edges.size(); ++i) {
//       if (path.has_edge(cons->edges[i])) {
//         count += cons->coefficients[i];
//       }
//     }
//     if (count != 0) {
//       failed_constraints->push_back(cons);
//       counts->push_back(count);
//     }
//   }
//   return failed_constraints->size() == 0;
// }

// void HypergraphConstraints::subgradient(
//     const Hyperpath &path,
//     vector<double> *subgrad) const {
//   vector<const Constraint *> constraints;
//   vector<int> count;
//   check_constraints(path, &constraints, &count);
//   for (uint i = 0; i < constraints.size(); ++i) {
//     (*subgrad)[constraints[i]->id] = count[i];
//   }
// }

// HypergraphWeights<LogViterbiWeight> *
// HypergraphConstraints::convert(
//     const vector<double> &dual_vector) const {
//   HypergraphWeights<LogViterbiWeight> weights =
//       new HypergraphWeights<LogViterbiWeight>(hypergraph_);

//   for (uint i = 0; i < dual_vector.size(); ++i) {
//     double dual = dual_vector[i];
//     const Constraint &cons = *constraints_[i];
//     weights.set_bias(weights.bias() *
//                      LogViterbiWeight(dual * cons.bias));

//     for (uint j = 0; j < cons.edges.size(); ++j) {
//       weights[cons.edges[j]] *=
//           LogViterbiWeight(dual * cons.coefficients[j]);
//     }
//   }
//   return weights;
// }
