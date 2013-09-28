// Copyright [2013] Alexander Rush
#include "Hypergraph/Constraints.h"

#include "Hypergraph/Hypergraph.h"

bool HypergraphConstraints::check_constraints(
    const Hyperpath &path,
    vector<const Constraint *> *failed_constraints,
    vector<int> *counts) const {
  foreach (Constraint *cons, constraints_) {
    int count = cons->bias;
    //foreach (HEdge edge, cons->edges_) {
    for (int i = 0; i < cons->edges.size(); ++i) {
      if (path.has_edge(cons->edges[i])) {
        count += cons->coefficients[i];
      }
    }
    if (count != 0) {
      failed_constraints->push_back(cons);
      counts->push_back(count);
    }
  }
  return failed_constraints->size() == 0;
}

void HypergraphConstraints::subgradient(const Hyperpath &path, 
                                        SparseVec *subgrad) const {
  vector<const Constraint *> constraints;
  vector<int> count;
  check_constraints(path, &constraints, &count);
  for (int i = 0; i < constraints.size(); ++i) {
    (*subgrad)(constraints[i]->id) = count[i];
  }
}

void HypergraphConstraints::convert(
    const SparseVec &dual_vector,
    SparseVec *edge_duals,
    double *bias_dual) const {
  *bias_dual = 0.0;
  for (SparseVec::const_iterator i = dual_vector.begin();
       i != dual_vector.end(); ++i ) {
    double dual = dual_vector[i.index()];
    const Constraint &cons = *constraints_[i.index()];
    *bias_dual += dual * cons.bias;
    for (int j = 0; j < cons.edges.size(); ++j) {
      (*edge_duals)[cons.edges[j]->id()] += dual * cons.coefficients[j];
    }
  }
}
