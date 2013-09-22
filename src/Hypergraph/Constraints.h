// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_CONSTRAINTS_H_
#define HYPERGRAPH_CONSTRAINTS_H_

#include "Hypergraph/Hypergraph.h"
#include "./common.h"

struct Constraint {
 public:
  Constraint(string _label, int _id) : label(_label), id(_id) {}

  void set_bias(int _bias) { bias = _bias; }

  void add_edge(HEdge edge, int coefficient) {
    edges.push_back(edge);
    coefficients.push_back(coefficient);
  }

  string label;
  vector<HEdge> edges;
  vector<int> coefficients;
  int bias;
  int id;
};

class HypergraphConstraints {
 public:
  HypergraphConstraints(const Hypergraph *hypergraph)
      : hypergraph_(hypergraph) {}

  Constraint *add_constraint(string label) {
    Constraint *cons = new Constraint(label, constraints_.size());
    constraints_.push_back(cons);
    return constraints_.back();
  }

  bool check_constraints(
      const Hyperpath &path,
      vector<const Constraint *> *failed_constraints,
      vector<int> *count) const;

  SparseVec convert(const SparseVec &dual_vector,
                    SparseVec *edge_duals,
                    double *bias_dual) const;

  SparseVec subgradient(const Hyperpath &path) const;

 private:
  const Hypergraph *hypergraph_;
  vector<Constraint *> constraints_;
};


#endif  // HYPERGRAPH_HYPERGRAPH_H_
