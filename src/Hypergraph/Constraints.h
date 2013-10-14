// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_CONSTRAINTS_H_
#define HYPERGRAPH_CONSTRAINTS_H_

#include <string>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"

struct Constraint {
 public:
  Constraint(string _label, int _id) : label(_label), id(_id) {}

  void set_constant(int _bias) { bias = _bias; }

  void add_edge_term(HEdge edge, int coefficient) {
    edges.push_back(edge);
    coefficients.push_back(coefficient);
  }

  bool has_edge(HEdge edge) const {
    for (HEdge cedge : edges) {
      if (cedge->id() == edge->id()) return true;
    }
    return false;
  }

  string label;
  vector<HEdge> edges;
  vector<int> coefficients;
  int bias;
  int id;
};

class HypergraphConstraints {
 public:
  explicit HypergraphConstraints(const Hypergraph *hypergraph)
      : hypergraph_(hypergraph) {}

  Constraint *add_constraint(string label) {
    Constraint *cons = new Constraint(label, constraints_.size());
    constraints_.push_back(cons);
    return cons;
  }

  bool check_constraints(
      const Hyperpath &path,
      vector<const Constraint *> *failed_constraints,
      vector<int> *count) const;

  void convert(const vector<double> &dual_vector,
               vector<double> *edge_duals,
               double *bias_dual) const;

  void subgradient(const Hyperpath &path,
                   vector<double> *subgrad) const;

  const Hypergraph *hypergraph() const { return hypergraph_; }

  const vector<const Constraint *> &constraints() const {
      return constraints_;
  }

  void check(const Hypergraph &graph) const {
    if (!graph.same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match constraints.");
    }
  }

 private:
  const Hypergraph *hypergraph_;
  vector<const Constraint *> constraints_;
};


#endif  // HYPERGRAPH_CONSTRAINTS_H_
