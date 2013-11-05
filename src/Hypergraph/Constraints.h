// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_CONSTRAINTS_H_
#define HYPERGRAPH_CONSTRAINTS_H_

#include <string>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"

struct Constraint {
 public:
  Constraint(string _label, int _id) : label(_label), id(_id) {}

  void set_constant(int _bias) { bias = _bias; }

  void add_edge_term(HEdge edge, int coefficient) {
    edges.push_back(edge);
    coefficients.push_back(coefficient);
  }

  bool has_edge(HEdge edge) const {
    foreach (HEdge cedge, edges) {
      if (cedge->id() == edge->id()) return true;
    }
    return false;
  }

  int get_edge_coefficient(HEdge edge) const {
    for (int i = 0; i < edges.size(); ++i) {
      if (edge->id() == edges[i]->id()) return coefficients[i];
    }
    return 0;
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

  /* bool check_constraints( */
  /*     const Hyperpath &path, */
  /*     vector<const Constraint *> *failed_constraints, */
  /*     vector<int> *count) const; */

  /* HypergraphWeights<LogViterbiWeight> * */
  /*     convert(const vector<double> &dual_vector) const; */

  /* void subgradient(const Hyperpath &path, */
  /*                  vector<double> *subgrad) const; */

  HypergraphWeights<SparseVectorWeight> *semi() const {
    HypergraphWeights<SparseVectorWeight> *weights =
        new HypergraphWeights<SparseVectorWeight>(hypergraph_);

    for (uint i = 0; i < constraints_.size(); ++i) {
      const Constraint &cons = *constraints_[i];
      SparseVector vec;
      vec.push_back(pair<int, int>(i, cons.bias));

      weights->set_bias(weights->bias() *
                       SparseVectorWeight(vec));

      for (uint j = 0; j < cons.edges.size(); ++j) {
        SparseVector vec;
        vec.push_back(pair<int, int>(i, cons.coefficients[j]));
        (*weights)[cons.edges[j]] *= SparseVectorWeight(vec);

      }
    }
    return weights;
  }

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
