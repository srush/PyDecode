// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
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
    const HypergraphPotentials<SemiringType> &potentials);

template<typename SemiringType>
Chart<SemiringType> *general_outside(
    const Hypergraph *graph,
    const HypergraphPotentials<SemiringType> &potentials,
    const Chart<SemiringType> &inside_chart);

template<typename SemiringType>
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<SemiringType> &potentials);

template<typename SemiringType>
class Marginals {
 public:

  Marginals(const Hypergraph *hypergraph,
            const HypergraphPotentials<SemiringType> *potentials,
            const Chart<SemiringType> *in_chart,
            const Chart<SemiringType> *out_chart)
      : hypergraph_(hypergraph),
      potentials_(potentials),
      in_chart_(in_chart),
      out_chart_(out_chart) {
        potentials->check(*hypergraph);
        in_chart->check(hypergraph);
        out_chart->check(hypergraph);
      }

  ~Marginals() {
    delete in_chart_;
    delete out_chart_;
  }

  HypergraphPotentials<BoolPotential> *threshold(
      const SemiringType &threshold) const {
    HypergraphPotentials<BoolPotential> *potentials =
        new HypergraphPotentials<BoolPotential>(hypergraph_);
    foreach (HEdge edge, hypergraph_->edges()) {
      (*potentials)[edge] = BoolPotential(threshold < marginal(edge));
    }
    return potentials;
  }

  // Compute the max-marginals for the potentialed hypergraph.
  static const Marginals *compute(
      const Hypergraph *hypergraph,
      const HypergraphPotentials<SemiringType> *potentials) {
    Chart<SemiringType> *in_chart =
        general_inside<SemiringType>(hypergraph, *potentials);
    Chart<SemiringType> *out_chart =
        general_outside<SemiringType>(hypergraph, *potentials,
                                      *in_chart);
    return new Marginals<SemiringType>(hypergraph, potentials,
                                       in_chart, out_chart);
  }


  // Get max-marginal for edge or node.
  SemiringType marginal(HEdge edge) const {
      SemiringType score = (*out_chart_)[edge->head_node()];
      score *= potentials_->score(edge);
      foreach (HNode node, edge->tail_nodes()) {
        score *= (*in_chart_)[node];
      }
      return score; /// (*in_chart_)[hypergraph_->root()];
  }

  SemiringType marginal(HNode node) const {
    return (*in_chart_)[node] * (*out_chart_)[node];
  }

  template<typename OtherSemi>
  OtherSemi dot(HypergraphPotentials<OtherSemi> other) const {
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
  const HypergraphPotentials<SemiringType> *potentials_;

  // Pointer to inside and outside charts.
  // Note these are owned by the object.
  const Chart<SemiringType> *in_chart_;
  const Chart<SemiringType> *out_chart_;
};



#endif  // HYPERGRAPH_ALGORITHMS_H_
