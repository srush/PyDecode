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
  typedef SemiringType S;
  typedef typename SemiringType::ValType V;

public:
  Chart<S>(const Hypergraph *hypergraph)
      : hypergraph_(hypergraph),
      chart_(hypergraph->nodes().size(), S::zero()) {}


  // V& operator[] (HNode node) { return chart_[node->id()]; }
  V operator[] (HNode node) const { return chart_[node->id()]; }

  V get(HNode node) const { return chart_[node->id()]; }
  inline void insert(const HNode& node, const V& val) { chart_[node->id()] = val; }


  void check(const Hypergraph *hypergraph) const {
    if (!hypergraph->same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match chart.");
    }
  }

protected:
  const Hypergraph *hypergraph_;
  vector<V> chart_;
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

Hyperpath *beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<SparseVectorPotential> &constraints,
    const Chart<LogViterbiPotential> &outside);

template<typename SemiringType>
class Marginals {
  typedef SemiringType S;
  typedef typename SemiringType::ValType V;
 public:

  Marginals(const Hypergraph *hypergraph,
            const HypergraphPotentials<S> *potentials,
            const Chart<S> *in_chart,
            const Chart<S> *out_chart)
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

  HypergraphPotentials<BoolPotential> *threshold(const V &threshold) const {
    HypergraphPotentials<BoolPotential> *potentials =
        new HypergraphPotentials<BoolPotential>(hypergraph_);
    foreach (HEdge edge, hypergraph_->edges()) {
      potentials->insert(edge, threshold < marginal(edge));
    }
    return potentials;
  }

  // Compute the max-marginals for the potentialed hypergraph.
  static const Marginals *compute(
                                  const Hypergraph *hypergraph,
                                  const HypergraphPotentials<S> *potentials) {
    Chart<S> *in_chart = general_inside<S>(hypergraph, *potentials);
    Chart<S> *out_chart = general_outside<S>(hypergraph, *potentials,
                                            *in_chart);
    return new Marginals<S>(hypergraph, potentials,
                           in_chart, out_chart);
  }


  // Get max-marginal for edge or node.
  V marginal(HEdge edge) const {
      V score = (*out_chart_)[edge->head_node()];
      score = SemiringType::times(score, potentials_->score(edge));
      foreach (HNode node, edge->tail_nodes()) {
        score = SemiringType::times(score, (*in_chart_)[node]);
      }
      return score; /// (*in_chart_)[hypergraph_->root()];
  }

  V marginal(HNode node) const {
    return SemiringType::times((*in_chart_)[node], (*out_chart_)[node]);
  }

  template<typename OtherSemi>
  typename OtherSemi::ValType dot(HypergraphPotentials<OtherSemi> other) const {
    typename OtherSemi::ValType out_score = OtherSemi::one();
    foreach (HEdge edge, hypergraph_->edges()) {
      out_score = OtherSemi::add(out_score,
                      OtherSemi::times(marginal(edge), other.score(edge)));
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
