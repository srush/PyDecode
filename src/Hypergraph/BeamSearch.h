// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_BEAMSEARCH_H_
#define HYPERGRAPH_BEAMSEARCH_H_

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "Hypergraph/Algorithms.h"


template <size_t N>
class LessThan {
public:
  bool operator() (const bitset<N> &lhs, const bitset<N> &rhs) const {
	for (int i = N-1; i > 0; i--) {
	  if (lhs[i] != rhs[i]) {
	  	return (lhs[i] < rhs[i]);
	  }
	}
	return false;
  }
};

struct BeamScore {
    BeamScore() {}

  BeamScore(HEdge e, double cs, double fs) :
        edge(e),
        current_score(cs),
        future_score(fs) {}

    double total_score() const {
        return current_score + future_score;
    }

    HEdge edge;
    double current_score;
    double future_score;
};

class BeamChart {
public:
  typedef map<binvec, BeamScore, LessThan<BITMAPSIZE> > BeamMap;
  typedef vector<pair<binvec, BeamScore> > Beam;

  BeamChart(const Hypergraph *hypergraph,
            int beam_size,
            const Chart<LogViterbiPotential> *future,
            double lower_bound)
      : hypergraph_(hypergraph),
          beam_size_(beam_size),
          future_(future),
          lower_bound_(lower_bound),
          chart_(hypergraph->nodes().size()),
          beam_(hypergraph->nodes().size()) {}

  BeamScore get(HNode node, binvec bitmap) {
      return chart_[node->id()][bitmap];
  }

  void insert(HNode node, HEdge edge, binvec bitmap, double val);

  void finish(HNode node);

  const BeamMap &get_map(HNode node) const {
  	return chart_[node->id()];
  }

  const Beam &get_beam(HNode node) const {
  	return beam_[node->id()];
  }

  void check(const Hypergraph *hypergraph) const {
    if (!hypergraph->same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match chart.");
    }
  }

  HEdge get_best_edge(HNode node) {
      const Beam &beam = get_beam(node);
      double score = -INF;
      HEdge best = NULL;
      for (int i = 0; i < beam.size(); ++i) {
          if (beam[i].second.current_score > score) {
              best = beam[i].second.edge;
              score = beam[i].second.current_score;
          }
      }
      return best;
  }

  Hyperpath *get_path();

protected:
  const Hypergraph *hypergraph_;

  int beam_size_;

  // The (upper bound) future score and a lower bound of total score.
  const Chart<LogViterbiPotential> *future_;
  double lower_bound_;

  vector<BeamMap> chart_;
  vector<Beam> beam_;
};

BeamChart *beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<BinaryVectorPotential> &constraints,
    const Chart<LogViterbiPotential> &outside,
    double lower_bound,
    int beam_size);


#endif  // HYPERGRAPH_BEAMSEARCH_H_
