// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_BEAMSEARCH_H_
#define HYPERGRAPH_BEAMSEARCH_H_

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "Hypergraph/Algorithms.h"

#include <list>

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

  BeamScore(HEdge e, double cs, double fs,
            const vector<int> &bp) :
        edge(e),
        current_score(cs),
            future_score(fs),
            back_position(bp) {}

    double total_score() const {
        return current_score + future_score;
    }

    HEdge edge;
    double current_score;
    double future_score;
    vector<int> back_position;
};

class BeamChart {
public:
    //typedef map<binvec, BeamScore, LessThan<BITMAPSIZE> > BeamMap;
  typedef vector<pair<binvec, BeamScore> > Beam;
  typedef vector<pair<binvec, BeamScore> *> BeamPointers;

  BeamChart(const Hypergraph *hypergraph,
            int beam_size,
            const Chart<LogViterbiPotential> *future,
            double lower_bound,
            const vector<int> &groups,
            int num_groups)
      : hypergraph_(hypergraph),
          beam_size_(beam_size),
          future_(future),
          lower_bound_(lower_bound),
          //chart_(hypergraph->nodes().size()),
          beam_nodes_(hypergraph->nodes().size()),
          beam_(num_groups),
          groups_(groups),
          nodes_left_(num_groups) {
              foreach (HNode node, hypergraph->nodes()) {
                  int group = groups_[node->id()];
                  nodes_left_[group].insert(node->id());
              }
          }

  /* BeamScore get(HNode node, binvec bitmap) { */
  /*     return chart_[node->id()][bitmap]; */
  /* } */

  void insert(HNode node, HEdge edge, binvec bitmap, double val,
              const vector<int> &back_position);

  void finish(HNode node);

  /* const BeamMap &get_map(HNode node) const { */
  /* 	return chart_[node->id()]; */
  /* } */

  const BeamPointers &get_beam(HNode node) const {
  	return beam_nodes_[node->id()];
  }

  void check(const Hypergraph *hypergraph) const {
    if (!hypergraph->same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match chart.");
    }
  }

  Hyperpath *get_path(int result);

protected:
  const Hypergraph *hypergraph_;

  int beam_size_;

  // The (upper bound) future score and a lower bound of total score.
  const Chart<LogViterbiPotential> *future_;
  double lower_bound_;

  //vector<BeamMap> chart_;
  vector<Beam> beam_;
  vector<BeamPointers> beam_nodes_;

  // Mapping from nodes to beam group.
  vector<int> groups_;
  vector<set<int> > nodes_left_;
};

BeamChart *beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<BinaryVectorPotential> &constraints,
    const Chart<LogViterbiPotential> &outside,
    double lower_bound,
    int beam_size);


#endif  // HYPERGRAPH_BEAMSEARCH_H_
