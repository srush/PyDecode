// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_BEAMSEARCH_H_
#define HYPERGRAPH_BEAMSEARCH_H_

#include <list>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "Hypergraph/Algorithms.h"


struct BeamGroups {
    BeamGroups(const Hypergraph *graph,
               const vector<int> &groups,
               const vector<int> &group_limit,
               int num_groups)
    : hypergraph_(graph),
      groups_(groups),
      group_limit_(group_limit),
        num_groups_(num_groups),
        group_nodes_(num_groups) {
        if (hypergraph_->nodes().size() != groups.size()) {
            throw HypergraphException(
                "Hypergraph does not match groups.");
        }
        foreach (HNode node, hypergraph_->nodes()) {
            int group = groups_[node->id()];
            group_nodes_[group].push_back(node);
        }
    }

    int groups_size() const {
        return num_groups_;
    }

    int group_limit(int group) const {
        return group_limit_[group];
    }

    int group(HNode node) const {
        return groups_[node->id()];
    }

    const vector<HNode> &group_nodes(int group) const {
        return group_nodes_[group];
    }

    void check(const Hypergraph *hypergraph) const {
        if (!hypergraph->same(*hypergraph_)) {
            throw HypergraphException(
                "Hypergraph does not match groups.");
        }
    }

  private:
    const Hypergraph *hypergraph_;

    vector<int> groups_;
    vector<int> group_limit_;
    vector<vector<HNode> > group_nodes_;
    int num_groups_;
};

struct BeamHyp {
    BeamHyp() {}
  BeamHyp(HEdge _edge,
            HNode _node,
            binvec _sig,
            double _cs,
            double _fs,
            const vector<int> &_bp)
    : edge(_edge), node(_node), sig(_sig),
      current_score(_cs),
      future_score(_fs),
      back_position(_bp) {}

    double total_score() const {
        return current_score + future_score;
    }

    HEdge edge;
    HNode node;
    binvec sig;
    double current_score;
    double future_score;

    // Beam position for the tail nodes.
    vector<int> back_position;
};

class BeamChart {
  public:
    typedef vector<BeamHyp> Beam;
    typedef vector<BeamHyp * > BeamPointers;

    BeamChart(const Hypergraph *hypergraph,
              const BeamGroups *groups,
              const Chart<LogViterbiPotential> *future,
              double lower_bound)
      : hypergraph_(hypergraph),
            future_(future),
            lower_bound_(lower_bound),
            groups_(groups),
            current_group_(0),
            beam_(groups->groups_size()),
            beam_nodes_(hypergraph->nodes().size()) {}

    void insert(HNode node, HEdge edge, binvec bitmap, double val,
                const vector<int> &back_position);

    void finish(int group);

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

    // The (upper bound) future score and a lower bound of total score.
    const Chart<LogViterbiPotential> *future_;
    double lower_bound_;

    vector<Beam> beam_;
    vector<BeamPointers> beam_nodes_;

    // Mapping from nodes to beam group.
    const BeamGroups *groups_;
    int current_group_;
};

BeamChart *beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<BinaryVectorPotential> &constraints,
    const Chart<LogViterbiPotential> &outside,
    double lower_bound,
    const BeamGroups &groups);


#endif  // HYPERGRAPH_BEAMSEARCH_H_
