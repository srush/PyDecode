// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_BEAMSEARCH_H_
#define HYPERGRAPH_BEAMSEARCH_H_

#include <list>
#include <stack>
#include <deque>
#include <vector>
#include <boost/intrusive/rbtree.hpp>


#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "Hypergraph/Algorithms.h"

using namespace boost::intrusive;

struct BeamGroups {
    // groups : vector size of nodes.
    // group_limit : max size of each group.
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


template<typename BVP>
class BeamChart {
  public:
    bool exact;

    struct BeamHyp
              : public set_base_hook<>{

       set_member_hook<> member_hook_;


        BeamHyp() {}

        BeamHyp(HEdge _edge,
                HNode _node,
                typename BVP::ValType _sig,
                double _cs,
                double _fs,
                int _bp_left,
                int _bp_right)
        : edge(_edge), node(_node), sig(_sig),
            current_score(_cs),
            future_score(_fs),
            total_score(_cs + _fs),
            back_position_left(_bp_left),
            back_position_right(_bp_right) {}

         void reset(HEdge _edge,
                    HNode _node,
                    typename BVP::ValType _sig,
                    double _cs,
                    double _fs,
                    int _bp_left,
                    int _bp_right) {
             edge = _edge;
             node = _node;
             sig = _sig;
             current_score = _cs;
             future_score = _fs;
             total_score = _cs + _fs;
             back_position_left = _bp_left;
             back_position_right = _bp_right;
         }

        bool operator<(const BeamHyp &other) const {
            return total_score >= other.total_score;
        }

        HEdge edge;
        HNode node;
        typename BVP::ValType sig;
        double current_score;
        double future_score;
        double total_score;

        // Beam position for the tail nodes.
        int back_position_left;
        int back_position_right;
    };

    struct delete_disposer {
        void operator()(BeamHyp *delete_this) {
            delete delete_this;
        }
    };

    typedef rbtree<BeamHyp> Beam;
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
            beam_size_(groups->groups_size(), 0),
            beam_nodes_(hypergraph->nodes().size()),
            exact(true) {
        for (int i = 0; i < groups->groups_size(); ++i) {
            beam_[i] = new Beam();
        }

    }

    ~BeamChart();

    // Insert an a hypothesis into the chart.
    //
    void insert(HNode node,
                HEdge edge,
                const typename BVP::ValType &sig,
                double val,
                const int back_position_left,
                const int back_position_right
                );

    // Finish a beam group.
    void finish(int group);

    //
    const BeamPointers &get_beam(HNode node) const {
        return beam_nodes_[node->id()];
    }

    static BeamChart<BVP> *beam_search(
            const Hypergraph *graph,
            const HypergraphPotentials<LogViterbiPotential> &potentials,
            const HypergraphPotentials<BVP> &constraints,
            const Chart<LogViterbiPotential> &outside,
            double lower_bound,
            const BeamGroups &groups);

    Hyperpath *get_path(int result);

    void check(const Hypergraph *hypergraph) const {
        if (!hypergraph->same(*hypergraph_)) {
            throw HypergraphException("Hypergraph does not match chart.");
        }
    }



  protected:
    const Hypergraph *hypergraph_;

    // The (upper bound) future score and a lower bound of total score.
    const Chart<LogViterbiPotential> *future_;
    double lower_bound_;

    vector<Beam *> beam_;
    vector<int> beam_size_;
    vector<BeamPointers> beam_nodes_;

    // Mapping from nodes to beam group.
    const BeamGroups *groups_;
    int current_group_;

    stack<BeamHyp *> hyp_pool_;

};


template<typename BVP>
bool comp(typename BeamChart<BVP>::BeamHyp *hyp1,
          typename BeamChart<BVP>::BeamHyp *hyp2);

#endif  // HYPERGRAPH_BEAMSEARCH_H_
