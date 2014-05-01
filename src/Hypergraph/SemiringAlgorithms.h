// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRINGALGORITHMS_H_
#define HYPERGRAPH_SEMIRINGALGORITHMS_H_

#include <queue>
#include <set>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "Hypergraph/Potentials.h"

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
    Chart<S>(const Hypergraph *graph)
            : graph_(graph),
            chart_(graph->nodes().size(), S::zero()) {}

    void clear() { fill(chart_.begin(), chart_.end(), S::zero()); }

    void initialize_inside() {
        foreach (HNode node, graph_->nodes()) {
            if (node->terminal()) {
                insert(node, S::one());
            }
        }
    }

    // V& operator[] (HNode node) { return chart_[node->id()]; }
    inline V operator[] (HNode node) const { return chart_[node->id()]; }

    V get(HNode node) const { return chart_[node->id()]; }
    inline void insert(const HNode& node, const V& val) {
        chart_[node->id()] = val;
    }

    inline V compute_edge_score(HEdge edge, const V &start) {
        V score = start;
        foreach (HNode tail, edge->tail_nodes()) {
            score = S::times(score, chart_[tail->id()]);
        }
        return score;
    }

    void check(const Hypergraph *hypergraph) const {
        if (!hypergraph->same(*graph_)) {
            throw HypergraphException("Hypergraph does not match chart.");
        }
    }

    vector<V> chart() { return chart_; }
  protected:
    const Hypergraph *graph_;
    vector<V> chart_;
};

class BackPointers {
  public:
    explicit BackPointers(const Hypergraph *graph)
            : graph_(graph),
        chart_(graph_->nodes().size(), NULL) {}

    inline HEdge operator[] (HNode node) const {
        return chart_[node->id()];
    }

    HEdge get(HNode node) const { return chart_[node->id()]; }
    inline void insert(HNode node, HEdge val) {
        chart_[node->id()] = val;
    }

    void check(const Hypergraph *hypergraph) const {
        if (!hypergraph->same(*graph_)) {
            throw HypergraphException(
                "Hypergraph does not match backpointers.");
        }
    }

    Hyperpath *construct_path() const;

  protected:
    const Hypergraph *graph_;
    vector<HEdge> chart_;
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
void general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<SemiringType> &potentials,
    Chart<SemiringType> *chart,
    BackPointers *back);

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
            out_chart_(out_chart),
            node_marginals_(hypergraph->nodes().size()) {
                potentials->check(*hypergraph);
                in_chart->check(hypergraph);
                out_chart->check(hypergraph);
                foreach (HNode node, hypergraph_->nodes()) {
                    node_marginals_[node->id()] = marginal(node);
                }
            }

    ~Marginals() {
        delete in_chart_;
        delete out_chart_;
    }

    HypergraphPotentials<BoolPotential> *threshold(const V &threshold) const {
        HypergraphVectorPotentials<BoolPotential> *potentials =
                new HypergraphVectorPotentials<BoolPotential>(hypergraph_);
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
        return score;
    }

    V marginal(HNode node) const {
        return SemiringType::times((*in_chart_)[node], (*out_chart_)[node]);
    }

    template<typename OtherSemi>
            typename OtherSemi::ValType
            dot(HypergraphPotentials<OtherSemi> other) const {
        typename OtherSemi::ValType out_score = OtherSemi::one();
        foreach (HEdge edge, hypergraph_->edges()) {
            out_score =
                    OtherSemi::add(
                        out_score,
                        OtherSemi::times(marginal(edge),
                                         other.score(edge)));
        }
        return out_score;
    }

    const Hypergraph *hypergraph() const {
        return hypergraph_;
    }

    const vector<typename SemiringType::ValType> &node_marginals() const {
        return node_marginals_;
    }

    // Lazily creates array if non-existent.
    const vector<typename SemiringType::ValType> &edge_marginals() const {
        if (edge_marginals_.size() == 0) {
            edge_marginals_.resize(hypergraph_->edges().size());
            foreach (HEdge edge, hypergraph_->edges()) {
                edge_marginals_[edge->id()] = marginal(edge);
            }
        }
        return edge_marginals_;
    }
   private:
    const Hypergraph *hypergraph_;
    const HypergraphPotentials<SemiringType> *potentials_;

    // Pointer to inside and outside charts.
    // Note these are owned by the object.
    const Chart<SemiringType> *in_chart_;
    const Chart<SemiringType> *out_chart_;

    vector<typename SemiringType::ValType> node_marginals_;
    mutable vector<typename SemiringType::ValType> edge_marginals_;
};

#endif  // HYPERGRAPH_SEMIRINGALGORITHMS_H_
