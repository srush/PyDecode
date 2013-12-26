// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "Hypergraph/Potentials.h"




// Compute tail node set for each hypergraph node.
vector<set<int> > *children_nodes(const Hypergraph &graph);

struct IdComparator {
    bool operator()(HEdge edge1, HEdge edge2) const {
        return edge1->id() < edge2->id();
    }
};

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

  protected:
    const Hypergraph *graph_;
    vector<V> chart_;
};

class BackPointers {
  public:
    BackPointers(const Hypergraph *graph)
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
            throw HypergraphException("Hypergraph does not match backpointers.");
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
Hyperpath *count_constrained_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<SemiringType> &weight_potentials,
    const HypergraphPotentials<CountingPotential> &count_potentials,
    int limit);

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

  private:
    const Hypergraph *hypergraph_;
    const HypergraphPotentials<SemiringType> *potentials_;

    // Pointer to inside and outside charts.
    // Note these are owned by the object.
    const Chart<SemiringType> *in_chart_;
    const Chart<SemiringType> *out_chart_;
};

template<typename S>
class DynamicViterbi {
  public:
    DynamicViterbi<S>(Hypergraph *graph)
            : graph_(graph),
            last_chart_(NULL),
            last_bp_(NULL),
            children_sets_(children_nodes(*graph)) {}

    void initialize(
        const HypergraphPotentials<S> &potentials) {
        chart_ = new Chart<S>(graph_);
        bp_ = new BackPointers(graph_);
        general_viterbi<S>(graph_, potentials, chart_, bp_);
        update_pointers();
    }

    void update(
        const HypergraphPotentials<S> &updated_potentials,
        set<int> *updated);

    const BackPointers *back_pointers() const {
        last_bp_->check(graph_);
        return last_bp_;
    }

  private:
    void update_pointers() {
        delete last_chart_;
        delete last_bp_;
        last_chart_ = chart_;
        last_bp_ = bp_;
    }

    const Hypergraph *graph_;
    const Chart<S> *last_chart_;
    const BackPointers *last_bp_;
    Chart<S> *chart_;
    BackPointers *bp_;
    const vector<set<int> > *children_sets_;
};


HypergraphProjection *extend_hypergraph_by_count(
    Hypergraph *graph,
    const HypergraphPotentials<CountingPotential> &potentials,
    int lower_limit,
    int upper_limit,
    int goal);


Chart<SetPotential> *edge_domination(const Hypergraph &graph);
Chart<SetPotential> *node_domination(const Hypergraph &graph);

template<class Set1, class Set2>
bool is_disjoint(const Set1 &set1, const Set2 &set2);


vector<set<int> > *children_sparse(
    const Hypergraph *graph,
    const HypergraphPotentials<SparseVectorPotential> &potentials);

set<int> *updated_nodes(
    const Hypergraph *graph,
    const vector<set<int> > &children,
    const set<int> &updated);


#endif  // HYPERGRAPH_ALGORITHMS_H_
