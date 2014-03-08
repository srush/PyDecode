// Copyright [2013] Alexander Rush
// Classes used to associate potentials with hypergraph.

#ifndef HYPERGRAPH_POTENTIALS_H_
#define HYPERGRAPH_POTENTIALS_H_

#include <map>
#include <vector>

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "./common.h"

class HypergraphMap;

template<typename SemiringType>
class HypergraphPotentials {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

  public:
    HypergraphPotentials(const Hypergraph *hypergraph,
                         V bias)
            : hypergraph_(hypergraph),
            bias_(bias) {}

    virtual ~HypergraphPotentials() {}

    virtual V dot(const Hyperpath &path) const = 0;

    virtual V score(HEdge edge) const = 0;
    virtual V operator[] (HEdge edge) const = 0;

    const V &bias() const { return bias_; }
    V &bias() { return bias_; }

    HypergraphPotentials<S> *project_potentials(
        const HypergraphMap &projection) const;

    virtual vector<V> &potentials() = 0;
    virtual const vector<V> &potentials() const = 0;

    /**
     * Pairwise "times" with another set of potentials.
     *
     * @return New hypergraph potentials.
     */
    virtual HypergraphPotentials<S> *times(
            const HypergraphPotentials<S> &potentials) const = 0;

    void check(const Hypergraph &graph) const {
        if (!graph.same(*hypergraph_)) {
            throw HypergraphException("Hypergraph does not match potentials.");
        }
    }

    void check(const HypergraphPotentials<S> &potentials) const {
        if (!potentials.hypergraph_->same(*hypergraph_)) {
            throw HypergraphException(
                "Hypergraph potentials do not match potentials.");
        }
    }

    // Create a copy of the weights.
    virtual HypergraphPotentials *clone() const = 0;

    // TODO(srush): This should be private. Fix template issue.
  public:
    const Hypergraph *hypergraph_;
    V bias_;
};


template<typename SemiringType>
  class HypergraphVectorPotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
    HypergraphVectorPotentials(const Hypergraph *hypergraph,
                               const vector<V> &potentials,
                               V bias)
            : HypergraphPotentials<SemiringType>(hypergraph, bias),
            potentials_(potentials) {
            assert(potentials.size() == hypergraph->edges().size());
    }
    vector<V> &potentials() { return potentials_; }
    const vector<V> &potentials() const { return potentials_; }

    explicit HypergraphVectorPotentials(const Hypergraph *hypergraph)
            : HypergraphPotentials<SemiringType>(hypergraph,
                                                 SemiringType::one()),
            potentials_(hypergraph->edges().size(), SemiringType::one()) {}

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            const vector<V> &potentials,
                            V bias) {
        return new HypergraphVectorPotentials<SemiringType>(
            hypergraph, potentials, bias);
    }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            score = SemiringType::times(score,
                                        potentials_[edge->id()]);
        }
        return SemiringType::times(score, this->bias_);
    }

    V score(HEdge edge) const { return potentials_[edge->id()]; }
    inline V operator[] (HEdge edge) const {
        return potentials_[edge->id()];
    }

    void insert(const HEdge& edge, const V& val) {
        potentials_[edge->id()] = val;
    }

    HypergraphPotentials<S> *times(
        const HypergraphPotentials<S> &potentials) const;

    HypergraphPotentials<S> *clone() const {
        return new HypergraphVectorPotentials(this->hypergraph_,
                                              potentials_,
                                              this->bias_);
    }

 protected:
    vector<V> potentials_;
};

template<typename SemiringType>
  class HypergraphSparsePotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
    HypergraphSparsePotentials(const Hypergraph *hypergraph,
                               const map<int, int> &potential_map,
                               const vector<V> &potentials,
                               V bias)
            : HypergraphPotentials<SemiringType>(hypergraph, bias),
            potential_map_(potential_map),
            potentials_(potentials) {
                // assert(potentials.size() == hypergraph->edges().size());
            }

    vector<V> &potentials() { return potentials_; }
    const vector<V> &potentials() const { return potentials_; }

    explicit HypergraphSparsePotentials(const Hypergraph *hypergraph)
            : HypergraphPotentials<SemiringType>(hypergraph,
                                                 SemiringType::one()) {}

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            const map<int, int> &potentials_map,
                            const vector<V> &potentials,
                            V bias) {
        return new HypergraphSparsePotentials<SemiringType>(
            hypergraph, potentials_map, potentials, bias);
    }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V cur_score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            cur_score = SemiringType::times(cur_score, score(edge));
        }
        return SemiringType::times(cur_score, this->bias_);
    }

    inline V score(HEdge edge) const {
        map<int, int>::const_iterator iter =
                potential_map_.find(edge->id());
        if (iter == potential_map_.end()) {
            return S::one();
        } else {
            return potentials_[iter->second];
        }
    }
    inline V operator[] (HEdge edge) const {
        return score(edge);
    }

    void insert(const HEdge& edge, const V& val) {
        int i = potentials_.size();
        potential_map_[edge->id()] = i;
        potentials_.push_back(val);
    }

    HypergraphPotentials<S> *times(
        const HypergraphPotentials<S> &potentials) const;

    HypergraphPotentials<S> *clone() const {
        return new HypergraphSparsePotentials(this->hypergraph_,
                                              potential_map_,
                                              potentials_,
                                              this->bias_);
    }

 protected:
    map<int, int> potential_map_;
    vector<V> potentials_;
};

template<typename SemiringType>
  class HypergraphMappedPotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

  public:
    HypergraphMappedPotentials(
        HypergraphPotentials<S> *base_potentials,
        const HypergraphMap *projection);

    static HypergraphPotentials<SemiringType> *
            make_potentials(HypergraphPotentials<S> *base_potentials,
                            const HypergraphMap *projection) {
        return new HypergraphMappedPotentials<SemiringType>(
            base_potentials, projection);
    }

    vector<V> &potentials() { return base_potentials_->potentials(); }
    const vector<V> &potentials() const {
        return base_potentials_->potentials(); }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V base_score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            base_score = SemiringType::times(base_score,
                                             score(edge));
        }
        return SemiringType::times(base_score, this->bias_);
    }

    V score(HEdge edge) const;

    inline V operator[] (HEdge edge) const {
        return score(edge);
    }

    HypergraphPotentials<S> *times(
        const HypergraphPotentials<S> &other) const;

    HypergraphPotentials<S> *clone() const {
        return new HypergraphMappedPotentials(
            base_potentials_->clone(),
            projection_);
    }

 protected:
    HypergraphPotentials<S> *base_potentials_;
    const HypergraphMap *projection_;
};

void pairwise_dot(
    const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
    const vector<double> &vec,
    HypergraphPotentials<LogViterbiPotential> *weights);

void non_zero_weights(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &weights,
    HypergraphVectorPotentials<BoolPotential> *updates);


#endif  // HYPERGRAPH_POTENTIALS_H_
