// Copyright [2013] Alexander Rush
// Classes used to associate potentials with hypergraph.

#ifndef HYPERGRAPH_POTENTIALS_H_
#define HYPERGRAPH_POTENTIALS_H_

#include <map>
#include <vector>

#include "Hypergraph/Hypergraph.hh"
#include "Hypergraph/Semirings.hh"
#include "Hypergraph/Map.hh"
#include "./common.h"

class HypergraphMap;

template<typename SemiringType>
class HypergraphPotentials {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

  public:
    HypergraphPotentials(const Hypergraph *hypergraph)
            : hypergraph_(hypergraph) {}

    virtual ~HypergraphPotentials() {}

    virtual V dot(const Hyperpath &path) const = 0;

    virtual V score(HEdge edge) const = 0;
    virtual V operator[] (HEdge edge) const = 0;

    HypergraphPotentials<S> *project_potentials(
        const HypergraphMap &projection) const;

    virtual V *potentials() = 0;
    virtual const V *potentials() const = 0;

    /* /\** */
    /*  * Pairwise "times" with another set of potentials. */
    /*  * */
    /*  * @return New hypergraph potentials. */
    /*  *\/ */
    /* virtual HypergraphPotentials<S> *times( */
    /*         const HypergraphPotentials<S> &potentials) const = 0; */

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
};


template<typename SemiringType>
  class HypergraphVectorPotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
    HypergraphVectorPotentials(const Hypergraph *hypergraph,
                               vector<V> *potentials,
                               bool copy=true)
            : HypergraphPotentials<SemiringType>(hypergraph)
    {
        assert(potentials->size() == hypergraph->edges().size());
        if (copy) {
            potentials_ = new vector<V>(*potentials);
        } else {
            potentials_ = potentials;
        }
    }

    ~HypergraphVectorPotentials() {
        delete potentials_;
    }


    explicit HypergraphVectorPotentials(const Hypergraph *hypergraph)
            : HypergraphPotentials<SemiringType>(hypergraph) {
        potentials_ = new vector<V>(hypergraph->edges().size(),
                                    SemiringType::one());
    }

    V *potentials() { return this->potentials_->data(); }
    const V *potentials() const { return this->potentials_->data(); }

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            vector<V> *potentials,
                            bool copy=true) {
        return new HypergraphVectorPotentials<SemiringType>(
            hypergraph, potentials);
    }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            score = SemiringType::times(score,
                                        (*potentials_)[this->hypergraph_->id(edge)]);
        }
        return score;
    }

    V score(HEdge edge) const { return (*potentials_)[this->hypergraph_->id(edge)]; }
    inline V operator[] (HEdge edge) const {
        return (*potentials_)[this->hypergraph_->id(edge)];
    }

    void insert(const HEdge& edge, const V& val) {
        (*potentials_)[this->hypergraph_->id(edge)] = val;
    }

    /* HypergraphPotentials<S> *times( */
    /*     const HypergraphPotentials<S> &potentials) const; */

    HypergraphPotentials<S> *clone() const {
        return new HypergraphVectorPotentials(this->hypergraph_,
                                              potentials_);
    }

 protected:
    vector<V> *potentials_;

    // Is the potential vector owned by the class.
    bool owned_;
};

template<typename SemiringType>
  class HypergraphPointerPotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
    HypergraphPointerPotentials(const Hypergraph *hypergraph,
                                V *potentials)
            : HypergraphPotentials<SemiringType>(hypergraph)
    {
        assert(potentials->size() == hypergraph->edges().size());
        potentials_ = potentials;
    }

    ~HypergraphPointerPotentials() { }

    V *potentials() { return potentials_; }
    const V *potentials() const { return potentials_;}

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            V *potentials) {
        return new HypergraphPointerPotentials<SemiringType>(
            hypergraph, potentials);
    }


    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            score = SemiringType::times(score,
                                        potentials_[this->hypergraph_->id(edge)]);
        }
        return score;
    }

    V score(HEdge edge) const { return potentials_[this->hypergraph_->id(edge)]; }
    inline V operator[] (HEdge edge) const {
        return potentials_[this->hypergraph_->id(edge)];
    }

    HypergraphPotentials<S> *clone() const {
        return new HypergraphPointerPotentials(this->hypergraph_,
                                               potentials_);
    }

 protected:
    V *potentials_;
};


template<typename SemiringType>
  class HypergraphSparsePotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
    HypergraphSparsePotentials(const Hypergraph *hypergraph,
                               const map<int, int> &potential_map,
                               const vector<V> &potentials)
            : HypergraphPotentials<SemiringType>(hypergraph),
            potential_map_(potential_map),
            potentials_(potentials) {
                // assert(potentials.size() == hypergraph->edges().size());
            }

    V *potentials() { return potentials_.data(); }
    const V *potentials() const { return potentials_.data(); }

    explicit HypergraphSparsePotentials(const Hypergraph *hypergraph)
            : HypergraphPotentials<SemiringType>(hypergraph) {}

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            const map<int, int> &potentials_map,
                            const vector<V> &potentials) {
        return new HypergraphSparsePotentials<SemiringType>(
            hypergraph, potentials_map, potentials);
    }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V cur_score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            cur_score = SemiringType::times(cur_score, score(edge));
        }
        return cur_score;
    }

    inline V score(HEdge edge) const {
        map<int, int>::const_iterator iter =
                potential_map_.find(this->hypergraph_->id(edge));
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
        potential_map_[this->hypergraph_->id(edge)] = i;
        potentials_.push_back(val);
    }

    /* HypergraphPotentials<S> *times( */
    /*     const HypergraphPotentials<S> &potentials) const; */

    HypergraphPotentials<S> *clone() const {
        return new HypergraphSparsePotentials(this->hypergraph_,
                                              potential_map_,
                                              potentials_);
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

    V *potentials() { return base_potentials_->potentials(); }
    const V *potentials() const {
        return base_potentials_->potentials(); }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V base_score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            base_score = SemiringType::times(base_score,
                                             score(edge));
        }
        return base_score;
    }

    V score(HEdge edge) const;

    inline V operator[] (HEdge edge) const {
        return score(edge);
    }

    /* HypergraphPotentials<S> *times( */
    /*     const HypergraphPotentials<S> &other) const; */

    HypergraphPotentials<S> *clone() const {
        return new HypergraphMappedPotentials(
            base_potentials_->clone(),
            projection_);
    }

 protected:
    HypergraphPotentials<S> *base_potentials_;
    const HypergraphMap *projection_;
};

/* void pairwise_dot( */
/*     const HypergraphPotentials<SparseVectorPotential> &sparse_potentials, */
/*     const vector<double> &vec, */
/*     HypergraphPotentials<LogViterbiPotential> *weights); */

void non_zero_weights(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &weights,
    HypergraphVectorPotentials<BoolPotential> *updates);


#endif  // HYPERGRAPH_POTENTIALS_H_
