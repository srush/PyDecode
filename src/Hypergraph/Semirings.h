// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>
#include <bitset>
#include "Hypergraph/Factory.h"
#include "Hypergraph/Hypergraph.h"
#include "./common.h"

#define BITMAPSIZE 500

#define VALID_BINARY_VECTORS(X, Y)                \
    (X & Y).none()

class ViterbiPotential {
public:
    typedef double ValType;
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs = std::max(lhs, rhs);
        return lhs;
    }
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs *= rhs;
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::max(lhs, rhs);
        return ViterbiPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs *= rhs;
        return ViterbiPotential::normalize(lhs);
    }

    static inline ValType& normalize(ValType& val) {
        if (val < 0.0) val = 0.0;
        else if (val > 1.0) val = 1.0;
        return val;
    }

    static inline ValType zero() { return 0.0; }
    static inline ValType one() { return 1.0; }

    static inline ValType randValue() { return dRand(0.0,1.0); }
};

class LogViterbiPotential : public ViterbiPotential {
public:
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::max(lhs, rhs);
        return LogViterbiPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return LogViterbiPotential::normalize(lhs);
    }

    static inline ValType zero() { return -INF; }
    static inline ValType one() { return 0.0; }

    static inline ValType& normalize(ValType& val) {
        return val = val < -INF ? -INF : val;
    }

    static inline ValType randValue() { return dRand(-INF, 0.0); }
};

class InsidePotential : public ViterbiPotential {
public:
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return InsidePotential::normalize(lhs);
    }
};

class RealPotential {
public:
    typedef double ValType;

    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs = std::min(lhs, rhs);
        return lhs;
    }
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::min(lhs, rhs);
        return RealPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return RealPotential::normalize(lhs);
    }

    static inline ValType zero() { return INF; }
    static inline ValType one() { return 0.0; }

    static inline ValType& normalize(ValType& val) {
        if (val < NEGATIVE_INFINITY) val = NEGATIVE_INFINITY;
        else if (val > INF - INFINITY_BUFFER) val = INF;
        return val;
    }

    static ValType randValue() { return dRand(NEGATIVE_INFINITY, INF - INFINITY_BUFFER); }

protected:
    // Need specialized negative ranges for normalization purposes
    // and to hold to Semiring properties.
    static ValType INFINITY_BUFFER;
    static ValType NEGATIVE_INFINITY;
};

class TropicalPotential : public RealPotential {
public:
    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::min(lhs, rhs);
        return TropicalPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return TropicalPotential::normalize(lhs);
    }

    static inline ValType& normalize(ValType& val) {
        return val = val > INF ? INF : val < 0.0 ? 0.0 : val;
    }

    static ValType randValue() { return dRand(0.0, INF); }
};

class BoolPotential {
public:
    typedef bool ValType;

    static inline ValType add(const ValType& lhs, const ValType &rhs) {
        return lhs || rhs;
    }
    static inline ValType times(const ValType& lhs, const ValType &rhs) {
        return lhs && rhs;
    }

    static inline ValType safe_add(const ValType& lhs, const ValType &rhs) {
        return lhs || rhs;
    }
    static inline ValType safe_times(const ValType& lhs, const ValType &rhs) {
        return lhs && rhs;
    }

    static inline ValType& normalize(ValType& val) {
        return val;
    }

    static inline ValType one() { return true; }
    static inline ValType zero() { return false; }

    static inline ValType randValue() { return dRand(0.0,1.0) > .5; }
};

class CountingPotential {
public:
    typedef int ValType;

    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs *= rhs;
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return CountingPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType &rhs ) {
        lhs *= rhs;
        return CountingPotential::normalize(lhs);
    }

    static inline ValType& normalize(ValType& val) {
        return val = val < 0 ? 0 : val;
    }

    static inline ValType one() { return 1; }
    static inline ValType zero() { return 0; }

    static inline ValType randValue() {
        return rand();
    }
};

class IntPotential {
public:
    typedef int ValType;

    static inline ValType add(ValType lhs, const ValType &rhs) {
        /* lhs *= rhs; */
        /* return lhs; */
    }
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }

    static inline ValType one() { return 0; }
    static inline ValType zero() { return -1e5; }
};

/**
 * Comparison pair. *Experimental*
 * Type (s, t) op (s', t')
 * +: if (s > s') then (s, t) else (s', t')
 * *: (s * s', t * t')
 * 0: (0, 0)
 * 1: (1, 1)
 */
template<typename SemiringFirst, typename SemiringSecond>
class CompPotential {
public:
    typedef std::pair<typename SemiringFirst::ValType, typename SemiringSecond::ValType> ValType;

    static inline ValType add(ValType lhs, const ValType& rhs) {
        if (lhs.first < rhs.first) lhs = rhs;
        return lhs;
    }
    static inline ValType times(ValType lhs, const ValType& rhs) {
        lhs.first = SemiringFirst::times(lhs.first, rhs.first);
        lhs.second = SemiringSecond::times(lhs.second, rhs.second);
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType& rhs) {
        if (lhs.first < rhs.first) lhs = rhs;
        return CompPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType& rhs) {
        lhs.first = SemiringFirst::times(lhs.first, rhs.first);
        lhs.second = SemiringSecond::times(lhs.second, rhs.second);
        return CompPotential::normalize(lhs);
    }

    static inline ValType one() { return ValType(SemiringFirst::one(), SemiringSecond::one()); }
    static inline ValType zero() { return ValType(SemiringFirst::zero(), SemiringSecond::zero()); }

    static inline ValType& normalize(ValType& val) {
        val.first = SemiringFirst::normalize(val.first);
        val.second = SemiringSecond::normalize(val.second);
        return val;
    }

    static inline ValType randValue() {
        return ValType(SemiringFirst::randValue(), SemiringSecond::randValue());
    }
};



typedef pair<int, int> SparsePair;
// Sparse vector [(i_1, val_1), (i_2, val_2), ... (i_n, val_n)]
// Assumptions.
// (1) i_{j-1} < i_{j}
// (2) if no i_j then val_j = 0
// (3) if i_0 == -1 then it is zero value.
typedef vector<SparsePair> SparseVector;

struct Operator {
    virtual int operator()(int x, int y) const = 0;
};

struct PlusOperator : public Operator {
    int operator()(int x, int y) const { return x + y; }
};

struct MinOperator : public Operator {
    int operator()(int x, int y) const { return min(x, y); }
};

struct MaxOperator : public Operator {
    int operator()(int x, int y) const { return max(x, y); }
};

SparseVector combine_sparse_vectors(const SparseVector &value,
                                    const SparseVector &rhs,
                                    const Operator &op);

class SparseVectorPotential {
public:
    typedef SparseVector ValType;

    static inline bool is_zero(const ValType &val) {
        return val.size() == 1 && val[0].first == -1;
    }

    static inline ValType add(ValType lhs, const ValType& rhs) {
        if (SparseVectorPotential::is_zero(lhs)) return rhs;
        if (SparseVectorPotential::is_zero(rhs)) return lhs;
        return lhs;
    }

    static ValType times(ValType lhs, const ValType& rhs) {
        if (SparseVectorPotential::is_zero(lhs) ||
            SparseVectorPotential::is_zero(rhs)) {
            return SparseVectorPotential::zero();
        }
        return combine_sparse_vectors(lhs, rhs, PlusOperator());
    }

    static inline ValType one() { return ValType(); }
    static inline ValType zero() {
        SparseVector vec(1);
        vec[0] = pair<int, int>(-1, -1);
        return vec;
    }

    static inline ValType randValue() {
        SparseVector randVec;
        for(int i = 0; i < 20; i++) {
            randVec.push_back(SparsePair(rand(),rand()));
        }
        return randVec;
    }

    static inline ValType& normalize(ValType& val) {
        return val;
    }
};


class MinSparseVectorPotential : public SparseVectorPotential {
public:
    typedef SparseVector ValType;

    static inline ValType add(ValType lhs, const ValType& rhs) {
        if (SparseVectorPotential::is_zero(lhs)) return rhs;
        if (SparseVectorPotential::is_zero(rhs)) return lhs;
        return combine_sparse_vectors(lhs, rhs, MinOperator());
    }
};

class MaxSparseVectorPotential : public SparseVectorPotential {
public:
    typedef SparseVector ValType;

    static inline ValType add(ValType lhs, const ValType& rhs) {
        if (SparseVectorPotential::is_zero(lhs)) return rhs;
        if (SparseVectorPotential::is_zero(rhs)) return lhs;
        return combine_sparse_vectors(lhs, rhs, MaxOperator());
    }
};

/**
 * Tree. *Experimental*
 *
 * +: No action
 * *: NULL if either is NULL. Otherwise create a new node with rhs and lhs as tails
 * 0: Empty Vector
 * 1: Empty Vector
 */
class TreePotential {
public:
    typedef Hypernode* ValType;

    static inline ValType add(ValType lhs, const ValType& rhs) {
        return lhs;
    }
    static ValType times(ValType lhs, const ValType& rhs) {
        if (rhs == NULL or lhs == NULL) {
            lhs = NULL;
        } else {
            vector<HNode> tails;
            tails.push_back(lhs);
            tails.push_back(rhs);
            Hypernode *node = new Hypernode("");
            Hyperedge *edge = new Hyperedge("", node, tails);
            node->add_edge(edge);
            lhs = node;
        }
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType& rhs) {
        return lhs;
    }
    static ValType safe_times(ValType lhs, const ValType& rhs) {
        if (rhs == NULL or lhs == NULL) {
            lhs = NULL;
        } else {
            vector<HNode> tails;
            tails.push_back(lhs);
            tails.push_back(rhs);
            Hypernode *node = new Hypernode("");
            Hyperedge *edge = new Hyperedge("", node, tails);
            node->add_edge(edge);
            lhs = node;
        }
        return lhs;
    }

    static inline ValType one() { return new Hypernode(""); }
    static inline ValType zero() { return NULL; }

    static inline ValType& normalize(ValType& val) {
        // Is this necessary?
        return val;
    }

    static inline ValType randValue() {
        // Figure this one out.
        return TreePotential::one();
    }
};


// Classes used to associate projections with Hypergraphs

class HypergraphProjection;



template<typename SemiringType>
class HypergraphPotentials {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;
 public:
    HypergraphPotentials(const Hypergraph *hypergraph,
                         V bias)
            : hypergraph_(hypergraph),
            bias_(bias) {}

    virtual V dot(const Hyperpath &path) const = 0;

    virtual V score(HEdge edge) const = 0;
    virtual inline V operator[] (HEdge edge) const = 0;

    const V &bias() const { return bias_; }
    V &bias() { return bias_; }

    HypergraphPotentials<S> *project_potentials(
        const HypergraphProjection &projection) const;

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
            throw HypergraphException("Hypergraph potentials do not match potentials.");
        }
    }

    // Create a copy of the weights.
    virtual HypergraphPotentials *clone() const = 0;

    // TODO: This should be private. Fix template issue.
 public:

    const Hypergraph *hypergraph_;
    //vector<V> potentials_;
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

    HypergraphVectorPotentials(const Hypergraph *hypergraph)
            : HypergraphPotentials<SemiringType>(hypergraph, SemiringType::one()),
            potentials_(hypergraph->edges().size(), SemiringType::one()) {}

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            const vector<V> &potentials,
                            V bias) {
        return new HypergraphVectorPotentials<SemiringType>(hypergraph, potentials, bias);
    }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            score = SemiringType::times(score, potentials_[edge->id()]);
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

    /* const Hypergraph *hypergraph_; */
    vector<V> potentials_;
    /* V bias_; */

};

class HypergraphProjection {
 public:
    HypergraphProjection(const Hypergraph *original,
                         const Hypergraph *_new_graph,
                         const vector<HNode> *node_map,
                         const vector<HEdge> *edge_map,
                         bool bidirectional)
            : big_graph_(original),
            new_graph_(_new_graph),
            node_map_(node_map),
            edge_map_(edge_map),
            bidirectional_(bidirectional) {
                assert(node_map->size() == big_graph()->nodes().size());
                assert(edge_map->size() == big_graph()->edges().size());


                if (bidirectional) {
                    node_reverse_map_.resize(new_graph_->nodes().size(), NULL);
                    edge_reverse_map_.resize(new_graph_->edges().size(), NULL);
                    foreach (HNode node, original->nodes()) {
                        if ((*node_map_)[node->id()] == NULL) continue;
                        node_reverse_map_[(*node_map_)[node->id()]->id()] = node;
                    }
                    foreach (HEdge edge, original->edges()) {
                        if ((*edge_map_)[edge->id()] == NULL) continue;
                        edge_reverse_map_[(*edge_map_)[edge->id()]->id()] = edge;
                    }
                }

#ifndef NDEBUG

                foreach (HNode node, *node_map) {
                    assert(node == NULL ||
                                 node->id() < (int)_new_graph->nodes().size());
                }
                foreach (HEdge edge, *edge_map) {
                    assert(edge == NULL ||
                                 edge->id() < (int)_new_graph->edges().size());
                }
#endif
            }

    ~HypergraphProjection() {
        delete node_map_;
        delete edge_map_;
    }

    static HypergraphProjection *compose_projections(
        const HypergraphProjection *projection1, bool reverse1,
        const HypergraphProjection *projection2) {
        const Hypergraph *big_graph = projection1->big_graph();
        if (reverse1) big_graph = projection1->new_graph();

        vector<HEdge> *edge_map = new vector<HEdge>(big_graph->edges().size(), NULL);
        vector<HNode> *node_map = new vector<HNode>(big_graph->nodes().size(), NULL);
        foreach (HEdge edge, big_graph->edges()) {
            HEdge proj;
            if (reverse1) {
                proj = projection1->unproject(edge);
            } else {
                proj = projection1->project(edge);
            }
            if (proj != NULL && proj->id() >= 0) {
                (*edge_map)[edge->id()] = projection2->project(proj);
            }
        }
        foreach (HNode node, big_graph->nodes()) {
            HNode proj;
            if (reverse1) {
                proj = projection1->unproject(node);
            } else {
                proj = projection1->project(node);
            }
            if (proj != NULL && proj->id() >= 0) {
                (*node_map)[node->id()] = projection2->project(proj);
            }
        }

        return new HypergraphProjection(big_graph,
                                        projection2->new_graph(),
                                        node_map, edge_map, false);
    }


    static HypergraphProjection *project_hypergraph(
            const Hypergraph *hypergraph,
            const HypergraphPotentials<BoolPotential> &edge_mask);

    Hyperpath *project(const Hyperpath &original_path) const {
        original_path.check(*big_graph());
        vector<HEdge> edges;
        foreach (HEdge edge, original_path.edges()) {
            edges.push_back((*edge_map_)[edge->id()]);
        }
        return new Hyperpath(new_graph(), edges);
    }

    HEdge project(HEdge original) const {
        return (*edge_map_)[original->id()];
    }

    HNode project(HNode original) const {
        return (*node_map_)[original->id()];
    }

    HNode unproject(HNode original) const {
        assert(bidirectional_);
        return node_reverse_map_[original->id()];
    }

    HEdge unproject(HEdge original) const {
        assert(bidirectional_);
        return edge_reverse_map_[original->id()];
    }

    const Hypergraph *big_graph() const{
        return big_graph_;
    }
    const Hypergraph *new_graph() const{
        return new_graph_;
    }

    const Hypergraph *big_graph_;
    const Hypergraph *new_graph_;

 private:

    // Owned.
    const vector<HNode> *node_map_;
    const vector<HEdge> *edge_map_;
    vector<HNode> node_reverse_map_;
    vector<HEdge> edge_reverse_map_;

    bool bidirectional_;
};

template<typename SemiringType>
class HypergraphProjectedPotentials :
public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
    HypergraphProjectedPotentials(
        HypergraphPotentials<S> *base_potentials,
        const HypergraphProjection *projection)
            : HypergraphPotentials<SemiringType>(
                projection->big_graph(),
                base_potentials->bias()),
            base_potentials_(base_potentials),
            projection_(projection) {
                base_potentials->check(*projection->new_graph());
                //assert(base_potentials->size() == this->hypergraph_->edges().size());
            }

    static HypergraphPotentials<SemiringType> *
            make_potentials(HypergraphPotentials<S> *base_potentials,
                            const HypergraphProjection *projection) {
        return new HypergraphProjectedPotentials<SemiringType>(base_potentials, projection);
    }

    vector<V> &potentials() { return base_potentials_->potentials(); }
    const vector<V> &potentials() const { return base_potentials_->potentials(); }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V base_score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            base_score = SemiringType::times(base_score,
                                             score(edge));
        }
        return SemiringType::times(base_score, this->bias_);
    }

    V score(HEdge edge) const {
        HEdge new_edge = projection_->project(edge);
        return base_potentials_->score(new_edge);
    }
    inline V operator[] (HEdge edge) const {
        return score(edge);
    }

    HypergraphPotentials<S> *times(
        const HypergraphPotentials<S> &other) const {
        vector<typename SemiringType::ValType> new_potentials(projection_->new_graph()->edges().size());
        int i = -1;
        foreach (HEdge edge, projection_->new_graph()->edges()) {
            i++;
            new_potentials[i] = SemiringType::times(
                base_potentials_->score(edge),
                other.score(edge));
        }
        return new HypergraphProjectedPotentials<SemiringType>(
            new HypergraphVectorPotentials<SemiringType>(
                projection_->new_graph(),
                new_potentials,
                SemiringType::times(this->bias_, other.bias())),
            projection_);

    }

    HypergraphPotentials<S> *clone() const {
        return new HypergraphProjectedPotentials(
            base_potentials_->clone(),
            projection_);
    }


 protected:
    HypergraphPotentials<S> *base_potentials_;
    const HypergraphProjection *projection_;
};

void
pairwise_dot(const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
             const vector<double> &vec,
             HypergraphPotentials<LogViterbiPotential> *weights);

typedef bitset<BITMAPSIZE> binvec;

/**
 * Binary vector. *Experimental*
 *
 * +: Bitwise AND.
 * *: Bitwise OR.
 * 0: All one bitset.
 * 1: All zero bitset.
 */

class BinaryVectorPotential {
public:
	typedef bitset<BITMAPSIZE> ValType;

	static inline ValType add(ValType lhs, const ValType& rhs) {
		lhs &= rhs;
		return lhs;
	}

	static inline ValType times(ValType value, const ValType& rhs) {
		value |= rhs;
		return value;
	}

	static inline ValType one() {
		ValType vec = ValType(0x0);
		return vec;
	}

	static inline ValType zero() {
		ValType vec = ValType(0x0);
		vec.set();
		return vec;
	}

	static inline ValType randValue() { return ValType(dRand(0, 0xfffffff)); }

	static inline ValType& normalize(ValType& val) {
        return val;
    }
};

bool valid_binary_vectors(const bitset<BITMAPSIZE> &lhs,
                          const bitset<BITMAPSIZE> &rhs);

#endif // HYPERGRAPH_SEMIRING_H_
