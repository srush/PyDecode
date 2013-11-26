// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>
#include "Hypergraph/Factory.h"
#include "Hypergraph/Hypergraph.h"
#include "./common.h"
/**
 * When deriving or creating your own semiring type, you must specify:
 * -add/times & safe_add/safe_times (mainly for testing)
 * -one/zero
 * -normalize
 * -randValue
 *
 * These properties must also hold:
 */

/**
 * Viterbi Semiring Potential for tracking the probability
 * of the highest scoring derivation.
 * Concept from Huang 2006.
 */
class ViterbiPotential {
public:
    typedef double ValType; /**< This semiring is represented by doubles. */

    /**
     * Add is performed by returning the max of the two elements. NOT domain-safe.
     * @return     The max of the two elements.
     */
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs = std::max(lhs, rhs);
        return lhs;
    }
    /**
     * Times is the same as the underlying * operation. NOT domain-safe.
     * @return     The multiplication of the two elements.
     */
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs *= rhs;
        return lhs;
    }

    /**
     * A normalized version of the add operation. Domain Safe.
     * @return     Normalized max of the two arguments.
     */
    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::max(lhs, rhs);
        return ViterbiPotential::normalize(lhs);
    }
    /**
     * A normalized version of the times operation. Domain Safe.
     * @return     Normalized times of the two arguments.
     */
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs *= rhs;
        return ViterbiPotential::normalize(lhs);
    }

    /**
     * Normalizes a ValType to the specific domain of this Semiring.
     * @param  val ValType reference to normalize
     * @return     A reference to the modified argument after it is normalized
     *               to the correct domain.
     */
    static inline ValType& normalize(ValType& val) {
        if (val < 0.0) val = 0.0;
        else if (val > 1.0) val = 1.0;
        return val;
    }

    /**
     * The annihlator element of the semiring.
     * @return 0.0
     */
    static inline ValType zero() { return 0.0; }
    /**
     * The identity element of the semiring.
     * @return 1.0
     */
    static inline ValType one() { return 1.0; }

    /**
     * Provides a semi-random ValType that is within the domain specified
     * by the semiring.
     * @return A random value between 0.0 and 1.0
     */
    static inline ValType randValue() { return dRand(0.0,1.0); }
};

/**
 * Log Viterbi Semiring Potential for tracking the most likely derivation
 * in log space. Inherits from ViterbiPotential.
 */
class LogViterbiPotential : public ViterbiPotential {
public:
    /**
     * Adds the two log potentials. NOT domain-safe.
     * @return     An addition of the two arguments.
     */
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * A normalized version of the add operation. Domain Safe.
     * @return     Normalized max of the two arguments.
     */
    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::max(lhs, rhs);
        return LogViterbiPotential::normalize(lhs);
    }
    /**
     * A normalized version of the times operation. Domain Safe.
     * @return     Normalized times of the two arguments.
     */
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return LogViterbiPotential::normalize(lhs);
    }


    /**
     * The annihlator element of the semiring.
     * @return Negative infinity.
     */
    static inline ValType zero() { return -INF; }
    /**
     * The identity element of the semiring.
     * @return 0.0
     */
    static inline ValType one() { return 0.0; }

    /**
     * Normalizes a ValType to the specific domain of this Semiring.
     * @param  val ValType reference to normalize
     * @return     A reference to the modified argument after it is normalized
     *               to the correct domain.
     */
    static inline ValType& normalize(ValType& val) {
        return val = val < -INF ? -INF : val;
    }

    /**
     * Provides a semi-random ValType that is within the domain specified
     * by the semiring.
     * @return A random value between 0.0 and 1.0
     */
    static inline ValType randValue() { return dRand(-INF, 0.0); }
};

/**
 * Inside Semiring Potential for expressing the total probability of all derivations.
 * Inherits from ViterbiPotential.
 * Concept from Huang 2006.
 */
class InsidePotential : public ViterbiPotential {
public:
    /**
     * Adds the two potentials. Not domain-safe.
     * @return     An addition of the two arguments.
     */
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * A normalized version of the add operation. Domain Safe.
     * @return     Normalized addition of the two arguments.
     */
    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return InsidePotential::normalize(lhs);
    }
};


/**
 * Real Semiring Potential for providing the shortest distance.
 * Concept from Huang 2006.
 */
class RealPotential {
public:
    typedef double ValType; /**< This semiring is represented by doubles. */

    /**
     * Add is performed by returning the min of the two elements. NOT domain-safe.
     * @return     The min of the two elements.
     */
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs = std::min(lhs, rhs);
        return lhs;
    }
    /**
     * Adds the two ValTypes. NOT domain-safe.
     * @return     The addition of the two elements.
     */
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * A normalized version of the add operation. Domain Safe.
     * @return     Normalized max of the two arguments.
     */
    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::min(lhs, rhs);
        return RealPotential::normalize(lhs);
    }
    /**
     * A normalized version of the times operation. Domain Safe.
     * @return     Normalized times of the two arguments.
     */
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return RealPotential::normalize(lhs);
    }

    /**
     * The annihlator element of the semiring.
     * @return Infinity
     */
    static inline ValType zero() { return INF; }
    /**
     * The identity element of the semiring.
     * @return 0.0
     */
    static inline ValType one() { return 0.0; }

    /**
     * Normalizes a ValType to the specific domain of this Semiring.
     * @param  val ValType reference to normalize
     * @return     A reference to the modified argument after it is normalized
     *               to the correct domain.
     */
    static inline ValType& normalize(ValType& val) {
        if (val < NEGATIVE_INFINITY) val = NEGATIVE_INFINITY;
        else if (val > INF - INFINITY_BUFFER) val = INF;
        return val;
    }

    /**
     * Provides a semi-random ValType that is within the domain specified
     * by the semiring.
     * @return A random value in the domain of all reals and positive infinity.
     */
    static ValType randValue() { return dRand(NEGATIVE_INFINITY, INF - INFINITY_BUFFER); }

protected:
    // Need specialized negative ranges for normalization purposes
    // and to hold to Semiring properties when dealing with infinities.
    static ValType INFINITY_BUFFER;
    static ValType NEGATIVE_INFINITY;
};

/**
 * Tropical Semiring Potential for providing the shortest distance with
 * non-negative weights. Inherits from RealPotential.
 * Concept from Huang 2006.
 */
class TropicalPotential : public RealPotential {
public:
    /**
     * A normalized version of the add operation. Domain Safe.
     * @return     Normalized min of the two arguments.
     */
    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = std::min(lhs, rhs);
        return TropicalPotential::normalize(lhs);
    }
    /**
     * A normalized version of the times operation. Domain Safe.
     * @return     Normalized addition of the two arguments.
     */
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return TropicalPotential::normalize(lhs);
    }

    /**
     * Normalizes a ValType to the specific domain of this Semiring.
     * @param  val ValType reference to normalize
     * @return     A reference to the modified argument after it is normalized
     *               to the correct domain.
     */
    static inline ValType& normalize(ValType& val) {
        return val = val > INF ? INF : val < 0.0 ? 0.0 : val;
    }

    /**
     * Provides a semi-random ValType that is within the domain specified
     * by the semiring.
     * @return A random value between 0.0 and infinity (here 1e8)
     */
    static ValType randValue() { return dRand(0.0, INF); }
};


/**
 * Boolean Semiring Potential for use in logical deductions, and recognition.
 * Concept from Huang 2006.
 */
class BoolPotential {
public:
    typedef bool ValType;  /**< This semiring is represented by booleans. */

    /**
     * Add is performed by returning the logical OR of the two elements.
     * Domain safe due to ValType domain restrictions.
     * @return     Logical OR of the two elements.
     */
    static inline ValType add(const ValType& lhs, const ValType &rhs) {
        return lhs || rhs;
    }
    /**
     * Times is performed by returning the logical AND of the two elements.
     * Domain safe due to ValType domain restrictions.
     * @return     Logical AND of the two elements.
     */
    static inline ValType times(const ValType& lhs, const ValType &rhs) {
        return lhs && rhs;
    }

    /**
     * No difference from above add
     * @return     Logical OR of the two arguments.
     */
    static inline ValType safe_add(const ValType& lhs, const ValType &rhs) {
        return lhs || rhs;
    }
    /**
     * No difference from above times
     * @return     Logical AND of the two arguments.
     */
    static inline ValType safe_times(const ValType& lhs, const ValType &rhs) {
        return lhs && rhs;
    }

    /**
     * This function has no implementation due to the type specific domain.
     * It merely returns a reference to the argument, and due to it being inlined,
     * will likely result in no extra operation.
     * @param  val ValType reference to normalize
     * @return     A reference to the given argument
     */
    static inline ValType& normalize(ValType& val) {
        return val;
    }

    /**
     * The identity element of the semiring.
     * @return true
     */
    static inline ValType one() { return true; }
    /**
     * The annihlator element of the semiring.
     * @return false
     */
    static inline ValType zero() { return false; }

    /**
     * Provides a semi-random ValType.
     * @return A semi-random value either true or false
     */
    static inline ValType randValue() { return dRand(0.0,1.0) > .5; }
};

/**
 * Counting Semiring Potential for tracking the number of derivations.
 * Concept from Huang 2006.
 */
class CountingPotential {
public:
    typedef int ValType; /**< This semiring is represented by integers. */

    /**
     * Add is performed by returning the addition of the two elements. NOT domain-safe.
     * @return     The addition of the two elements.
     */
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }
    /**
     * Times is the same as the underlying * operation. NOT domain-safe.
     * @return     The multiplication of the two elements.
     */
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs *= rhs;
        return lhs;
    }

    /**
     * A normalized version of the add operation. Domain Safe.
     * @return     Normalized addition of the two arguments.
     */
    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return CountingPotential::normalize(lhs);
    }
    /**
     * A normalized version of the times operation. Domain Safe.
     * @return     Normalized multiplication of the two arguments.
     */
    static inline ValType safe_times(ValType lhs, const ValType &rhs ) {
        lhs *= rhs;
        return CountingPotential::normalize(lhs);
    }

    /**
     * Normalizes a ValType to the specific domain of this Semiring.
     * @param  val ValType reference to normalize
     * @return     A reference to the modified argument after it is normalized
     *               to the correct domain.
     */
    static inline ValType& normalize(ValType& val) {
        return val = val < 0 ? 0 : val;
    }

    /**
     * The identity element of the semiring.
     * @return 1
     */
    static inline ValType one() { return 1; }
    /**
     * The annihlator element of the semiring.
     * @return 0
     */
    static inline ValType zero() { return 0; }

    /**
     * Provides a semi-random ValType that is within the domain specified
     * by the semiring.
     * @return A random value between 0 and infinity (here 1e8)
     */
    static inline ValType randValue() {
        return rand();
    }
};

/**
 * Comparison pair. *Experimental*
 * 
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
typedef vector<SparsePair> SparseVector;

/**
 * Sparse vector. *Experimental*
 *
 * +: Elementwise min
 * *: Elementwise +
 * 0: Empty Vector
 * 1: Empty Vector
 */
class SparseVectorPotential {
public:
    typedef SparseVector ValType;

    static inline ValType add(const ValType& lhs, const ValType& rhs) {
        return lhs;
    }

    static ValType times(const ValType& lhs, const ValType& rhs) {
        int i = 0, j = 0;
        SparseVector vec;
        while (i < lhs.size() || j < rhs.size()) {
          if (j >= rhs.size() || (i < lhs.size() && lhs[i].first < rhs[j].first)) {
            vec.push_back(pair<int, int>(lhs[i].first, lhs[i].second));
            ++i;
          } else if (i >= lhs.size() || (j < rhs.size() && lhs[i].first > rhs[j].first)) {
            vec.push_back(pair<int, int>(rhs[j].first, rhs[j].second));
            ++j;
          } else {
            vec.push_back(pair<int, int>(lhs[i].first, lhs[i].second + rhs[j].second));
            ++i;
            ++j;
          }
        }
        return vec;
    }

    static inline ValType safe_add(const ValType& lhs, const ValType& rhs) {
        return lhs;
    }

    static ValType safe_times(ValType lhs, const ValType& rhs) {
        lhs = SparseVectorPotential::times(lhs, rhs);
        return SparseVectorPotential::normalize(lhs);
    }

    static inline ValType one() { return ValType(); }
    static inline ValType zero() { return ValType(); }

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

/**
 * Tree. *Experimental*
 * Value of chart would be best backpointer.
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
                        const vector<V> &potentials,
                        V bias)
    : hypergraph_(hypergraph),
        potentials_(potentials),
        bias_(bias) {
            assert(potentials.size() == hypergraph->edges().size());
    }

    HypergraphPotentials(const Hypergraph *hypergraph)
        : hypergraph_(hypergraph),
            potentials_(hypergraph->edges().size(), SemiringType::one()),
            bias_(SemiringType::one()) {}

    V dot(const Hyperpath &path) const {
        path.check(*hypergraph_);
        V score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            score = SemiringType::times(score, potentials_[edge->id()]);
        }
        return SemiringType::times(score, bias_);
    }

    V score(HEdge edge) const { return potentials_[edge->id()]; }
    inline V operator[] (HEdge edge) const {
        return potentials_[edge->id()];
    }
    // inline V& operator[] (HEdge edge) {
    //  return potentials_[edge->id()];
    // }

    void insert(const HEdge& edge, const V& val) {
        potentials_[edge->id()] = val;
    }

    const V &bias() const { return bias_; }
    V &bias() { return bias_; }

    HypergraphPotentials<S> *project_potentials(
        const HypergraphProjection &projection) const;

    /**
     * Pairwise "times" with another set of potentials.
     *
     * @return New hypergraphpotentials.
     */
    HypergraphPotentials<S> *times(
            const HypergraphPotentials<S> &potentials) const;

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

    const Hypergraph *hypergraph() const { return hypergraph_; }

 protected:
    const Hypergraph *hypergraph_;
    vector<V> potentials_;
    V bias_;
};


class HypergraphProjection {
 public:
    HypergraphProjection(const Hypergraph *original,
                                             const Hypergraph *_new_graph,
                                             const vector<HNode> *node_map,
                                             const vector<HEdge> *edge_map)
            : original_graph(original),
            new_graph(_new_graph),
            node_map_(node_map),
            edge_map_(edge_map) {
                assert(node_map->size() == original_graph->nodes().size());
                assert(edge_map->size() == original_graph->edges().size());
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

    static HypergraphProjection *project_hypergraph(
            const Hypergraph *hypergraph,
            const HypergraphPotentials<BoolPotential> &edge_mask);

    HEdge project(HEdge original) const {
        return (*edge_map_)[original->id()];
    }

    HNode project(HNode original) const {
        return (*node_map_)[original->id()];
    }

    const Hypergraph *original_graph;
    const Hypergraph *new_graph;

 private:

    // Owned.
    const vector<HNode> *node_map_;
    const vector<HEdge> *edge_map_;
};

const HypergraphPotentials<LogViterbiPotential> *
pairwise_dot(const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
             const vector<double> &vec);


#endif // HYPERGRAPH_SEMIRING_H_
