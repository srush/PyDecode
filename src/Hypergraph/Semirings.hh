// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRINGS_H_
#define HYPERGRAPH_SEMIRINGS_H_

#include <algorithm>
#include <bitset>
#include <utility>
#include <vector>
#include <set>
#include <cmath>

#include "Hypergraph/Factory.hh"
#include "Hypergraph/Hypergraph.hh"
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

    static inline ValType &normalize(ValType &val) {
        if (val < 0.0) val = 0.0;
        else if (val > 1.0) val = 1.0;
        return val;
    }

    static inline ValType zero() { return 0.0; }
    static inline ValType one() { return 1.0; }

    static inline ValType randValue() { return dRand(0.0, 1.0); }
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

    static inline ValType &normalize(ValType &val) {
        return val = val < -INF ? -INF : val;
    }

    static inline bool valid(const ValType& lhs,
                             const ValType& rhs) {return true;}

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


class LogProbPotential : public ViterbiPotential {
  public:
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }
    static inline ValType zero() { return -INF; }
    static inline ValType one() { return 0.0; }

    static inline ValType add(ValType lhs, const ValType &rhs) {
        ValType pi = max(lhs, rhs);
        ValType pi2 = min(lhs, rhs);
        lhs = pi + log(1 + exp(pi2 - pi));
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        ValType pi = max(lhs, rhs);
        ValType pi2 = min(lhs, rhs);
        lhs = pi + log(1 + exp(pi2 - pi));
        return LogProbPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return LogProbPotential::normalize(lhs);
    }


    static inline ValType &normalize(ValType &val) {
        return val = val < -INF ? -INF : val;
    }

    static inline ValType randValue() { return dRand(-INF, 0.0); }
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

    static inline ValType &normalize(ValType &val) {
        if (val < NEGATIVE_INFINITY) val = NEGATIVE_INFINITY;
        else if (val > INF - INFINITY_BUFFER) val = INF;
        return val;
    }

    static ValType randValue() {
        return dRand(NEGATIVE_INFINITY, INF - INFINITY_BUFFER);
    }

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

    static inline ValType &normalize(ValType &val) {
        return val = val > INF ? INF : val < 0.0 ? 0.0 : val;
    }

    static ValType randValue() { return dRand(0.0, INF); }
};

class BoolPotential {
  public:
    typedef char ValType;

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

    static inline ValType &normalize(ValType &val) {
        return val;
    }

    static inline ValType one() { return true; }
    static inline ValType zero() { return false; }

    static inline ValType randValue() { return dRand(0.0, 1.0) > .5; }
};

class SetPotential {
  public:
    typedef set<int> ValType;

    static inline ValType add(const ValType& lhs, const ValType &rhs) {
        set<int> intersection;
        set_intersection(lhs.begin(), lhs.end(), rhs.begin(),
                         rhs.end(),
                         inserter(intersection, intersection.begin()));
        return intersection;
    }
    static inline ValType times(const ValType& lhs, const ValType &rhs) {
        set<int> union_;
        set_union(lhs.begin(), lhs.end(), rhs.begin(),
                  rhs.end(), inserter(union_, union_.begin()));
        return union_;
    }

    static inline ValType &normalize(ValType &val) {
        return val;
    }

    static inline ValType one() { return set<int>(); }
    static inline ValType zero() { return set<int>(); }
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

    static inline const ValType &normalize(ValType &val) {
        return val = val < 0 ? 0 : val;
    }

    static inline ValType one() { return 1; }
    static inline ValType zero() { return 0; }

    static inline ValType randValue() {
        return rand();
    }
};

class MinMaxPotential {
  public:
    typedef double ValType;

    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs = min(lhs, rhs);
        return lhs;
    }
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs = max(lhs, rhs);
        return lhs;
    }

    static inline ValType safe_add(ValType lhs, const ValType &rhs) {
        lhs = min(lhs, rhs);
        return MinMaxPotential::normalize(lhs);
    }
    static inline ValType safe_times(ValType lhs, const ValType &rhs ) {
        lhs = max(lhs, rhs);
        return MinMaxPotential::normalize(lhs);
    }

    static inline const ValType &normalize(ValType &val) {
        return val;
    }

    static inline ValType one() { return INF; }
    static inline ValType zero() { return -INF; }

    static inline ValType randValue() {
        return rand();
    }
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
    typedef std::pair<typename SemiringFirst::ValType,
            typename SemiringSecond::ValType> ValType;

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

    static inline ValType one() {
        return ValType(SemiringFirst::one(), SemiringSecond::one());
    }
    static inline ValType zero() {
        return ValType(SemiringFirst::zero(), SemiringSecond::zero());
    }

    static inline ValType &normalize(ValType &val) {
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
        for (int i = 0; i < 20; i++) {
            randVec.push_back(SparsePair(rand(), rand()));
        }
        return randVec;
    }

    static inline ValType &normalize(ValType &val) {
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

    static inline ValType randValue() {
        return ValType(dRand(0, 0xfffffff));
    }

    static inline ValType &normalize(ValType &val) {
        return val;
    }

    static inline bool valid(const ValType& lhs,
                             const ValType& rhs) {
        return ((lhs & rhs).none());
    }
};


class AlphabetPotential {
  public:
    typedef vector<int> ValType;
    static const int kSize = 54;
    static inline ValType add(ValType lhs, const ValType& rhs) {
        // not used.
        return lhs;
    }

    /* static inline bool equals(const ValType &lhs, const ValType &rhs) { */
    /*     // not used. */
    /*     return lhs; */
    /* } */

    static inline ValType times(ValType value, const ValType& rhs) {
        for (int i = 0; i < kSize; ++i) {
            if (rhs[i] != -1)
                value[i] = rhs[i];
        }
        return value;
    }

    static inline ValType one() {
        ValType vec(kSize, -1);
        return vec;
    }

    static inline ValType zero() {
        // not used.
        ValType vec(kSize, -1);
        return vec;
    }

    static inline ValType randValue() {
        return ValType(0, 0);
    }

    static inline ValType &normalize(ValType &val) {
        return val;
    }

    static inline bool valid(const ValType& lhs,
                             const ValType& rhs) {
        for (int i = 0; i < kSize; ++i) {
            if (lhs[i] != -1 && rhs[i] != -1 && lhs[i] != rhs[i]) {
                return false;
            }
            /* for (int j = 0; j < kSize; ++j) { */
            /*     if (i != j && lhs[j] != -1 && rhs[i] != -1 && lhs[j] == rhs[i]) { */
            /*         return false; */
            /*     } */
            /* } */
        }
        return true;
    }
};



bool valid_binary_vectors(const bitset<BITMAPSIZE> &lhs,
                          const bitset<BITMAPSIZE> &rhs);

#endif  // HYPERGRAPH_SEMIRINGS_H_
