// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRINGS_H_
#define HYPERGRAPH_SEMIRINGS_H_

#include <algorithm>
#include <bitset>
#include <utility>
#include <vector>
#include <set>
#include <cmath>

#include "Hypergraph/Hypergraph.hh"
#include "./common.h"

class Viterbi {
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

    static inline ValType zero() { return 0.0; }
    static inline ValType one() { return 1.0; }
};

class LogViterbi {
  public:
    typedef double ValType;
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs = std::max(lhs, rhs);
        return lhs;
    }

    static inline ValType zero() { return -INF; }
    static inline ValType one() { return 0.0; }

    static bool valid(const ValType &lhs, const ValType &rhs) {
        return true;
    }
};


class Real {
  public:
    typedef double ValType;
    static inline ValType add(ValType lhs, const ValType &rhs) {
        lhs += rhs;
        return lhs;
    }
    static inline ValType times(ValType lhs, const ValType &rhs) {
        lhs *= rhs;
        return lhs;
    }

    static inline ValType zero() { return 0.0; }
    static inline ValType one() { return 1.0; }

};


class Log {
  public:
    typedef double ValType;
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
};

class Boolean {
  public:
    typedef char ValType;
    static inline ValType add(const ValType& lhs, const ValType &rhs) {
        return lhs || rhs;
    }
    static inline ValType times(const ValType& lhs, const ValType &rhs) {
        return lhs && rhs;
    }

    static inline ValType one() { return true; }
    static inline ValType zero() { return false; }
};

class Counting {
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

    static inline ValType one() { return 1; }
    static inline ValType zero() { return 0; }
};

class MinMax {
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

    static inline ValType one() { return INF; }
    static inline ValType zero() { return -INF; }
};

#endif  // HYPERGRAPH_SEMIRINGS_H_
