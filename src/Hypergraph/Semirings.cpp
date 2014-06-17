// Copyright [2013] Alexander Rush

#include "Hypergraph/Semirings.hh"
#include <utility>
#include <vector>


RealPotential::ValType RealPotential::INFINITY_BUFFER = INF/8;
RealPotential::ValType RealPotential::NEGATIVE_INFINITY = -INF/10;

bool valid_binary_vectors(const bitset<BITMAPSIZE> &lhs,
                          const bitset<BITMAPSIZE> &rhs) {
    return ((lhs & rhs).none());
}


SparseVector combine_sparse_vectors(const SparseVector &value,
                                    const SparseVector &rhs,
                                    const Operator &op) {
    int i = 0, j = 0;
    SparseVector vec;
    while (i < value.size() || j < rhs.size()) {
        if (j >= rhs.size() ||
            (i < value.size() && value[i].first < rhs[j].first)) {
            int val = op(0, value[i].second);
            if (val != 0) {
                vec.push_back(
                    pair<int, int>(value[i].first, val));
            }
            ++i;
        } else if (i >= value.size() ||
                   (j < rhs.size() && value[i].first > rhs[j].first)) {
            int val = op(0, rhs[j].second);
            if (val != 0) {
                vec.push_back(
                    pair<int, int>(rhs[j].first,
                                   val));
            }
            ++j;
        } else {
            int val = op(value[i].second, rhs[j].second);
            if (val != 0) {
                vec.push_back(
                    pair<int, int>(value[i].first,
                                   val));
            }
            ++i;
            ++j;
        }
    }
    return vec;
}
