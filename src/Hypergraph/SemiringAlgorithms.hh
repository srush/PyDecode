// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRINGALGORITHMS_H_
#define HYPERGRAPH_SEMIRINGALGORITHMS_H_

#include <queue>
#include <set>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.hh"
#include "Hypergraph/Semirings.hh"

template<typename SemiringType>
void general_inside(
    const Hypergraph *graph,
    const typename SemiringType::ValType *weights,
    typename SemiringType::ValType *chart);

template<typename SemiringType>
void general_outside(
    const Hypergraph *graph,
    const typename SemiringType::ValType *weights,
    const typename SemiringType::ValType *inside_chart,
    typename SemiringType::ValType *chart);

template<typename SemiringType>
void general_viterbi(
    const Hypergraph *graph,
    const typename SemiringType::ValType *potentials,
    typename SemiringType::ValType *chart,
    int *back_pointers,
    bool *mask);

template<typename SemiringType>
void transform(const Hypergraph *graph,
               const typename SemiringType::ValType *weights,
               const int *labeling,
               typename SemiringType::ValType *label_weights,
               int label_size);

template<typename SemiringType>
void general_kbest(
    const Hypergraph *graph,
    const typename SemiringType::ValType *weights,
    int K,
    vector<Hyperpath *> *);

template<typename S>
void node_marginals(const Hypergraph *hypergraph,
                    const typename S::ValType *in_chart,
                    const typename S::ValType *out_chart,
                    typename S::ValType *node_marginals);

template<typename S>
void edge_marginals(const Hypergraph *hypergraph,
                    const typename S::ValType *weights,
                    const typename S::ValType *in_chart,
                    const typename S::ValType *out_chart,
                    typename S::ValType *edge_marginals);

#endif  // HYPERGRAPH_SEMIRINGALGORITHMS_H_
