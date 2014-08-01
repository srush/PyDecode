// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRINGALGORITHMS_H_
#define HYPERGRAPH_SEMIRINGALGORITHMS_H_

#include <queue>
#include <set>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.hh"
#include "Hypergraph/Semirings.hh"


// class BackPointers {
//   public:
//     explicit BackPointers(const Hypergraph *graph,
//                           HEdge *chart)
//             : graph_(graph), chart_(chart) {
//         // chart_ = new HEdge[graph_->nodes().size()];
//         for (int i = 0; i < graph_->nodes().size(); ++i){
//             chart_[i] = -1;
//         }
//     }

//     inline HEdge operator[] (HNode node) const {
//         return chart_[node];
//     }

//     HEdge get(HNode node) const { return chart_[node]; }
//     inline void insert(HNode node, HEdge val) {
//         chart_[node] = val;
//     }

//     void check(const Hypergraph *hypergraph) const {
//         if (!hypergraph->same(*graph_)) {
//             throw HypergraphException(
//                 "Hypergraph does not match backpointers.");
//         }
//     }

//     Hyperpath *construct_path() const;

//     HEdge *chart() { return chart_; }

//   protected:
//     const Hypergraph *graph_;
//     HEdge *chart_;
// };


/* class KBackPointers { */
/*   public: */
/*     explicit KBackPointers(const Hypergraph *graph) */
/*             : graph_(graph), */
/*         chart_(graph_->nodes().size(), -1) {} */

/*     inline HEdge operator[] (HNode node) const { */
/*         return chart_[node]; */
/*     } */

/*     HEdge get(HNode node) const { return chart_[node]; } */
/*     inline void insert(HNode node, HEdge val) { */
/*         chart_[node] = val; */
/*     } */

/*     void check(const Hypergraph *hypergraph) const { */
/*         if (!hypergraph->same(*graph_)) { */
/*             throw HypergraphException( */
/*                 "Hypergraph does not match backpointers."); */
/*         } */
/*     } */

/*     Hyperpath *construct_path() const; */

/*   protected: */
/*     const Hypergraph *graph_; */
/*     vector<HEdge> chart_; */
/* }; */



template<typename SemiringType>
void general_inside(
    const Hypergraph *graph,
    const typename SemiringType::ValType *potentials,
    typename SemiringType::ValType *chart);

template<typename SemiringType>
void general_outside(
    const Hypergraph *graph,
    const typename SemiringType::ValType *potentials,
    const typename SemiringType::ValType *inside_chart,
    typename SemiringType::ValType *chart);

template<typename SemiringType>
void general_viterbi(
    const Hypergraph *graph,
    const typename SemiringType::ValType *potentials,
    typename SemiringType::ValType *chart,
    int *back_pointers,
    bool *mask);

// template<typename S>
// Hyperpath *count_constrained_viterbi(
//     const Hypergraph *graph,
//     const HypergraphPotentials<S> &weight_potentials,
//     const HypergraphPotentials<CountingPotential> &count_potentials,
//     int limit);


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
