// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <set>
#include <vector>

#include "./common.h"

#include "Hypergraph/Automaton.hh"
#include "Hypergraph/Hypergraph.hh"
#include "Hypergraph/Semirings.hh"
#include "Hypergraph/SemiringAlgorithms.hh"

Hypergraph *filter(const Hypergraph *graph,
                   const bool *edge_mask);

Hypergraph *binarize(const Hypergraph *graph);

Hypergraph *intersect(Hypergraph *graph,
                      const int *labels,
                      const DFA &dfa);

Hypergraph *intersect_count(Hypergraph *graph,
                            const int *edge_counts,
                            int lower_limit,
                            int upper_limit,
                            int goal);


/* Chart<SetPotential> *edge_domination(const Hypergraph &graph); */
/* Chart<SetPotential> *node_domination(const Hypergraph &graph); */
// vector<set<int> > *children_nodes(const Hypergraph &graph);

// vector<set<int> > *children_sparse(
//     const Hypergraph *graph,
//     const HypergraphPotentials<SparseVectorPotential> &potentials);

// set<int> *updated_nodes(
//     const Hypergraph *graph,
//     const vector<set<int> > &children,
//     const set<int> &updated);





// struct LatticeLabel {
//     LatticeLabel() {}
//   LatticeLabel(int i_, int j_) : i(i_), j(j_) {}
//     int i, j;
// };

// Hypergraph *make_lattice(int width, int height,
//                          const vector<vector<int> > &transitions,
//                          vector<LatticeLabel> *labels);




#endif  // HYPERGRAPH_ALGORITHMS_H_
