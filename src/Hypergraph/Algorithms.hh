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
#include "Hypergraph/Potentials.hh"

// Compute tail node set for each hypergraph node.
vector<set<int> > *children_nodes(const Hypergraph &graph);

struct IdComparator {
    bool operator()(HEdge edge1, HEdge edge2) const {
        return edge1 < edge2;
    }
};

struct DFANode {
  DFANode(int _left_state, int _right_state,
          HNode _node, int _id) :
    left_state(_left_state),
        right_state(_right_state),
        node(_node),
        id(_id) {}

    DFANode() {}

    int left_state;
    int right_state;
    HNode node;
    int id;
};

HypergraphMap *extend_with_dfa(
    Hypergraph *graph,
    const HypergraphPotentials<CountingPotential> &potentials,
    const DFA &dfa,
    vector<DFANode> *labels);

HypergraphMap *extend_hypergraph_by_count(
    Hypergraph *graph,
    const HypergraphPotentials<CountingPotential> &potentials,
    int lower_limit,
    int upper_limit,
    int goal);

/* Chart<SetPotential> *edge_domination(const Hypergraph &graph); */
/* Chart<SetPotential> *node_domination(const Hypergraph &graph); */

template<class Set1, class Set2>
bool is_disjoint(const Set1 &set1, const Set2 &set2);

vector<set<int> > *children_sparse(
    const Hypergraph *graph,
    const HypergraphPotentials<SparseVectorPotential> &potentials);

set<int> *updated_nodes(
    const Hypergraph *graph,
    const vector<set<int> > &children,
    const set<int> &updated);


HypergraphMap *project_hypergraph(
    const Hypergraph *hypergraph,
    const HypergraphPotentials<BoolPotential> &edge_mask);

HypergraphMap *binarize(const Hypergraph *hypergraph);

struct LatticeLabel {
    LatticeLabel() {}
  LatticeLabel(int i_, int j_) : i(i_), j(j_) {}
    int i, j;
};

Hypergraph *make_lattice(int width, int height,
                         const vector<vector<int> > &transitions,
                         vector<LatticeLabel> *labels);




#endif  // HYPERGRAPH_ALGORITHMS_H_
