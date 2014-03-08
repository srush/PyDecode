// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_ALGORITHMS_H_
#define HYPERGRAPH_ALGORITHMS_H_

#include <queue>
#include <set>
#include <vector>

#include "./common.h"

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "Hypergraph/SemiringAlgorithms.h"
#include "Hypergraph/Potentials.h"

// Compute tail node set for each hypergraph node.
vector<set<int> > *children_nodes(const Hypergraph &graph);

struct IdComparator {
    bool operator()(HEdge edge1, HEdge edge2) const {
        return edge1->id() < edge2->id();
    }
};

HypergraphMap *extend_hypergraph_by_count(
    Hypergraph *graph,
    const HypergraphPotentials<CountingPotential> &potentials,
    int lower_limit,
    int upper_limit,
    int goal);

Chart<SetPotential> *edge_domination(const Hypergraph &graph);
Chart<SetPotential> *node_domination(const Hypergraph &graph);

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

#endif  // HYPERGRAPH_ALGORITHMS_H_
