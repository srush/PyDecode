// Copyright [2013] Alexander Rush

#include "Hypergraph/Semirings.h"

// These are not guaranteed to trigger, unless something is used in this file...
// Which is why I moved the operators below to here.
REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, ViterbiPotential);
REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, LogViterbiPotential);
REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, BoolPotential);
REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, InsidePotential);
REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, RealPotential);
REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, TropicalPotential);
REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, CountingPotential);
// REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, CompPotential<ViterbiPotential, LogViterbiPotential>);
// REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, SparseVectorPotential);
// REGISTRY_TYPE_DEFINITION(RandomSemiringRegistry, TreePotential);


// These are defined here while for the type registration above
// to work, there must be something in this file that is guaranteed
// to compile.
bool operator==(const BaseSemiring& lhs, const BaseSemiring& rhs) {
    return lhs.value == rhs.value;
}

BaseSemiring operator+(BaseSemiring lhs, const BaseSemiring &rhs) {
    lhs += rhs;
    return lhs;
}

BaseSemiring operator*(BaseSemiring lhs, const BaseSemiring &rhs) {
    lhs *= rhs;
    return lhs;
}


SparseVectorPotential& SparseVectorPotential::operator*=(const SparseVectorPotential& rhs) {
  int i = 0, j = 0;
  SparseVector vec;
  while (i < value.size() || j < rhs.value.size()) {
    if (j >= rhs.value.size() || (i < value.size() && value[i].first < rhs.value[j].first)) {
      vec.push_back(pair<int, int>(value[i].first, value[i].second));
      ++i;
    } else if (i >= value.size() || (j < rhs.value.size() && value[i].first > rhs.value[j].first)) {
      vec.push_back(pair<int, int>(rhs.value[j].first, rhs.value[j].second));
      ++j;
    } else {
      vec.push_back(pair<int, int>(i, value[i].second + rhs.value[j].second));
      ++i;
      ++j;
    }
  }
  value = vec;
  return *this;
}

SparseVector SparseVectorPotential::randValue() { 
    SparseVector randVec;
    int n = 20;
    for(int i = 0; i < n; i++) {
        randVec.push_back(SparsePair(rand(),rand()));
    }
    return randVec;
}

template <>
inline double HypergraphPotentials<double>::dot(const Hyperpath &path) const {
  path.check(*hypergraph_);
  double score = 0.0;
  foreach (HEdge edge, path.edges()) {
    score += potentials_[edge->id()];
  }
  return score + bias_;
}

template<typename SemiringType>
HypergraphPotentials<SemiringType> *HypergraphPotentials<SemiringType>::times(const HypergraphPotentials<SemiringType> &other) const {
  check(other);
  vector<SemiringType> new_potentials(potentials_);
  for (uint i = 0; i < other.potentials_.size(); ++i) {
    new_potentials[i] *= other.potentials_[i];
  }
  return new HypergraphPotentials<SemiringType>(hypergraph_,
                               new_potentials,
                               bias_ * other.bias_);
}

template<typename SemiringType>
HypergraphPotentials<SemiringType> *HypergraphPotentials<SemiringType>::project_potentials(
    const HypergraphProjection &projection) const {
  vector<SemiringType> potentials(projection.new_graph->edges().size());
  foreach (HEdge edge, projection.original_graph->edges()) {
    HEdge new_edge = projection.project(edge);
    if (new_edge != NULL && new_edge->id() >= 0) {
      assert(new_edge->id() < projection.new_graph->edges().size());
      potentials[new_edge->id()] = score(edge);
    }
  }
  return new HypergraphPotentials<SemiringType>(projection.new_graph, potentials, bias_);
}



inline HypergraphProjection *HypergraphProjection::project_hypergraph(
    const Hypergraph *hypergraph,
    const HypergraphPotentials<BoolPotential> &edge_mask) {
  vector<HNode> *node_map =
      new vector<HNode>(hypergraph->nodes().size(), NULL);
  vector<HEdge> *edge_map =
      new vector<HEdge>(hypergraph->edges().size(), NULL);

  Hypergraph *new_graph = new Hypergraph();
  foreach (HNode node, hypergraph->nodes()) {
    if (node->terminal()) {
      // The node is a terminal, so just add it.
      (*node_map)[node->id()] =
          new_graph->add_terminal_node(node->label());
    } else {
      (*node_map)[node->id()] = new_graph->start_node(node->label());

      // Try to add each of the edges of the node.
      foreach (HEdge edge, node->edges()) {
        if (!(bool)edge_mask[edge]) continue;
        vector<HNode> tails;
        bool all_tails_exist = true;
        foreach (HNode tail_node, edge->tail_nodes()) {
          HNode new_tail_node = (*node_map)[tail_node->id()];
          if (new_tail_node == NULL) {
            // The tail node was pruned.
            all_tails_exist = false;
            break;
          } else {
            tails.push_back(new_tail_node);
          }
        }
        if (all_tails_exist) {
          HEdge new_edge = new_graph->add_edge(tails, edge->label());
          (*edge_map)[edge->id()] = new_edge;
        }
      }
      bool success = true;
      if (!new_graph->end_node()) {
        (*node_map)[node->id()] = NULL;
        success = false;
      }
      if (hypergraph->root()->id() == node->id()) {
        assert(success);
      }
    }
  }
  new_graph->finish();
  return new HypergraphProjection(hypergraph, new_graph,
                                  node_map, edge_map);
}

inline const HypergraphPotentials<LogViterbiPotential> *
pairwise_dot(const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
             const vector<double> &vec) {
  HypergraphPotentials<LogViterbiPotential> *potentials =
      new HypergraphPotentials<LogViterbiPotential>(sparse_potentials.hypergraph());
  foreach (HEdge edge, sparse_potentials.hypergraph()->edges()) {
    SparseVector edge_constraints =
        static_cast<SparseVector>(sparse_potentials.score(edge));
    foreach (SparsePair pair, edge_constraints) {
      (*potentials)[edge] *=
          LogViterbiPotential(pair.second * vec[pair.first]);
    }
  }
  SparseVector bias_constraints =
      static_cast<SparseVector>(sparse_potentials.bias());
  foreach (SparsePair pair, bias_constraints) {
    potentials->bias() *= LogViterbiPotential(pair.second * vec[pair.first]);
  }
  return potentials;
};

