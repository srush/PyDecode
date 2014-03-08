// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_MAP_H_
#define HYPERGRAPH_MAP_H_

#include <vector>

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Potentials.h"


class HypergraphMap {
 public:
    HypergraphMap(const Hypergraph *domain_graph,
                  const Hypergraph *range_graph,
                  const vector<HNode> *node_map,
                  const vector<HEdge> *edge_map,
                  bool bidirectional);

    ~HypergraphMap();

    HypergraphMap *compose(const HypergraphMap &projection) const;

    HypergraphMap *invert() const;

    Hyperpath *map(const Hyperpath &domain_path) const {
        domain_path.check(*domain_graph());
        vector<HEdge> edges;
        foreach (HEdge edge, domain_path.edges()) {
            edges.push_back(map(edge));
        }
        return new Hyperpath(range_graph(), edges);
    }

    HEdge map(HEdge original) const {
        assert(original->id() < edge_map_->size());
        return (*edge_map_)[original->id()];
    }

    HNode map(HNode original) const {
        assert(original->id() < node_map_->size());
        return (*node_map_)[original->id()];
    }

    const Hypergraph *domain_graph() const {
        return domain_graph_;
    }

    const Hypergraph *range_graph() const {
        return range_graph_;
    }

 private:
    const Hypergraph *domain_graph_;
    const Hypergraph *range_graph_;

    // Owned.
    const vector<HNode> *node_map_;
    const vector<HEdge> *edge_map_;

    bool bidirectional_;
};

#endif  // HYPERGRAPH_MAP_H_
