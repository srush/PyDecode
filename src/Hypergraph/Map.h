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

    static HypergraphMap *make_reverse_map(
        const Hypergraph *domain_graph,
        const Hypergraph *range_graph,
        const vector<vector<HNode> > &reverse_node_map,
        const vector<vector<HEdge> > &reverse_edge_map) {

        // Create node maps.
        vector<HNode> *node_map =
                new vector<HNode>(domain_graph->nodes().size(), NULL);
        vector<HEdge> *edge_map =
                new vector<HEdge>(domain_graph->edges().size(), NULL);


        foreach (HNode node, range_graph->nodes()) {
            foreach (HNode new_node, reverse_node_map[node->id()]) {
                if (new_node->id() == -1) continue;
                (*node_map)[new_node->id()] = node;
            }
        }
        foreach (HEdge edge, range_graph->edges()) {
            foreach (HEdge new_edge, reverse_edge_map[edge->id()]) {
                if (new_edge->id() == -1) continue;
                (*edge_map)[new_edge->id()] = edge;
            }
        }
        return new HypergraphMap(domain_graph, range_graph,
                                 node_map, edge_map, false);
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
