// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_MAP_H_
#define HYPERGRAPH_MAP_H_

#include <vector>

#include "Hypergraph/Hypergraph.hh"
#include "Hypergraph/Potentials.hh"


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
        //assert(original->id() < edge_map_->size());
        return (*edge_map_)[domain_graph_->id(original)];
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
                new vector<HEdge>(domain_graph->edges().size(), -1);


        foreach (HNode node, range_graph->nodes()) {
            foreach (HNode new_node, reverse_node_map[node->id()]) {
                if (new_node->id() == -1) continue;
                (*node_map)[new_node->id()] = node;
            }
        }
        foreach (HEdge edge, range_graph->edges()) {
            foreach (HEdge new_edge, reverse_edge_map[domain_graph->id(edge)]) {
                if (domain_graph->new_id(new_edge) == -1) continue;
                (*edge_map)[domain_graph->new_id(new_edge)] = edge;
            }
        }
        return new HypergraphMap(domain_graph, range_graph,
                                 node_map, edge_map, false);
    }


    const vector<int> &edge_map() const {
        return *edge_map_;
    }

    const vector<HNode> &node_map() const {
        return *node_map_;
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
