#include "Hypergraph/Map.hh"

#include <vector>

HypergraphMap::HypergraphMap(const Hypergraph *domain_graph,
                             const Hypergraph *range_graph,
                             const vector<HNode> *node_map,
                             const vector<HEdge> *edge_map,
                             bool bidirectional)
        : domain_graph_(domain_graph),
          range_graph_(range_graph),
          node_map_(node_map),
          edge_map_(edge_map),
          bidirectional_(bidirectional) {
    assert(node_map->size() == domain_graph_->nodes().size());
    assert(edge_map->size() == domain_graph_->edges().size());

#ifndef NDEBUG
    foreach (HNode node, *node_map) {
        assert(node == NULL ||
               node->id() < range_graph->nodes().size());
    }
    foreach (HEdge edge, *edge_map) {
        assert(edge == -1 ||
               range_graph->id(edge) < range_graph->edges().size());
    }
#endif
}

HypergraphMap::~HypergraphMap() {
    delete node_map_;
    delete edge_map_;
    node_map_ = NULL;
    edge_map_ = NULL;
}

HypergraphMap *HypergraphMap::compose(const HypergraphMap &other) const {
    if (!other.range_graph()->same(*domain_graph())) {
      throw HypergraphException("Hypergraphs do not match.");
    }

    vector<HEdge> *edge_map =
            new vector<HEdge>(other.domain_graph()->edges().size(), NULL);
    vector<HNode> *node_map =
            new vector<HNode>(other.domain_graph()->nodes().size(), NULL);
    foreach (HEdge edge, other.domain_graph()->edges()) {
        HEdge proj = other.map(edge);
        if (proj >= 0) {
            (*edge_map)[edge] = map(proj);
        }
    }

    foreach (HNode node, other.domain_graph()->nodes()) {
        HNode proj = other.map(node);
        if (proj != NULL && proj->id() >= 0) {
            (*node_map)[node->id()] = map(proj);
        }
    }

    return new HypergraphMap(other.domain_graph(),
                             range_graph(),
                             node_map, edge_map,
                             bidirectional_ && other.bidirectional_);
}

HypergraphMap *HypergraphMap::invert() const {
    assert(bidirectional_);
    vector<HNode> *node_reverse_map_ =
            new vector<HNode>(range_graph_->nodes().size(), NULL);
    vector<HEdge> *edge_reverse_map_ =
            new vector<HEdge>(range_graph_->edges().size(), NULL);

    foreach (HNode node, domain_graph()->nodes()) {
        HNode mapped_node = (*node_map_)[node->id()];
        if (mapped_node == NULL || mapped_node->id() < 0) continue;
        (*node_reverse_map_)[mapped_node->id()] = node;
    }
    foreach (HEdge edge, domain_graph()->edges()) {
        HEdge mapped_edge = (*edge_map_)[edge];
        if (mapped_edge < 0) continue;
        (*edge_reverse_map_)[mapped_edge] = edge;
    }
    return new HypergraphMap(range_graph_, domain_graph_,
                             node_reverse_map_, edge_reverse_map_, true);
}
