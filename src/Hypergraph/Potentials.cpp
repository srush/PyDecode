// Copyright [2013] Alexander Rush

#include "Hypergraph/Potentials.h"

#define SPECIALIZE_HYPER_FOR_SEMI(X)\
    template class HypergraphPotentials<X>;\
    template class HypergraphMapPotentials<X>;\
    template class HypergraphProjectedPotentials<X>;\
    template class HypergraphVectorPotentials<X>;


template<typename SemiringType>
HypergraphPotentials<SemiringType> *
HypergraphVectorPotentials<SemiringType>::times(
    const HypergraphPotentials<SemiringType> &other) const {
    HypergraphPotentials<SemiringType>::check(other);
    vector<typename SemiringType::ValType> new_potentials(potentials_);
    int i = -1;
    foreach (HEdge edge, this->hypergraph_->edges()) {
        i++;
        new_potentials[i] = SemiringType::times(new_potentials[i],
                                                other[edge]);
    }
    return new HypergraphVectorPotentials<SemiringType>(
        this->hypergraph_,
        new_potentials,
        SemiringType::times(this->bias_, other.bias_));
}

template<typename SemiringType>
HypergraphPotentials<SemiringType> *
HypergraphPotentials<SemiringType>::project_potentials(
        const HypergraphProjection &projection) const {
    vector<typename SemiringType::ValType>
            potentials(projection.new_graph()->edges().size());
    foreach (HEdge edge, projection.big_graph()->edges()) {
        HEdge new_edge = projection.project(edge);
        if (new_edge != NULL && new_edge->id() >= 0) {
            assert(new_edge->id() < projection.new_graph()->edges().size());
            potentials[new_edge->id()] = score(edge);
        }
    }
    return new HypergraphVectorPotentials<SemiringType>(
        projection.new_graph(), potentials, bias_);
}

HypergraphProjection *HypergraphProjection::project_hypergraph(
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
                                    node_map, edge_map, true);
}

template<typename SemiringType>
HypergraphPotentials<SemiringType> *
HypergraphMapPotentials<SemiringType>::times(
    const HypergraphPotentials<SemiringType> &other) const {
    HypergraphPotentials<SemiringType>::check(other);
    vector<typename SemiringType::ValType> new_potentials(potentials_);
    int i = -1;
    foreach (HEdge edge, this->hypergraph_->edges()) {
        i++;
        new_potentials[i] = SemiringType::times(new_potentials[i],
                                                other[edge]);
    }
    return new HypergraphVectorPotentials<SemiringType>(
        this->hypergraph_,
        new_potentials,
        SemiringType::times(this->bias_, other.bias_));
}

void pairwise_dot(
    const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
    const vector<double> &vec,
    HypergraphPotentials<LogViterbiPotential> *weights) {
    int i = 0;
    vector<double> &pots = weights->potentials();
    foreach (SparseVector &edge_constraints, sparse_potentials.potentials()) {
        foreach (SparsePair &pair, edge_constraints) {
            pots[i] =
                    LogViterbiPotential::times(pots[i],
                                               pair.second * vec[pair.first]);
        }
        ++i;
    }
    SparseVector bias_constraints = sparse_potentials.bias();
    LogViterbiPotential::ValType &bias = weights->bias();
    foreach (SparsePair &pair, bias_constraints) {
        bias = LogViterbiPotential::times(
            weights->bias(),
            LogViterbiPotential::ValType(pair.second * vec[pair.first]));
    }
};

void non_zero_weights(const Hypergraph *graph,
                      const HypergraphPotentials<LogViterbiPotential> &weights,
                      HypergraphVectorPotentials<BoolPotential> *updates) {
    foreach (HEdge edge, graph->edges()) {
        updates->insert(edge, weights[edge] != 0.0);
    }
}

SPECIALIZE_HYPER_FOR_SEMI(ViterbiPotential)
SPECIALIZE_HYPER_FOR_SEMI(LogViterbiPotential)
SPECIALIZE_HYPER_FOR_SEMI(InsidePotential)
SPECIALIZE_HYPER_FOR_SEMI(BoolPotential)
SPECIALIZE_HYPER_FOR_SEMI(SparseVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(MinSparseVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(MaxSparseVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(BinaryVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(CountingPotential)
SPECIALIZE_HYPER_FOR_SEMI(SetPotential)
