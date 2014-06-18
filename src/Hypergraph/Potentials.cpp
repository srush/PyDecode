// Copyright [2013] Alexander Rush

#include <vector>

#include "Hypergraph/Potentials.hh"

#define SPECIALIZE_HYPER_FOR_SEMI(X)\
    template class HypergraphPotentials<X>;\
    template class HypergraphPointerPotentials<X>; \
    template class HypergraphSparsePotentials<X>;\
    template class HypergraphMappedPotentials<X>;\
    template class HypergraphVectorPotentials<X>;


// template<typename SemiringType>
// HypergraphPotentials<SemiringType> *
// HypergraphVectorPotentials<SemiringType>::times(
//     const HypergraphPotentials<SemiringType> &other) const {
//     HypergraphPotentials<SemiringType>::check(other);
//     vector<typename SemiringType::ValType> *new_potentials
//             = new vector<typename SemiringType::ValType>(*potentials_);
//     int i = -1;
//     foreach (HEdge edge, this->hypergraph_->edges()) {
//         i++;
//         (*new_potentials)[i] = SemiringType::times(
//             (*new_potentials)[i],
//             other[edge]);
//     }
//     return new HypergraphVectorPotentials<SemiringType>(
//         this->hypergraph_,
//         new_potentials,
//         SemiringType::times(this->bias_, other.bias_),
//         false);
// }

template<typename SemiringType>
HypergraphPotentials<SemiringType> *
HypergraphPotentials<SemiringType>::project_potentials(
        const HypergraphMap &projection) const {
    check(*projection.domain_graph());
    vector<typename SemiringType::ValType> *potentials =
            new vector<typename SemiringType::ValType>(
                projection.range_graph()->edges().size());

    foreach (HEdge edge, projection.domain_graph()->edges()) {
        HEdge new_edge = projection.map(edge);
        if (new_edge >= 0) {
            assert(new_edge->id() < projection.range_graph()->edges().size());
            (*potentials)[projection.range_graph()->id(new_edge)] = score(edge);
        }
    }
    return new HypergraphVectorPotentials<SemiringType>(
        projection.range_graph(), potentials, false);
}

// template<typename SemiringType>
// HypergraphPotentials<SemiringType> *
// HypergraphSparsePotentials<SemiringType>::times(
//     const HypergraphPotentials<SemiringType> &other) const {
//     HypergraphPotentials<SemiringType>::check(other);
//     vector<typename SemiringType::ValType> *new_potentials=
//             new vector<typename SemiringType::ValType>(potentials_);
//     int i = -1;
//     foreach (HEdge edge, this->hypergraph_->edges()) {
//         i++;
//         (*new_potentials)[i] = SemiringType::times(
//             (*new_potentials)[i],
//             other[edge]);
//     }
//     return new HypergraphVectorPotentials<SemiringType>(
//         this->hypergraph_,
//         new_potentials,
//         SemiringType::times(this->bias_, other.bias_),
//         false);
// }

template<typename SemiringType>
HypergraphMappedPotentials<SemiringType>::HypergraphMappedPotentials(
    HypergraphPotentials<SemiringType> *base_potentials,
    const HypergraphMap *projection)
        : HypergraphPotentials<SemiringType>(
            projection->domain_graph()),
          base_potentials_(base_potentials),
          projection_(projection) {
    base_potentials->check(*projection->range_graph());
}

// template<typename SemiringType>
// HypergraphPotentials<SemiringType> *
// HypergraphMappedPotentials<SemiringType>::times(
//     const HypergraphPotentials<SemiringType> &other) const {
//     vector<typename SemiringType::ValType> *new_potentials =
//             new vector<typename SemiringType::ValType>(
//                 projection_->range_graph()->edges().size());
//     int i = -1;
//     foreach (HEdge edge, projection_->range_graph()->edges()) {
//         i++;
//         (*new_potentials)[i] = SemiringType::times(
//             base_potentials_->score(edge),
//             other.score(edge));
//     }
//     return new HypergraphMappedPotentials<SemiringType>(
//         new HypergraphVectorPotentials<SemiringType>(
//             projection_->range_graph(),
//             new_potentials,
//             SemiringType::times(this->bias_, other.bias()),
//             false),
//         projection_);
// }

template<typename SemiringType>
typename SemiringType::ValType
HypergraphMappedPotentials<SemiringType>::score(HEdge edge) const {
    HEdge new_edge = projection_->map(edge);
    return base_potentials_->score(new_edge);
}

// void pairwise_dot(
//     const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
//     const vector<double> &vec,
//     HypergraphPotentials<LogViterbiPotential> *weights) {
//     int i = 0;
//     vector<double> &pots = weights->potentials();
//     foreach (const SparseVector &edge_constraints,
//              sparse_potentials.potentials()) {
//         foreach (const SparsePair &pair, edge_constraints) {
//             if (vec[pair.first] != 0 && pair.second != 0) {
//                 pots[i] =
//                   LogViterbiPotential::times(pots[i],
//                                              pair.second * vec[pair.first]);
//             }
//         }
//         ++i;
//     }
//     SparseVector bias_constraints = sparse_potentials.bias();
//     LogViterbiPotential::ValType &bias = weights->bias();
//     foreach (const SparsePair &pair, bias_constraints) {
//         bias = LogViterbiPotential::times(
//             weights->bias(),
//             LogViterbiPotential::ValType(pair.second * vec[pair.first]));
//     }
// };

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
SPECIALIZE_HYPER_FOR_SEMI(MinMaxPotential)
SPECIALIZE_HYPER_FOR_SEMI(BoolPotential)
SPECIALIZE_HYPER_FOR_SEMI(LogProbPotential)
SPECIALIZE_HYPER_FOR_SEMI(SparseVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(MinSparseVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(MaxSparseVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(BinaryVectorPotential)
SPECIALIZE_HYPER_FOR_SEMI(CountingPotential)
SPECIALIZE_HYPER_FOR_SEMI(SetPotential)
SPECIALIZE_HYPER_FOR_SEMI(AlphabetPotential)
