// Copyright [2013] Alexander Rush


#include "Hypergraph/BeamSearch.hh"

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>
#include <queue>
#include "./common.h"
#include "Hypergraph/Semirings.hh"
#include <boost/intrusive/rbtree.hpp>

using namespace boost::intrusive;

template<typename BVP>
BeamChart<BVP>::~BeamChart<BVP>() {
    for (int i = 0; i < beam_.size(); ++i) {
        typename Beam::iterator iter = beam_[i]->begin();
        beam_[i]->clear_and_dispose(delete_disposer());
        delete beam_[i];
    }

    while (!hyp_pool_.empty()) {
        BeamHyp *hyp;
        hyp = hyp_pool_.top();
        delete hyp;
        hyp_pool_.pop();
    }
}

template<typename BVP>
BeamChart<BVP> *BeamChart<BVP>::cube_pruning(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<BVP> &constraints,
    const Chart<LogViterbiPotential> &future,
    double lower_bound,
    const BeamGroups &groups,
    bool recombine) {

    // Check the inputs.
    potentials.check(*graph);
    constraints.check(*graph);
    groups.check(graph);

    typedef LogViterbiPotential LVP;

    BeamChart<BVP> *chart = new BeamChart<BVP>(graph, &groups,
                                               &potentials,
                                               &constraints,
                                               &future,
                                               lower_bound,
                                               recombine);

    // 1) Initialize the chart with terminal nodes.
    typename BVP::ValType one = BVP::one();


    foreach (HNode node, graph->nodes()) {
        if (node->terminal()) {
            chart->insert(node, NULL, one, LVP::one(),
                          -1, -1);
            continue;
        }
    }

    for (int group = 0; group < groups.groups_size(); ++group) {
        priority_queue<BeamHyp> pqueue;
        foreach (HNode node, groups.group_nodes(group)) {
            // 2) Enumerate over each edge (in topological order).
            foreach (HEdge edge, node->edges()) {
                chart->queue_up(node, edge, 0, 0, &pqueue);
            }
            for (int i = 0; i < groups.group_limit(group); ++i) {
                BeamHyp hyp = pqueue.top();
                pqueue.pop();
                bool unary = graph->tail_nodes(hyp.edge) == 1;
                HEdge edge = hyp.edge;
                HNode node_left = graph->tail_node(edge, 0);

                stack<pair<int, int> > inner_queue;
                inner_queue.push(pair<int, int>(hyp.back_position_left+1, hyp.back_position_right));
                if (!unary) {
                    inner_queue.push(pair<int, int>(hyp.back_position_left, hyp.back_position_right+1));
                }

                while (!inner_queue.empty()) {
                    pair<int, int> pos = inner_queue.top();
                    inner_queue.pop();

                    if (pos.first < chart->get_beam(node_left).size() - 1 &&
                        (!unary ||
                         pos.second < chart->get_beam(graph->tail_node(edge, 1)).size() - 1)) {
                        bool attempt = chart->queue_up(hyp.node, hyp.edge,
                                                   pos.first,
                                                   pos.second,
                                                   &pqueue);
                        if (!attempt) {
                            inner_queue.push(pair<int, int>(pos.first+1, pos.second));
                            if (!unary) {
                                inner_queue.push(pair<int, int>(pos.first, pos.second+1));
                            }
                        }
                    }
                }

                chart->insert(hyp.node,
                              hyp.edge,
                              hyp.sig,
                              hyp.current_score,
                              hyp.back_position_left,
                              hyp.back_position_right);
            }
        }
        chart->finish(group);
    }
    return chart;
}

template<typename BVP>
BeamChart<BVP> *BeamChart<BVP>::beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<BVP> &constraints,
    const Chart<LogViterbiPotential> &future,
    double lower_bound,
    const BeamGroups &groups,
    bool recombine) {

    // Check the inputs.
    potentials.check(*graph);
    constraints.check(*graph);
    groups.check(graph);

    typedef LogViterbiPotential LVP;

    BeamChart<BVP> *chart = new BeamChart<BVP>(graph, &groups,
                                               &potentials,
                                               &constraints,
                                               &future,
                                               lower_bound,
                                               recombine);

    // 1) Initialize the chart with terminal nodes.
    typename BVP::ValType one = BVP::one();
    foreach (HNode node, graph->nodes()) {
        if (node->terminal()) {
            chart->insert(node, NULL, one, LVP::one(),
                          -1, -1);
            continue;
        }
    }

    for (int group = 0; group < groups.groups_size(); ++group) {
        foreach (HNode node, groups.group_nodes(group)) {
            // 2) Enumerate over each edge (in topological order).
            foreach (HEdge edge, node->edges()) {
                const typename BVP::ValType &sig =
                        constraints.score(edge);
                double score = potentials.score(edge);

                // Assume unary/binary edges.
                HNode node_left = graph->tail_node(edge, 0);
                const typename BeamChart<BVP>::BeamPointers &beam_left =
                        chart->get_beam(node_left);

                bool unary = graph->tail_nodes(edge) == 1;

                // Optimization.
                // vector<bool> valid_right;
                // binvec and_sig_right;
                if (!unary) {
                    // valid_right.resize(beam_right.size(), true);
                    // and_sig_right.flip();
                    // binvec and_sig;
                    // and_sig.flip();

                    // foreach (const BeamHyp *p, beam_left) {
                    //     and_sig &= p->sig;
                    // }

                    // HNode node_right = edge->tail_nodes()[1];
                    // const BeamChart::BeamPointers &beam_right =
                    //         chart->get_beam(node_right);
                    // valid_right.resize(beam_right.size(), false);
                    // int j = 0;
                    // foreach (const BeamHyp *p, beam_right) {
                    //     const binvec &right_sig = p->sig;
                    //     if (BVP::valid(sig, right_sig) &&
                    //         BVP::valid(and_sig, right_sig)) {
                    //         valid_right[j] = true;
                    //         and_sig_right &= p->sig;
                    //     }
                    //     ++j;
                    // }
                }
                // End Optimization.

                int i = -1;
                foreach (const BeamHyp *p_left, beam_left) {
                    ++i;

                    // Check valid.
                    if (!BVP::valid(sig, p_left->sig)) continue;
                        // ||
                        // !BVP::valid(and_sig_right,
                        //             p_left->sig)) continue;

                    // Construct sig and score.
                    const typename BVP::ValType mid_sig = BVP::times(sig, p_left->sig);
                    double mid_score =
                            LVP::times(score, p_left->current_score);


                    if (unary) {
                        chart->insert(node, edge, mid_sig, mid_score,
                                      i, -1);
                        continue;
                    }

                    // Do right node.
                    HNode node_right = graph->tail_node(edge, 1);
                    const typename BeamChart<BVP>::BeamPointers &beam_right =
                            chart->get_beam(node_right);

                    int j = -1;
                    foreach (const BeamHyp *p_right, beam_right) {
                        ++j;

                        // Check if this signature is valid.
                        // if (!valid_right[j]) continue;
                        if (!BVP::valid(mid_sig,
                                        p_right->sig)) continue;

                        // Construct scores and sig.
                        const typename BVP::ValType full_sig =
                                BVP::times(mid_sig, p_right->sig);
                        double full_score =
                                LVP::times(mid_score, p_right->current_score);

                        // Insert into the chart.
                        chart->insert(node, edge, full_sig, full_score,
                                      i, j);
                    }
                }
            }
        }
        chart->finish(group);
    }
    return chart;
}

template<typename BVP>
Hyperpath *BeamChart<BVP>::get_path(int result) {
    // Collect backpointers.
    vector<HEdge> path;
    queue<pair<HNode, int> > to_examine;
    to_examine.push(pair<HNode, int>(hypergraph_->root(), result));
    if (result >= get_beam(hypergraph_->root()).size()) {
        return NULL;
    }

    while (!to_examine.empty()) {
        pair<HNode, int> p = to_examine.front();
        HNode node = p.first;
        int position = p.second;

        BeamHyp *score = get_beam(node)[position];
        HEdge edge = score->edge;

        to_examine.pop();
        if (edge == -1) {
            assert(node->terminal());
            continue;
        }
        path.push_back(edge);
        for (int i = 0; i < hypergraph_->tail_nodes(edge); ++i) {
            HNode node = hypergraph_->tail_node(edge, i);
            to_examine.push(pair<HNode, int>(node,
                                             (i == 0 ? score->back_position_left :
                                              score->back_position_right
                                              )));
        }
    }
    sort(path.begin(), path.end(), IdComparator());
    return new Hyperpath(hypergraph_, path);
}

// template<typename BVP>
// void BeamChart<BVP>::insert(HNode node,
//                             HEdge edge,
//                             const typename BVP::ValType &sig,
//                             double score,
//                             int bp_left, int bp_right) {

//     // Check that the node is not bounded out.
//     double future_score = (*future_)[node];
//     double total_score = score + future_score;
//     if (total_score < lower_bound_) return;

//     // Get the group and size limit of the current node.
//     int group = groups_->group(node);
//     int beam_size = groups_->group_limit(group);
//     assert(current_group_ == group);

//     // Get the current beam.
//     Beam &b = beam_[group];
//     int &cur_size = beam_size_[group];

//     // If beam is full, check that we're at least better than the last one.
//     int limit = min(cur_size, beam_size);
//     if (limit >= beam_size &&
//         total_score < b.end()->total_score) return;

//     // Check overlap.
//     typename Beam::iterator iter = b.begin();

//     int id = node->id();
//     for (int i = 0; i < limit; ++i) {
//         if (iter->node->id() == id && iter->sig == sig) {
//             if (total_score > iter->total_score) {
//                 b.erase(iter);
//                 cur_size--;
//             }
//             return;
//         }
//     }

//     bool added = false;
//     iter = b.begin();
//     for (int i = 0; i < limit; ++i) {
//         if (total_score > iter->total_score) {
//             b.insert(iter,
//                      BeamHyp(edge, node, sig, score, future_score, bp_left, bp_right));
//             cur_size++;
//             added = true;
//             break;
//         }
//         iter++;
//     }
//     if (added) {
//         while (cur_size > beam_size) {
//             b.pop_back();
//             cur_size--;
//         }
//         return;
//     }

//     if (cur_size < beam_size)  {
//         b.push_back(
//             BeamHyp(edge, node, sig, score, future_score, bp_left, bp_right));
//         cur_size++;
//     }
// }

template<typename BVP>
bool comp(typename BeamChart<BVP>::BeamHyp *hyp1,
          typename BeamChart<BVP>::BeamHyp *hyp2) {
    return (hyp1->total_score >= hyp2->total_score);
}

template<typename BVP>
void BeamChart<BVP>::insert(HNode node,
                            HEdge edge,
                            const typename BVP::ValType &sig,
                            double score,
                            int bp_left, int bp_right) {

    // Check that the node is not bounded out.
    double future_score = (*future_)[node];
    double total_score = score + future_score;
    if (total_score < lower_bound_) return;

    // Get the group and size limit of the current node.
    int group = groups_->group(node);
    int beam_size = groups_->group_limit(group);
    assert(current_group_ == group);

    // Get the current beam.
    Beam *b = beam_[group];

    // If beam is full, check that we're at least better than the last one.


    int limit = min(static_cast<int>(b->size()), beam_size);
    typename Beam::reverse_iterator riter = b->rbegin();
    if (limit >= beam_size &&
        total_score < riter->total_score) {
        // Pruned;
        exact = false;
        return;
    }

    // Check overlap.
    // typename Beam::iterator iter = b.begin();
    // for (int i = 0; i < limit; ++i) {
    //     if ((*iter)->node->id() == node->id() && (*iter)->sig == sig) {
    //         if (total_score > (*iter)->total_score) {
    //             //(*iter) = new BeamHyp(edge, node, sig, score, future_score, bp_left, bp_right);
    //             //return;
    //             b.erase(iter);
    //             break;
    //         } else {
    //             return;
    //         }
    //     }
    //     iter++;
    // }

    BeamHyp *element;
    if (hyp_pool_.empty()) {
        element = new BeamHyp(edge, node, sig, score,
                                       future_score, bp_left, bp_right);
    } else {
        element = hyp_pool_.top();
        element->reset(edge, node, sig, score,
                       future_score, bp_left, bp_right);
        hyp_pool_.pop();
    }

    b->insert_equal(*element);
    typename Beam::reverse_iterator iter = b->rbegin();
    iter++;
    while (b->size() > beam_size) {
        typename Beam::reverse_iterator erase_iter = b->rbegin();
        hyp_pool_.push(&(*erase_iter));
        b->erase(iter.base());
        // Pruned;
        exact = false;
    }

    // iter = lower_bound(b.begin(), b.end(), element, comp<BVP>);
    // assert(total_score >= (*iter)->total_score);
    // if (iter != b.end()) {
    //     b.insert(iter, element);
    //     if (b.size() >= 2 * beam_size) {
    //         b.resize(beam_size);
    //     }
    //     return;
    // }

    // iter = b.begin();
    // for (int i = 0; i < limit; ++i) {
    //     if (total_score >= (*iter)->total_score) {
    //         b.insert(iter, element);
    //         if (b.size() >= 2 * beam_size) {
    //             b.resize(beam_size);
    //         }
    //         return;
    //     }
    //     iter++;
    // }

    // if (b.size() < beam_size) {
    //     b.push_back(element);
    // }
}

// template<typename BVP>
// void BeamChart<BVP>::insert(HNode node,
//                             HEdge edge,
//                             const typename BVP::ValType &sig,
//                             double score,
//                             int bp_left, int bp_right) {

//     // Check that the node is not bounded out.
//     double future_score = (*future_)[node];
//     double total_score = score + future_score;
//     if (total_score < lower_bound_) return;

//     // Get the group and size limit of the current node.
//     int group = groups_->group(node);
//     int beam_size = groups_->group_limit(group);
//     assert(current_group_ == group);

//     // Get the current beam.
//     Beam &b = *beam_[group];

//     // If beam is full, check that we're at least better than the last one.
//     int limit = min(static_cast<int>(b.size()), beam_size);
//     if (limit >= beam_size &&
//         total_score < b[beam_size - 1]->total_score) return;

//     // Check overlap.
//     typename Beam::iterator iter = b.begin();
//     for (int i = 0; i < limit; ++i) {
//         if ((*iter)->node->id() == node->id() && (*iter)->sig == sig) {
//             if (total_score > (*iter)->total_score) {
//                 //(*iter) = new BeamHyp(edge, node, sig, score, future_score, bp_left, bp_right);
//                 //return;
//                 b.erase(iter);
//                 break;
//             } else {
//                 return;
//             }
//         }
//         iter++;
//     }

//     BeamHyp element = new BeamHyp(edge, node, sig, score, future_score, bp_left, bp_right);

//     iter = lower_bound(b.begin(), b.end(), element, comp<BVP>);
//     assert(total_score >= (*iter)->total_score);
//     if (iter != b.end()) {
//         b.insert(iter, element);
//         if (b.size() >= 2 * beam_size) {
//             b.resize(beam_size);
//         }
//         return;
//     }

//     // iter = b.begin();
//     // for (int i = 0; i < limit; ++i) {
//     //     if (total_score >= (*iter)->total_score) {
//     //         b.insert(iter, element);
//     //         if (b.size() >= 2 * beam_size) {
//     //             b.resize(beam_size);
//     //         }
//     //         return;
//     //     }
//     //     iter++;
//     // }

//     if (b.size() < beam_size) {
//         b.push_back(element);
//     }
// }

template<typename BVP>
void BeamChart<BVP>::finish(int group) {
    // Finished all nodes in the group.
    assert(group == current_group_);
    Beam *b = beam_[group];
    int beam_size = groups_->group_limit(group);

    int size = min(static_cast<int>(b->size()), beam_size);
    vector<bool> seen(size, false);

    if (recombine_) {
        typename Beam::iterator iter = b->begin();
        for (int i = 0; i < size; ++i) {
            typename Beam::iterator iter2 = b->begin();
            for (int j = 0; j < size; ++j) {

                if (j > i &&
                    iter->node == iter2->node &&
                    iter->sig == iter2->sig) {
                    seen[j] = true;
                }
                iter2++;
            }
            iter++;
        }

        iter = b->begin();
        for (int i = 0; i < size; ++i) {
            if (!seen[i]) {
                BeamPointers &bp = beam_nodes_[iter->node->id()];
                bp.push_back(&(*iter));
            }
            iter++;
        }
    } else {
        typename Beam::iterator iter = b->begin();
        for (int i = 0; i < size; ++i) {
            BeamPointers &bp = beam_nodes_[iter->node->id()];
            bp.push_back(&(*iter));
            iter++;
        }
    }
    current_group_++;
}

template class BeamChart<BinaryVectorPotential>;
template class BeamChart<AlphabetPotential>;
template class BeamChart<LogViterbiPotential>;
