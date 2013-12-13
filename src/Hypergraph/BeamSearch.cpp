// Copyright [2013] Alexander Rush

#include "./common.h"
#include "Hypergraph/BeamSearch.h"


BeamChart *beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<BinaryVectorPotential> &constraints,
    const Chart<LogViterbiPotential> &future,
    double lower_bound,
    const BeamGroups &groups) {

    // Check the inputs.
    potentials.check(graph);
    constraints.check(graph);
    groups.check(graph);

    typedef BinaryVectorPotential BVP;
    typedef LogViterbiPotential LVP;

    BeamChart *chart = new BeamChart(graph, &groups,
                                     &future, lower_bound);

    // 1) Initialize the chart with terminal nodes.
    foreach (HNode node, graph->nodes()) {
        if (node->terminal()) {
            vector<int> back_position;
            chart->insert(node, NULL, BVP::one(), LVP::one(),
                          back_position);
            continue;
        }
    }

    for (int group = 0; group < groups.groups_size(); ++group) {
        foreach (HNode node, groups.group_nodes(group)) {

            // 2) Enumerate over each edge (in topological order).
            foreach (HEdge edge, node->edges()) {

                const binvec &sig = constraints.score(edge);
                double score = potentials.score(edge);

                // Assume unary/binary edges.
                HNode node_left = edge->tail_nodes()[0];
                const BeamChart::BeamPointers &beam_left =
                        chart->get_beam(node_left);

                bool unary = edge->tail_nodes().size() == 1;

                // Optimization.
                vector<bool> valid_right;
                binvec and_sig_right;
                if (!unary) {
                    and_sig_right.flip();
                    binvec and_sig;
                    and_sig.flip();

                    foreach (const BeamHyp *p, beam_left) {
                        and_sig &= p->sig;
                    }

                    HNode node_right = edge->tail_nodes()[1];
                    const BeamChart::BeamPointers &beam_right =
                            chart->get_beam(node_right);
                    valid_right.resize(beam_right.size(), false);
                    int j = 0;
                    foreach (const BeamHyp *p, beam_right) {
                        const binvec &right_sig = p->sig;
                        if (VALID_BINARY_VECTORS(sig, right_sig) &&
                            VALID_BINARY_VECTORS(and_sig, right_sig)) {
                            valid_right[j] = true;
                            and_sig_right &= p->sig;
                        }
                        ++j;
                    }
                }
                // End Optimization.

                vector<int> back_position(edge->tail_nodes().size());
                int i = -1;
                foreach (const BeamHyp *p_left, beam_left) {
                    ++i;

                    // Check valid.
                    if (!VALID_BINARY_VECTORS(sig, p_left->sig) ||
                        !VALID_BINARY_VECTORS(and_sig_right, p_left->sig)) continue;

                    // Construct sig and score.
                    const binvec mid_sig = BVP::times(sig, p_left->sig);
                    double mid_score =
                            LVP::times(score, p_left->current_score);
                    back_position[0] = i;


                    if (unary) {
                        chart->insert(node, edge, mid_sig, mid_score,
                                      back_position);
                        continue;
                    }

                    // Do right node.
                    HNode node_right = edge->tail_nodes()[1];
                    const BeamChart::BeamPointers &beam_right =
                            chart->get_beam(node_right);

                    int j = -1;
                    foreach (const BeamHyp *p_right, beam_right) {
                        ++j;

                        // Check if this signature is valid.
                        if (!valid_right[j]) continue;
                        if (!VALID_BINARY_VECTORS(mid_sig, p_right->sig)) continue;

                        // Construct scores and sig.
                        back_position[1] = j;
                        const binvec full_sig =
                                BVP::times(mid_sig, p_right->sig);
                        double full_score =
                                LVP::times(mid_score, p_right->current_score);

                        // Insert into the chart.
                        chart->insert(node, edge, full_sig, full_score,
                                      back_position);
                    }
                }
            }
        }
        chart->finish(group);
    }
    return chart;
}

Hyperpath *BeamChart::get_path(int result) {
    // Collect backpointers.
    vector<HEdge> path;
    queue<pair<HNode, int> > to_examine;
    to_examine.push(pair<HNode, int>(hypergraph_->root(), result));
    while (!to_examine.empty()) {
        pair<HNode, int> p = to_examine.front();
        HNode node = p.first;
        int position = p.second;

        BeamHyp *score = get_beam(node)[position];
        HEdge edge = score->edge;

        to_examine.pop();
        if (edge == NULL) {
            assert(node->terminal());
            continue;
        }
        path.push_back(edge);
        for (int i = 0; i < edge->tail_nodes().size(); ++i) {
            HNode node = edge->tail_nodes()[i];
            to_examine.push(pair<HNode, int>(node,
                                             score->back_position[i]));

        }
    }
    sort(path.begin(), path.end(), IdComparator());
    return new Hyperpath(hypergraph_, path);
}

void BeamChart::insert(HNode node,
                       HEdge edge,
                       binvec bitmap,
                       double val,
                       const vector<int> &bp) {

    // Check that the node is not bounded out.


    double future_val = (*future_)[node];
    if (val + future_val < lower_bound_) return;
    int group = groups_->group(node);
    assert(current_group_ == group);
    Beam &b = beam_[group];
    int beam_size = groups_->group_limit(group);
    int limit = min((int)b.size(), beam_size);
    if (limit >= beam_size &&
        val + future_val < b[beam_size].total_score()) return;

    // Check overlap.

    //
    //if (bitmap.any()) {
        for (int i = 0; i < limit; ++i) {
            if (b[i].node->id() == node->id() && b[i].sig == bitmap) {
                if (val + future_val > b[i].total_score()) {
                    b[i] = BeamHyp(edge, node, bitmap, val, future_val, bp);
                    return;
                } else {
                    return;
                }
            }
        }
        //}


    for (int i = 0; i < limit; ++i) {
        if (val + future_val >= b[i].total_score()) {
            b.insert(b.begin() + i,
                     BeamHyp(edge, node, bitmap, val, future_val, bp));
            if (b.size() >= beam_size) {
                b.resize(beam_size);
            }
            return;
        }
    }


    if (b.size() < beam_size)  {
        b.push_back(
            BeamHyp(edge, node, bitmap, val, future_val, bp));
    }


    //
    // int i = -1;
    // if () {

    // }

    // Binary search.

    // int upper = limit;
    // int lower = 0;
    // int cur = (upper + lower) / 2;
    // while(true) {
    //     assert(cur >= lower);
    //     assert(cur <= upper);
    //     pair<binvec, BeamScore> &p = b[cur];
    //     if (cur == lower || cur == upper) {
    //         b.insert(b.begin() + cur,
    //                  pair<binvec, BeamScore>(bitmap,
    //                                          BeamScore(edge, val, future_val, bp)));
    //         return;
    //     }

    //     if (val >= p.second.current_score) {
    //         upper = cur;
    //         cur = (cur + lower) / 2;
    //     } else if (val < p.second.current_score) {
    //         lower = cur;
    //         cur = (cur + upper) / 2;
    //     }
    // }



    // BeamMap::iterator iter = chart_[node->id()].find(bitmap);
    // if (iter != chart_[node->id()].end()) {
    //     if (val + future_val > iter->second.total_score()) {
    //         iter->second = BeamScore(edge, val, future_val, bp);
    //     }
    // } else {
    //     chart_[node->id()][bitmap] = BeamScore(edge, val, future_val, bp);
    // }
}

void BeamChart::finish(int group) {
    // Finished all nodes in the group.
    assert(group == current_group_);
    Beam &b = beam_[group];
    for (int i = 0; i < b.size(); ++i) {
        BeamPointers &bp = beam_nodes_[b[i].node->id()];
        bp.push_back(&b[i]);
    }
    current_group_++;
}

// Non-binary beam search.
// Hyperpath *beam_search(
//     const Hypergraph *graph,
//     const HypergraphPotentials<LogViterbiPotential> &potentials,
// 	//const HypergraphPotentials<SparseVectorPotential> &constraints,
//     const HypergraphPotentials<BinaryVectorPotential> &constraints,
//     const Chart<LogViterbiPotential> &future) {
//   potentials.check(graph);
//   constraints.check(graph);
//   BeamChart *chart = new BeamChart(graph);

//   //vector<map<binvec, HEdge, LessThan> > back(graph->nodes().size());

//   foreach (HNode node, graph->nodes()) {
//       if (node->terminal()) {
//           chart->insert(node,
//                         BinaryVectorPotential::one(),
//                         LogViterbiPotential::one(),
//                         future[node]);
//       }
//   }
//   foreach (HEdge edge, graph->edges()) {
//       vector<const BeamChart::Beam *> beam_maps(edge->tail_nodes().size());
//       int i = 0;
//       foreach (HNode node, edge->tail_nodes()) {
//           beam_maps[i] = &chart->get_beam(node);
//           i++;
//       }
//       vector<int> position(edge->tail_nodes().size(), 0);
//       while(true) {
//           position[0] += 1;
//           bool done = false;
//           for (int i = 0; i < edge->tail_nodes().size(); ++i) {
//               if (position[i] >= beam_maps[i]->size()) {
//                   if (i == edge->tail_nodes().size()) {
//                       done = true;
//                       break;
//                   } else {
//                       position[i] = 0;
//                       position[i + 1]++;
//                   }
//               }
//           }
//           if (done) break;

//           bool failed = false;
//           binvec cur = constraints.score(edge);
//           double score = potentials.score(edge);

//           for (int i = 0; i < edge->tail_nodes().size(); ++i) {
//               const binvec &one = (*beam_maps[i])[position[i]].first;
//               double one_score =
//                       (*beam_maps[i])[position[i]].second.current_score;
//               if (!BinaryVectorPotential::valid(cur, one)) {
//                   failed = true;
//                   break;
//               }
//               cur = BinaryVectorPotential::times(cur, one);
//               score = LogViterbiPotential::times(score, one_score);
//           }
//           if (!failed) {
//               chart->insert(edge->head_node(),
//                             cur,
//                             score,
//                             future[edge->head_node()]);
//           }
//       }
//   }
// }
