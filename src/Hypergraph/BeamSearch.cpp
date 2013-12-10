// Copyright [2013] Alexander Rush

#include "Hypergraph/BeamSearch.h"

BeamChart *beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
    const HypergraphPotentials<BinaryVectorPotential> &constraints,
    const Chart<LogViterbiPotential> &future,
    double lower_bound,
    int beam_size) {

    // Check the inputs.
    potentials.check(graph);
    constraints.check(graph);
    typedef BinaryVectorPotential BVP;
    typedef LogViterbiPotential LVP;

    BeamChart *chart = new BeamChart(graph, beam_size,
                                     &future, lower_bound);

    // 1) Initialize the chart with terminal nodes.
    foreach (HNode node, graph->nodes()) {

        if (node->terminal()) {
            vector<int> back_position;
            chart->insert(node, NULL, BVP::one(), LVP::one(),
                          back_position);
            chart->finish(node);
            continue;
        }

        // 2) Enumerate over each edge (in topological order).
        foreach (HEdge edge, node->edges()) {

            const binvec &sig = constraints.score(edge);
            double score = potentials.score(edge);

            // Assume unary/binary edges.
            HNode node_left = edge->tail_nodes()[0];
            const BeamChart::Beam &beam_left = chart->get_beam(node_left);

            vector<int> back_position(edge->tail_nodes().size());
            for (int i = 0; i < beam_left.size(); ++i) {
                const binvec &left_sig = beam_left[i].first;
                if (!VALID_BINARY_VECTORS(sig, left_sig)) continue;

                double left_score = beam_left[i].second.current_score;
                const binvec mid_sig = BVP::times(sig, left_sig);
                double mid_score = LVP::times(score, left_score);
                back_position[0] = i;

                if (edge->tail_nodes().size() == 1) {

                    chart->insert(node, edge, mid_sig, mid_score,
                                  back_position);
                    continue;
                }

                // Do right node.
                HNode node_right = edge->tail_nodes()[1];
                const BeamChart::Beam &beam_right = chart->get_beam(node_right);

                for (int j = 0; j < beam_right.size(); ++j) {

                    const binvec &right_sig = beam_right[j].first;
                    if (!VALID_BINARY_VECTORS(mid_sig, right_sig)) continue;
                    back_position[1] = j;
                    double right_score = beam_right[j].second.current_score;
                    const binvec full_sig = BVP::times(mid_sig, right_sig);
                    double full_score = LVP::times(mid_score, right_score);

                    // Insert into the chart.
                    chart->insert(node, edge, full_sig, full_score,
                                  back_position);
                }
            }
        }
        chart->finish(node);
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
        BeamScore score = get_beam(node)[position].second;
        HEdge edge = score.edge;

        to_examine.pop();
        if (edge == NULL) {
            assert(node->terminal());
            continue;
        }
        path.push_back(edge);
        for (int i = 0; i < edge->tail_nodes().size(); ++i) {
            HNode node = edge->tail_nodes()[i];
            to_examine.push(pair<HNode, int>(node,
                                             score.back_position[i]));

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

    Beam &b = beam_[node->id()];
    for (int i = 0; i < min((int)b.size(), beam_size_); ++i) {
        pair<binvec, BeamScore> &cur = b[i];
        if (cur.first == bitmap) {
            if (val + future_val > cur.second.total_score()) {
                cur.second = BeamScore(edge, val, future_val, bp);
                return;
            } else {
                return;
            }
        }

    }
    for (int i = 0; i < min((int)b.size(), beam_size_); ++i) {
        pair<binvec, BeamScore> &cur = b[i];
        if (val + future_val > cur.second.total_score()) {
            b.insert(b.begin() + i,
                     pair<binvec, BeamScore>(bitmap,
                                             BeamScore(edge, val, future_val, bp)));
            return;
        }
    }

    if (b.size() < beam_size_) {
        b.push_back(
            pair<binvec, BeamScore>(bitmap,
                                    BeamScore(edge, val, future_val, bp)));
    }


    // BeamMap::iterator iter = chart_[node->id()].find(bitmap);
    // if (iter != chart_[node->id()].end()) {
    //     if (val + future_val > iter->second.total_score()) {
    //         iter->second = BeamScore(edge, val, future_val, bp);
    //     }
    // } else {
    //     chart_[node->id()][bitmap] = BeamScore(edge, val, future_val, bp);
    // }
}

void BeamChart::finish(HNode node) {
    return;
    // BeamMap &map = chart_[node->id()];
    // for (int k = 0; k < beam_size_; ++k) {
    //     double score = -INF;
    //     const binvec *b = NULL;
    //     BeamMap::const_iterator iter;
    //     for (iter = map.begin(); iter != map.end();
    //          ++iter) {
    //         double cur = iter->second.total_score();
    //         if (cur > score) {
    //             score = cur;
    //             b = &iter->first;
    //         }
    //     }
    //     if (b == NULL) break;
    //     beam_[node->id()].push_back(pair<binvec, BeamScore>(*b, map[*b]));
    //     map.erase(*b);
    // }
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
