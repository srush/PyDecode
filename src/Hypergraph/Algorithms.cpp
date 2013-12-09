// Copyright [2013] Alexander Rush


#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>

#include "Hypergraph/Algorithms.h"
//#include "Hypergraph/Subgradient.h"

#define SPECIALIZE_ALGORITHMS_FOR_SEMI(X)\
  template class Chart<X>;\
  template class HypergraphPotentials<X>;\
  template class Marginals<X>;\
  template Hyperpath *general_viterbi<X>(const Hypergraph *graph,const HypergraphPotentials<X> &potentials);

#define SPECIALIZE_FOR_SEMI_MIN(X)\
  template class Chart<X>;\
  template class HypergraphPotentials<X>;\
  template Chart<X> *general_inside<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials);\
  template Chart<X> *general_outside<X>(const Hypergraph *graph, const HypergraphPotentials<X> &potentials, const Chart<X> &);

using namespace std;

// General code.

struct IdComparator {
  bool operator()(HEdge edge1, HEdge edge2) const {
    return edge1->id() < edge2->id();
  }
};

template<typename S>
Chart<S> *
general_inside(const Hypergraph *graph,
               const HypergraphPotentials<S> &potentials) {
  potentials.check(*graph);

  // Run Viterbi Hypergraph algorithm.
  Chart<S> *chart = new Chart<S>(graph);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      chart->insert(node, S::one());
    }
  }
  foreach (HEdge edge, graph->edges()) {
    typename S::ValType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      score = S::times(score, (*chart)[node]);
    }
    chart->insert(edge->head_node(),
                  S::add((*chart)[edge->head_node()], score));
  }
  chart->insert(graph->root(),
                S::times((*chart)[graph->root()], potentials.bias()));
  return chart;
}

template<typename S>
Chart<S> *
general_outside(const Hypergraph *graph,
                const HypergraphPotentials<S> &potentials,
                const Chart<S> &inside_chart) {
  potentials.check(*graph);
  inside_chart.check(graph);
  Chart<S> *chart = new Chart<S>(graph);
  const vector<HEdge> &edges = graph->edges();
  chart->insert(graph->root(), S::one());//potentials.bias());

  for (int i = edges.size() - 1; i >= 0; --i) {
    HEdge edge = edges[i];
    typename S::ValType head_score = (*chart)[edge->head_node()];
    if (edge->head_node()->id() == graph->root()->id()) {
        head_score = potentials.bias();
    }
    foreach (HNode node, edge->tail_nodes()) {
      typename S::ValType other_score = S::one();
      foreach (HNode other_node, edge->tail_nodes()) {
        if (other_node->id() == node->id()) continue;
        other_score = S::times(other_score, inside_chart[other_node]);
      }
      chart->insert(node, S::add((*chart)[node],
                                 S::times(head_score,
                                          S::times(other_score,
                                                   potentials.score(edge)))));
    }
  }
  return chart;
}

template<typename S>
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<S> &potentials) {

  potentials.check(*graph);
  Chart<S> *chart = new Chart<S>(graph);
  vector<HEdge> back(graph->nodes().size(), NULL);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      chart->insert(node, S::one());
    }
  }
  foreach (HEdge edge, graph->edges()) {
    typename S::ValType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      score = S::times(score, (*chart)[node]);
    }
    if (score > (*chart)[edge->head_node()]) {
      chart->insert(edge->head_node(), score);
      back[edge->head_node()->id()] = edge;
    }
  }

  // Collect backpointers.
  vector<HEdge> path;
  queue<HNode> to_examine;
  to_examine.push(graph->root());
  while (!to_examine.empty()) {
    HNode node = to_examine.front();
    HEdge edge = back[node->id()];
    to_examine.pop();
    if (edge == NULL) {
      assert(node->terminal());
      continue;
    }
    path.push_back(edge);
    foreach (HNode node, edge->tail_nodes()) {
      to_examine.push(node);
    }
  }
  sort(path.begin(), path.end(), IdComparator());
  delete chart;
  return new Hyperpath(graph, path);
}

template<typename StatSem>
Hyperpath *general_viterbi(
    const Hypergraph *graph,
    const HypergraphPotentials<typename StatSem::ValType> &potentials) {

  potentials.check(*graph);
  Chart<typename StatSem::ValType> *chart = new Chart<typename StatSem::ValType>(graph);
  vector<HEdge> back(graph->nodes().size(), NULL);

  foreach (HNode node, graph->nodes()) {
    if (node->terminal()) {
      chart->insert(node, StatSem::one());
    }
  }
  foreach (HEdge edge, graph->edges()) {
    typename StatSem::ValType score = potentials.score(edge);
    foreach (HNode node, edge->tail_nodes()) {
      typename StatSem::ValType(score, (*chart)[node]);
    }
    if (score > (*chart)[edge->head_node()]) {
      chart->insert(edge->head_node(), score);
      back[edge->head_node()->id()] = edge;
    }
  }

  // Collect backpointers.
  vector<HEdge> path;
  queue<HNode> to_examine;
  to_examine.push(graph->root());
  while (!to_examine.empty()) {
    HNode node = to_examine.front();
    HEdge edge = back[node->id()];
    to_examine.pop();
    if (edge == NULL) {
      assert(node->terminal());
      continue;
    }
    path.push_back(edge);
    foreach (HNode node, edge->tail_nodes()) {
      to_examine.push(node);
    }
  }
  sort(path.begin(), path.end(), IdComparator());
  delete chart;
  return new Hyperpath(graph, path);
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


BeamChart *beam_search(
    const Hypergraph *graph,
    const HypergraphPotentials<LogViterbiPotential> &potentials,
	//const HypergraphPotentials<SparseVectorPotential> &constraints,
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
            chart->insert(node, NULL, BVP::one(), LVP::one());
            chart->finish(node);
            continue;
        }

        // 2) Enumerate over each edge (in topological order).
        foreach (HEdge edge, node->edges()) {
            const binvec &sig = constraints.score(edge);
            double score = potentials.score(edge);

            // Assume binary edges.
            HNode node_left = edge->tail_nodes()[0];
            HNode node_right = edge->tail_nodes()[1];
            const BeamChart::Beam &beam_left = chart->get_beam(node_left);
            const BeamChart::Beam &beam_right = chart->get_beam(node_left);

            for (int i = 0; i < beam_left.size(); ++i) {
                const binvec &left_sig = beam_left[i].first;
                if (!BVP::valid(sig, left_sig)) continue;

                double left_score = beam_left[i].second.current_score;
                const binvec mid_sig = BVP::times(sig, left_sig);
                double mid_score = LVP::times(score, left_score);

                for (int j = 0; j < beam_right.size(); ++j) {
                    const binvec &right_sig = beam_right[j].first;
                    if (!BVP::valid(mid_sig, right_sig)) continue;
                    double right_score = beam_right[j].second.current_score;
                    const binvec final_sig = BVP::times(mid_sig, right_sig);
                    double full_score = LVP::times(mid_score, right_score);

                    // Insert into the chart.
                    chart->insert(node, edge, final_sig, full_score);
                }
            }
        }
        chart->finish(node);
    }
    return chart;
}

Hyperpath *BeamChart::get_path() {
    // Collect backpointers.
    vector<HEdge> path;
    queue<HNode> to_examine;
    to_examine.push(hypergraph_->root());
    while (!to_examine.empty()) {
        HNode node = to_examine.front();
        HEdge edge = get_best_edge(node);
        to_examine.pop();
        if (edge == NULL) {
            assert(node->terminal());
            continue;
        }
        path.push_back(edge);
        foreach (HNode node, edge->tail_nodes()) {
            to_examine.push(node);
        }
    }
    sort(path.begin(), path.end(), IdComparator());
    return new Hyperpath(hypergraph_, path);
}

void BeamChart::insert(HNode node,
                       HEdge edge,
                       binvec bitmap,
                       double val) {
    // Check that the node is not bounded out.
    double future_val = (*future_)[node];
    if (val + future_val < lower_bound_) return;

    BeamMap::iterator iter = chart_[node->id()].find(bitmap);
    if (iter != chart_[node->id()].end()) {
        if (val + future_val >
            iter->second.total_score()) {
            iter->second = Score(edge, val, future_val);
        }
    } else {
        chart_[node->id()][bitmap] = Score(edge, val, future_val);
    }
}

void BeamChart::finish(HNode node) {
    BeamMap &map = chart_[node->id()];
    for (int k = 0; k < beam_size_; ++k) {
        double score = -INF;
        const binvec *b = NULL;
        BeamMap::const_iterator iter;
        for (iter = map.begin(); iter != map.end();
             ++iter) {
            double cur = iter->second.total_score();
            if (cur > score) {
                score = cur;
                b = &iter->first;
            }
        }
        if (b == NULL) break;
        beam_[node->id()].push_back(pair<binvec, Score>(*b, map[*b]));
        map.erase(*b);
    }
}

SPECIALIZE_ALGORITHMS_FOR_SEMI(ViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(LogViterbiPotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(InsidePotential)
SPECIALIZE_ALGORITHMS_FOR_SEMI(BoolPotential)
SPECIALIZE_FOR_SEMI_MIN(SparseVectorPotential)
SPECIALIZE_FOR_SEMI_MIN(MinSparseVectorPotential)
SPECIALIZE_FOR_SEMI_MIN(MaxSparseVectorPotential)
SPECIALIZE_FOR_SEMI_MIN(BinaryVectorPotential)

// End General code.
