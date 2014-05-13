// Copyright [2014] Alexander Rush

#ifndef HYPERGRAPH_AUTOMATON_H_
#define HYPERGRAPH_AUTOMATON_H_

#include <vector>
#include <map>

using namespace std;

class DFA {
  public:
    DFA(int num_states, int num_symbols,
        const vector<map<int, int> > &transition,
        const set<int> &final)
            : states_(num_states),
            transition_(transition),
            final_(final) {
                assert(transition.size() == num_states);
                assert(transition[0].size() == num_symbols);
                for (int i = 0; i < num_states; ++i) {
                    states_[i] = i;
                }
            }

    const vector<int> &states() const {
        return states_;
    }

    int transition(int state, int symbol) const {
        return transition_[state].at(symbol);
    }

    int valid_transition(int state, int symbol) const {
        return transition_[state].find(symbol) != transition_[state].end();
    }

    int final(int state) const {
        return final_.find(state) != final_.end();
    }


  private:

    vector<int> states_;
    const vector<map<int, int> > transition_;
    const set<int> final_;
};


#endif  // HYPERGRAPH_AUTOMATON_H_
