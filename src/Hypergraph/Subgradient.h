#ifndef HYPERGRAPH_SUBGRADIENT_H_
#define HYPERGRAPH_SUBGRADIENT_H_

#include <vector>
#include "./common.h"

using namespace std;

class SubgradRate {
 public:
  virtual double get_alpha(const vector<double> &past_duals,
                           const vector<double> &subgrad) const = 0;
};

class ConstantRate : public SubgradRate {
 public:
  double get_alpha(const vector<double> &past_duals,
                   const vector<double> &subgrad) const {
    return 1.0;
  }
};

class DecreasingRate : public SubgradRate {
 public:
  double get_alpha(const vector<double> &past_duals,
                   const vector<double> &subgrad) const {
     double rate = 1.0;
     for (uint i = 0; i < past_duals.size(); ++i) {
       rate *= 0.9;
     }
     return rate;
  }
};

// Input to the subgradient client
struct SubgradState {
  // The current round of the subgradient algorithm
  int round;

  // The current dual values.
  vector<double> *duals;
};

// Output of the subgradient client
struct SubgradResult {
  SubgradResult(int size) : subgrad(size) {
    for (int i = 0; i < size; ++i) subgrad[i] = 0.0;
  }

  // The dual value with these weights.
  double dual;

  // The subgradient at this iteration.
  vector<double> subgrad;
};


/**
 * Interface for subgradient clients.
 */
class SubgradientProducer {
 public:
  // Solve the problem with the current dual weights.
  virtual void solve(const SubgradState & cur_state,
                     SubgradResult *result) = 0;
};

 /**
  * Subgradient optimization manager. Takes an object to produce
  * subgradients given current dual values as well as an object
  * to determine the current update rate.
  */
class Subgradient {
 public:
  /**
   *
   * @param subgrad_producer Gives the subgradient at the current position
   * @param update_rate A class to decide the alpha to use at the current iteration
   */
  Subgradient(SubgradientProducer *subgrad_producer,
              const SubgradRate *update_rate,
              int num_constraints)
    : producer_(subgrad_producer),
      rate_(update_rate),
      best_dual_(INF),
      round_(1),
      duals_(num_constraints),
      num_constraints_(num_constraints),
      max_round_(200),
      debug_(false) {}

  void set_debug() { debug_ = true; }
  void set_max_rounds(int max_round) {
    max_round_ = max_round;
  }
  bool solve();

  const vector<double> &duals() const {
    return past_duals_;
  }

 private:
  bool run_one_round(bool *optimal);
  void update_weights(const vector<double> &);

  SubgradientProducer *producer_;
  const SubgradRate *rate_;

  double best_dual_;
  int round_;

  vector<double> duals_;
  vector<double> past_duals_;
  int num_constraints_;
  int max_round_;
  bool debug_;
};

#endif  // HYPERGRAPH_SUBGRADIENT_H_
