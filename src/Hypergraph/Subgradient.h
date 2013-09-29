#ifndef SUBGRADIENT_H_
#define SUBGRADIENT_H_

#include <vector>
#include <./common.h>
using namespace std;

typedef boost::numeric::ublas::mapped_vector<double> SparseVec;

class SubgradRate {
 public:
  virtual double get_alpha(vector<double> &past_duals,
                           const SparseVec &subgrad) const = 0;
};

class ConstantRate : public SubgradRate {
 public:
  double get_alpha(vector <double> &past_duals,
                   const SparseVec &subgrad) const {
    return 1.0;
  }
};

class DecreasingRate : public SubgradRate {
 public:
  double get_alpha(vector<double> &past_duals,
                   const SparseVec &subgrad) const {
    double rate = 1.0;
    for (int i = 0; i < past_duals.size(); ++i) {
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
  SparseVec *duals;
};

// Output of the subgradient client
struct SubgradResult {
  SubgradResult() : subgrad(10000) {}

  // The dual value with these weights.
  double dual;

  // The subgradient at this iteration.
  SparseVec subgrad;
};


/**
 * Interface for subgradient clients.
 */
class SubgradientProducer {
 public:
  // Solve the problem with the current dual weights.
  virtual void solve(const SubgradState & cur_state,
                     SubgradResult *result) const = 0;
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
 Subgradient(const SubgradientProducer *subgrad_producer,
             const SubgradRate *update_rate)
     : producer_(subgrad_producer),
      rate_(update_rate),
      best_dual_(-INF),
      round_(1),
      debug_(false),
    max_round_(200),
    duals_(10000){}

  void set_debug(){ debug_ = true; }
  void set_max_rounds(int max_round) {
    max_round_ = max_round; 
  }
  bool solve();

 private:
  bool run_one_round(bool *optimal);
  void update_weights(const SparseVec &);

  const SubgradientProducer *producer_;
  const SubgradRate *rate_;

  double best_dual_;
  int round_;

  SparseVec duals_;
  vector<double> past_duals_;
  bool debug_;
  int max_round_;
};

#endif
