#include "Hypergraph/Subgradient.h"

#include <math.h>
#include <time.h>
#include "./common.h"

#define TIMING 0
using namespace std;

bool Subgradient::solve() {
  bool optimal = false;
  while (run_one_round(&optimal) &&
         round_ < max_round_) {
    round_++;
  }
  return optimal;
}

bool Subgradient::run_one_round(bool *optimal) {
  *optimal = false;
  clock_t start = clock();

  SubgradResult result;
  SubgradState info;
  info.round = round_;
  info.duals = &duals_;
  producer_->solve(info, &result);

  clock_t end;
  if (TIMING) {
    end = clock();
    cout << "JUST UPDATE "<< Clock::diffclock(end, start) << endl;
  }

  // Update statistics.
  if (result.dual < best_dual_) {
    best_dual_ = result.dual;
  }
  past_duals_.push_back(result.dual);

  if (debug_) {
    cerr << "Round " << round_;
    cerr << " BEST_DUAL " << best_dual_;
    cerr << " CUR_DUAL " << result.dual;
    cerr << endl;
  }

  if (boost::numeric::ublas::norm_2(result.subgrad) == 0.0) {
    *optimal = true;
    return false;
  }
  update_weights(result.subgrad);
  return true;
}

void Subgradient::update_weights(const SparseVec &subgrad) {
  double alpha = rate_->get_alpha(past_duals_, subgrad);
  cerr << "alpha: " << alpha << " "
       << boost::numeric::ublas::norm_2(duals_) << endl;
  SparseVec updates = alpha * subgrad;
  duals_ -= updates;
}
