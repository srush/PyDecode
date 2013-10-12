#ifndef COMMON_H
#define COMMON_H

#include <boost/numeric/ublas/io.hpp>

#define INF 1e8

typedef unsigned int uint;
class Clock {
 public:
  static double diffclock(clock_t clock1,clock_t clock2){
    double diffticks=clock1-clock2;
    double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
    return diffms;
  }
};

#endif
