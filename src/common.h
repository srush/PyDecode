#ifndef COMMON_H
#define COMMON_H

#include <ctime>
#include <cstdlib>
#include "./foreach.h"

#define INF 1e9
#define foreach BOOST_FOREACH

typedef unsigned int uint;
class Clock {
 public:
  static double diffclock(clock_t clock1,clock_t clock2){
    double diffticks=clock1-clock2;
    double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
    return diffms;
  }
};

inline double dRand(double dMin, double dMax) {
    double d = (double)rand() / RAND_MAX;
    return dMin + d * (dMax - dMin);
}

#endif
