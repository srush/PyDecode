#ifndef COMMON_H
#define COMMON_H

#include <boost/foreach.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include<boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#define foreach BOOST_FOREACH
//typedef boost::numeric::ublas::vector<double> Vec;
//typedef boost::numeric::ublas::mapped_vector<double> SparseVec;
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
