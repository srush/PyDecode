
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>
#include "Hypergraph/Hypergraph.h"
#include "./common.h"



// A base class of a weight with traits of a semiring
// including + and * operators, and annihlator/identity elements.
template<typename ValType, typename SemiringWeight>
class BaseSemiringWeight {
public:
	BaseSemiringWeight(const SemiringWeight& other)
		: value(normalize(other.value)) {}
	BaseSemiringWeight(ValType val) : value(normalize(val)) {}

	operator ValType() const { return value; }

	SemiringWeight& operator=(SemiringWeight rhs) {
		normalize(rhs.value);
		std::swap(value, rhs.value);
		return *this;
	}

	SemiringWeight& operator=(ValType rhs) {
		normalize(rhs);
		std::swap(value, rhs);
		return *this;
	}

	friend bool operator==(const SemiringWeight& lhs,const SemiringWeight& rhs) {
		return lhs.value == rhs.value;
	}

	friend SemiringWeight operator+(SemiringWeight lhs, const SemiringWeight &rhs) {
		lhs += rhs;
		return lhs;
	}
	friend SemiringWeight operator*(SemiringWeight lhs, const SemiringWeight &rhs) {
		lhs *= rhs;
		return lhs;
	}

	SemiringWeight& operator+=(const SemiringWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	SemiringWeight& operator*=(const SemiringWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const SemiringWeight one() { return SemiringWeight(1.0); }
	static const SemiringWeight zero() { return SemiringWeight(0.0); }

	// Determines range of acceptable values
	ValType normalize(ValType val) { return val; };

protected:
	ValType value;
};

// Implements the Viterbi type of semiring as described in Huang 2006
// +: max
// *: *
// 0: 0
// 1: 1
class ViterbiWeight : public BaseSemiringWeight<double, ViterbiWeight> {
public:
ViterbiWeight(double value) : BaseSemiringWeight<double, ViterbiWeight>(normalize(value)) { }
ViterbiWeight() : BaseSemiringWeight<double, ViterbiWeight>(ViterbiWeight::zero()) {}
	ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	double normalize(double val) const {
		if (val < 0.0) val = 0.0;
		else if (val > 1.0) val = 1.0;
		return val;
	}

	static const ViterbiWeight one() { return ViterbiWeight(1.0); }
	static const ViterbiWeight zero() { return ViterbiWeight(0.0); }
};

// Implements the log-space Viterbi type of semiring.
// +: max
// *: +
// 0: -INF
// 1: 0
class LogViterbiWeight : public BaseSemiringWeight<double, LogViterbiWeight> {
public:
     LogViterbiWeight(double value) :
       BaseSemiringWeight<double, LogViterbiWeight>(normalize(value)) { }
     LogViterbiWeight() :
       BaseSemiringWeight<double, LogViterbiWeight>(LogViterbiWeight::zero()) {}

     LogViterbiWeight& operator+=(const LogViterbiWeight& rhs) {
       value = std::max(value, rhs.value);
       return *this;
     }
     LogViterbiWeight& operator*=(const LogViterbiWeight& rhs) {
       value = value + rhs.value;
       return *this;
     }

     double normalize(double val) const {
       if (val < -INF) return -INF;
       return val;
     }

     static const LogViterbiWeight one() { return LogViterbiWeight(0.0); }
     static const LogViterbiWeight zero() { return LogViterbiWeight(-INF); }
};


// Implements the Boolean type of semiring as described in Huang 2006
// +: logical or
// *: logical and
// 0: false
// 1: true
class BoolWeight : public BaseSemiringWeight<bool, BoolWeight> {
public:
BoolWeight(bool value) : BaseSemiringWeight<bool, BoolWeight>(normalize(value)) { }

	BoolWeight& operator+=(const BoolWeight& rhs) {
		value = value || rhs.value;
		return *this;
	}
	BoolWeight& operator*=(const BoolWeight& rhs) {
		value = value && rhs.value;
		return *this;
	}

	static const BoolWeight one() { return BoolWeight(true); }
	static const BoolWeight zero() { return BoolWeight(false); }

	bool normalize(bool val) const { return val; }
};

// Implements the Inside type of semiring as described in Huang 2006
// +: +
// *: *
// 0: 0
// 1: 1
class InsideWeight : public BaseSemiringWeight<double, InsideWeight> {
public:
InsideWeight(double value) : BaseSemiringWeight<double, InsideWeight>(normalize(value)) { }

	InsideWeight& operator+=(const InsideWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	InsideWeight& operator*=(const InsideWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const InsideWeight one() { return InsideWeight(1.0); }
	static const InsideWeight zero() { return InsideWeight(0.0); }

	double normalize(double val) const {
		if (val < 0.0) val = 0.0;
		return val;
	}
};

// Implements the Real type of semiring as described in Huang 2006
// +: min
// *: +
// 0: INF
// 1: 0
class RealWeight : public BaseSemiringWeight<double, RealWeight> {
public:
RealWeight(double value) : BaseSemiringWeight<double, RealWeight>(normalize(value)) { }

	RealWeight& operator+=(const RealWeight& rhs) {
		value = std::min(value, rhs.value);
		return *this;
	}
	RealWeight& operator*=(const RealWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	static const RealWeight one() { return RealWeight(0.0); }
	static const RealWeight zero() { return RealWeight(INF); }

	double normalize(double val) const { return val; }
};

// Implements the Inside type of semiring as described in Huang 2006
// +: min
// *: +
// 0: INF
// 1: 0
class TropicalWeight : public BaseSemiringWeight<double, TropicalWeight> {
public:
TropicalWeight(double value) : BaseSemiringWeight<double, TropicalWeight>(normalize(value)) { }

	TropicalWeight& operator+=(const TropicalWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	TropicalWeight& operator*=(const TropicalWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const TropicalWeight one() { return TropicalWeight(0.0); }
	static const TropicalWeight zero() { return TropicalWeight(INF); }

	double normalize(double val) const {
		if (val < 0.0) val = 0.0;
		return val;
	}
};




// Implements the Counting type of semiring as described in Huang 2006
// +: +
// *: *
// 0: 0
// 1: 1
class CountingWeight : public BaseSemiringWeight<int, CountingWeight> {
public:
CountingWeight(int value) : BaseSemiringWeight<int, CountingWeight>(normalize(value)) { }

	CountingWeight& operator+=(const CountingWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	CountingWeight& operator*=(const CountingWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const CountingWeight one() { return CountingWeight(1); }
	static const CountingWeight zero() { return CountingWeight(0); }

	int normalize(int val) const {
		if(val < 0) val = 0;
		return val;
	}
};


class TreeWeight : public BaseSemiringWeight<Hypernode *, TreeWeight> {
public:
TreeWeight(Hypernode *value) : BaseSemiringWeight<Hypernode *, TreeWeight>(normalize(value)) { }

	TreeWeight& operator+=(const TreeWeight& rhs) {
		return *this;
	}

	TreeWeight& operator*=(const TreeWeight& rhs) {
      if (rhs.value == NULL or value == NULL) {
        value = NULL;
      } else {
        vector<HNode> tails;
        tails.push_back(value);
        tails.push_back(rhs.value);
        Hypernode *node = new Hypernode("");
        Hyperedge *edge = new Hyperedge("", node, tails);
        node->add_edge(edge);
        value = node;
      }
      return *this;
	}

	static const TreeWeight one() {
      return TreeWeight(new Hypernode(""));
    }
	static const TreeWeight zero() { return TreeWeight(NULL); }

	Hypernode *normalize(Hypernode *val) const {
		return val;
	}
};


template<typename SemiringComp, typename SemiringOther>
class CompWeight : public BaseSemiringWeight<std::pair<SemiringComp, SemiringOther>, CompWeight<SemiringComp, SemiringOther> > {
public:
  typedef std::pair<SemiringComp, SemiringOther> MyVal;
  typedef CompWeight<SemiringComp, SemiringOther> MyClass;
  using BaseSemiringWeight<MyVal, MyClass>::value;

  CompWeight(MyVal value) : BaseSemiringWeight<MyVal, MyClass>(normalize(value)) { }

  MyClass& operator+=(const MyClass& rhs) {
    if (value.first < rhs.value.first) value = rhs.value;
    return *this;
  }

  MyClass& operator*=(const MyClass& rhs) {
    value.first = value.first * rhs.value.first;
    value.second = value.second * rhs.value.second;
    return *this;
  }

  static const MyClass one() { return MyClass(val(SemiringComp::one(), SemiringOther::one())); }
  static const MyClass zero() { return MyClass(val(SemiringComp::zero(), SemiringOther::zero())); }

  MyVal normalize(MyVal val) const {
    val.first = val.first.normalize(val.first);
    val.second = val.second.normalize(val.second);
    return val;
  }
};


template<typename SemiringType>
class HypergraphWeights {
 public:
  HypergraphWeights(const Hypergraph *hypergraph,
                    const vector<SemiringType> &weights,
                    SemiringType bias)
  : hypergraph_(hypergraph),
    weights_(weights),
    bias_(bias) {
      assert(weights.size() == hypergraph->edges().size());
  }

 SemiringType dot(const Hyperpath &path) const {
   path.check(*hypergraph_);
   SemiringType score = SemiringType::one();
   foreach (HEdge edge, path.edges()) {
     score *= weights_[edge->id()];
   }
   return score + bias_;
 }

  SemiringType score(HEdge edge) const { return weights_[edge->id()]; }

  SemiringType bias() const { return bias_; }

  HypergraphWeights<SemiringType> *modify(const vector<SemiringType> &,
                                          SemiringType) const;

  HypergraphWeights<SemiringType> *project_weights(
      const HypergraphProjection &projection) const;

  void check(const Hypergraph &graph) const {
    if (!graph.same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match weights.");
    }
  }

 protected:
  const Hypergraph *hypergraph_;
  vector<SemiringType> weights_;
  SemiringType bias_;
};


template <>
inline double HypergraphWeights<double>::dot(const Hyperpath &path) const {
  path.check(*hypergraph_);
  double score = 0.0;
  foreach (HEdge edge, path.edges()) {
    score += weights_[edge->id()];
  }
  return score + bias_;
}



template<typename SemiringType>
HypergraphWeights<SemiringType> *HypergraphWeights<SemiringType>::modify(
    const vector<SemiringType> &edge_duals,
    SemiringType bias_dual) const {
  vector<SemiringType> new_weights(weights_);
  for (uint i = 0; i < edge_duals.size(); ++i) {
    new_weights[i] += edge_duals[i];
  }
  return new HypergraphWeights<SemiringType>(hypergraph_,
                               new_weights,
                               bias_ + bias_dual);
}

template<typename SemiringType>
HypergraphWeights<SemiringType> *HypergraphWeights<SemiringType>::project_weights(
    const HypergraphProjection &projection) const {
  vector<SemiringType> weights(projection.new_graph->edges().size());
  foreach (HEdge edge, projection.original_graph->edges()) {
    HEdge new_edge = projection.project(edge);
    if (new_edge != NULL && new_edge->id() >= 0) {
      assert(new_edge->id() < projection.new_graph->edges().size());
      weights[new_edge->id()] = score(edge);
    }
  }
  return new HypergraphWeights<SemiringType>(projection.new_graph, weights, bias_);
}

#endif // HYPERGRAPH_SEMIRING_H_
