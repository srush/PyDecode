
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>

#include <boost/serialization/strong_typedef.hpp>

#include "../common.h"



// A virtual base class of a weight with traits of a semiring
// including + and * operators, and annihlator/identity elements.
template<typename DerivedWeight, typename ValType>
class SemiringWeight {
public:
	operator ValType() const { return value; }
	DerivedWeight& operator=(ValType rhs) {
		normalize(rhs);
		std::swap(value, rhs);
		return *this;
	}

	// Determines range of acceptable values
	virtual ValType& normalize(ValType& val) = 0;

	friend DerivedWeight operator+(DerivedWeight lhs, const DerivedWeight &rhs) {
		lhs += rhs;
		return lhs;
	}
	friend DerivedWeight operator*(DerivedWeight lhs, const DerivedWeight &rhs) {
		lhs *= rhs;
		return lhs;
	}

	const ValType& one() const { return identity; }
	const ValType& zero() const { return annihlator; }
	bool is_one() const { return value == identity; }
	bool is_zero() const { return value == annihlator; }

	virtual DerivedWeight& operator+=(const DerivedWeight& rhs) = 0;
	virtual DerivedWeight& operator*=(const DerivedWeight& rhs) = 0;

protected:
	SemiringWeight() {}
	SemiringWeight(ValType val, ValType ann, ValType id) : value(normalize(val)), annihlator(ann), identity(id) {}
	ValType value;
	const ValType annihlator;
	const ValType identity;
};

// Implements the Boolean type of semiring as described in Huang 2006
// +: logical or
// *: logical and
// 0: 0
// 1: 1
class BoolWeight : SemiringWeight<BoolWeight, bool> {
public:
	BoolWeight(bool value) : SemiringWeight<BoolWeight, bool>(value, false, true) { }

	virtual bool& normalize(bool& val) { return val; }

	virtual BoolWeight& operator+=(const BoolWeight& rhs) {
		value = value || rhs.value;
		return *this;
	}
	virtual BoolWeight& operator*=(const BoolWeight& rhs) {
		value = value && rhs.value;
		return *this;
	}
};

// Implements the Viterbi type of semiring as described in Huang 2006
// +: max
// *: *
// 0: 0
// 1: 1
class ViterbiWeight : SemiringWeight<ViterbiWeight, double> {
public:
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight, double>(value, 0.0, 1.0) { }

	virtual double& normalize(double& val)  { 
		if (val < 0.0) val = 0.0;
		else if (val > 1.0) val = 1.0;
		return val;
	}

	virtual ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	virtual ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}
};

// Implements the Inside type of semiring as described in Huang 2006
// +: +
// *: *
// 0: 0
// 1: 1
class InsideWeight : SemiringWeight<InsideWeight, double> {
public:
	InsideWeight(double value) : SemiringWeight<InsideWeight, double>(value, 0.0, 1.0) { }

	virtual double& normalize(double& val) { 
		if (val < 0.0) val = 0.0;
		return val;
	}

	virtual InsideWeight& operator+=(const InsideWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	virtual InsideWeight& operator*=(const InsideWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}
};

// Implements the Real type of semiring as described in Huang 2006
// +: min
// *: +
// 0: INF
// 1: 0
class RealWeight : SemiringWeight<RealWeight, double> {
public:
	RealWeight(double value) : SemiringWeight<RealWeight, double>(value, INF, 0.0) { }

	virtual double& normalize(double& val) { return val; }

	virtual RealWeight& operator+=(const RealWeight& rhs) {
		value = std::min(value, rhs.value);
		return *this;
	}
	virtual RealWeight& operator*=(const RealWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
};

// Implements the Inside type of semiring as described in Huang 2006
// +: min
// *: +
// 0: INF
// 1: 0
class TropicalWeight : SemiringWeight<TropicalWeight, double> {
public:
	TropicalWeight(double value) : SemiringWeight<TropicalWeight, double>(value, INF, 0.0) { }

	virtual double& normalize(double& val)  { 
		if (val < 0.0) val = 0.0;
		return val;
	}

	virtual TropicalWeight& operator+=(const TropicalWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	virtual TropicalWeight& operator*=(const TropicalWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}
};

// Implements the Counting type of semiring as described in Huang 2006
// +: +
// *: *
// 0: 0
// 1: 1
class CountingWeight : SemiringWeight<CountingWeight, int> {
public:
	CountingWeight(int value) : SemiringWeight<CountingWeight, int>(value, 0, 1) { }

	virtual int& normalize(int& val) { 
		if(val < 0) val = 0;
		return val;
	}

	virtual CountingWeight& operator+=(const CountingWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	virtual CountingWeight& operator*=(const CountingWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}
};

/* 

These two are how the python implemented the viterbi and prob, not sure if thats what you want

// Implements the Viterbi type of semiring
// +: max
// *: plus
// 0: -infinity
// 1: 0.0
class ViterbiWeight : SemiringWeight<ViterbiWeight, double> {
public:
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight, double>(value, -INF, 0.0) { value = value; }

	virtual ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	virtual ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	virtual bool is_zero() { return value <= annihlator; }
};

// Implements the Probability type of semiring
// +: max
// *: plus
// 0: 1.0
// 1: 0.0
class ProbWeight : SemiringWeight<ProbWeight, double> {
public:
	ProbWeight(double value) : SemiringWeight<ProbWeight, double>(value, 1.0, 0.0) { value = value; }

	virtual ProbWeight& operator+=(const ProbWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	virtual ProbWeight& operator*=(const ProbWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	virtual bool is_zero() { return value == annihlator; }
};


Not sure the intention of this semi-ring:

// Implements the Hypergraph type of semiring
// +: combine edge lists, forget nodes??
// *: combine node lists, forget edges??
// 0: empty object flagged as not zero?? 
// 1: empty object flagged as zero??
class HypergraphWeight : SemiringWeight<HypergraphWeight, double> {
public:
	HypergraphWeight(double value) : SemiringWeight<HypergraphWeight, double>(value, 0.0, 1.0) { value = value; }

	virtual HypergraphWeight& operator+=(const HypergraphWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	virtual HypergraphWeight& operator*=(const HypergraphWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	virtual bool is_zero() { return value == annihlator; }
};
*/

#endif // HYPERGRAPH_SEMIRING_H_