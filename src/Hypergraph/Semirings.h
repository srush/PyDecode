
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>

#include "./common.h"


// A base class of a weight with traits of a semiring
// including + and * operators, and annihlator/identity elements.
template<typename DerivedWeight, typename ValType>
class SemiringWeight {
public:
	operator ValType() const { return value; }
	
	DerivedWeight& operator=(DerivedWeight rhs) {
		normalize(rhs.value);
		std::swap(value, rhs.value);
		return *this;
	}

	DerivedWeight& operator=(ValType rhs) {
		normalize(rhs);
		std::swap(value, rhs);
		return *this;
	}

	friend DerivedWeight operator+(DerivedWeight lhs, const DerivedWeight &rhs) {
		lhs += rhs;
		return lhs;
	}
	friend DerivedWeight operator*(DerivedWeight lhs, const DerivedWeight &rhs) {
		lhs *= rhs;
		return lhs;
	}

	DerivedWeight& operator+=(const DerivedWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	DerivedWeight& operator*=(const DerivedWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const ValType one() { return 1.0; }
	static const ValType zero() { return 0.0; }

	// Determines range of acceptable values
	ValType& normalize(ValType& val) { return val; };

protected:
	SemiringWeight(const SemiringWeight& other)
		: value(other.value) {}
	SemiringWeight(ValType val) : value(val) {}

	ValType value;
};

// Implements the Viterbi type of semiring as described in Huang 2006
// +: max
// *: *
// 0: 0
// 1: 1
class ViterbiWeight : public SemiringWeight<ViterbiWeight, double> {
public:
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight, double>(value) { }

	ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	double& normalize(double& val) const { 
		if (val < 0.0) val = 0.0;
		else if (val > 1.0) val = 1.0;
		return val;
	}

	static const double one() { return 1.0; }
	static const double zero() { return 0.0; }
};

// Implements the Boolean type of semiring as described in Huang 2006
// +: logical or
// *: logical and
// 0: false
// 1: true
class BoolWeight : public SemiringWeight<BoolWeight, bool> {
public:
	BoolWeight(bool value) : SemiringWeight<BoolWeight, bool>(value) { }

	BoolWeight& operator+=(const BoolWeight& rhs) {
		value = value || rhs.value;
		return *this;
	}
	BoolWeight& operator*=(const BoolWeight& rhs) {
		value = value && rhs.value;
		return *this;
	}

	static const bool one() { return true; }
	static const bool zero() { return false; }

	bool& normalize(bool& val) const { return val; }
};

// Implements the Inside type of semiring as described in Huang 2006
// +: +
// *: *
// 0: 0
// 1: 1
class InsideWeight : public SemiringWeight<InsideWeight, double> {
public:
	InsideWeight(double value) : SemiringWeight<InsideWeight, double>(value) { }

	InsideWeight& operator+=(const InsideWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	InsideWeight& operator*=(const InsideWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const double one() { return 1.0; }
	static const double zero() { return 0.0; }

	double& normalize(double& val) const { 
		if (val < 0.0) val = 0.0;
		return val;
	}
};

// Implements the Real type of semiring as described in Huang 2006
// +: min
// *: +
// 0: INF
// 1: 0
class RealWeight : public SemiringWeight<RealWeight, double> {
public:
	RealWeight(double value) : SemiringWeight<RealWeight, double>(value) { }

	RealWeight& operator+=(const RealWeight& rhs) {
		value = std::min(value, rhs.value);
		return *this;
	}
	RealWeight& operator*=(const RealWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	static const double one() { return 0.0; }
	static const double zero() { return INF; }

	double& normalize(double& val) const { return val; }
};

// Implements the Inside type of semiring as described in Huang 2006
// +: min
// *: +
// 0: INF
// 1: 0
class TropicalWeight : public SemiringWeight<TropicalWeight, double> {
public:
	TropicalWeight(double value) : SemiringWeight<TropicalWeight, double>(value) { }

	TropicalWeight& operator+=(const TropicalWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	TropicalWeight& operator*=(const TropicalWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const double one() { return 0.0; }
	static const double zero() { return INF; }

	double& normalize(double& val) const { 
		if (val < 0.0) val = 0.0;
		return val;
	}
};

// Implements the Counting type of semiring as described in Huang 2006
// +: +
// *: *
// 0: 0
// 1: 1
class CountingWeight : public SemiringWeight<CountingWeight, int> {
public:
	CountingWeight(int value) : SemiringWeight<CountingWeight, int>(value) { }

	CountingWeight& operator+=(const CountingWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	CountingWeight& operator*=(const CountingWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const int one() { return 1; }
	static const int zero() { return 0; }

	int& normalize(int& val) const { 
		if(val < 0) val = 0;
		return val;
	}
};


/* 

These two are how the python implemented the viterbi and prob, not sure if thats what you want

// Implements the Viterbi type of semiring
// +: max
// *: plus
// 0: -infinity
// 1: 0.0
class ViterbiWeight : public SemiringWeight<ViterbiWeight, double> {
public:
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight, double>(-value) { }

	ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	bool is_zero() { return value <= annihlator; }
};

// Implements the Probability type of semiring
// +: max
// *: plus
// 0: 1.0
// 1: 0.0
class ProbWeight : public SemiringWeight<ProbWeight, double> {
public:
	ProbWeight(double value) : SemiringWeight<ProbWeight, double>(value) { }

	ProbWeight& operator+=(const ProbWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	ProbWeight& operator*=(const ProbWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	bool is_zero() { return value == annihlator; }
};


Not sure the intention of this semi-ring:

// Implements the Hypergraph type of semiring
// +: combine edge lists, forget nodes??
// *: combine node lists, forget edges??
// 0: empty object flagged as not zero?? 
// 1: empty object flagged as zero??
class HypergraphWeight : public SemiringWeight<HypergraphWeight, double> {
public:
	HypergraphWeight(double value) : SemiringWeight<HypergraphWeight, double>(value) { }

	HypergraphWeight& operator+=(const HypergraphWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	HypergraphWeight& operator*=(const HypergraphWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

	bool is_zero() { return value == annihlator; }
};
*/

#endif // HYPERGRAPH_SEMIRING_H_