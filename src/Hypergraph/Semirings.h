
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>

#include "./common.h"


// A virtual base class of a weight with traits of a semiring
// including + and * operators, and annihlator/identity elements.
template<typename DerivedWeight, typename ValType>
class SemiringWeight {
public:
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

	operator ValType() const { return value; }

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
	SemiringWeight(const SemiringWeight& other)
		: value(other.value), annihlator(other.annihlator), identity(other.identity) {}
	SemiringWeight(ValType ann, ValType id) : annihlator(ann), identity(id) {}

	// Determines range of acceptable values
	virtual ValType& normalize(ValType& val) = 0;

	ValType value;
	ValType annihlator;
	ValType identity;
};

// Implements a weight that behaves just like a regular double
// DEPRECATED: Use InsideWeight instead
// +: +
// *: *
// 0: 0
// 1: 1
class DoubleWeight : public SemiringWeight<DoubleWeight, double> {
public:
	DoubleWeight(const DoubleWeight& other) : SemiringWeight<DoubleWeight, double>(0.0, 1.0) { this->value = other.value; }
	DoubleWeight(double value) : SemiringWeight<DoubleWeight, double>(0.0, 1.0) { this->value = value; }
	DoubleWeight() : SemiringWeight<DoubleWeight, double>(0.0, 1.0) { this->value = 0.0; }

	virtual DoubleWeight& operator+=(const DoubleWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	virtual DoubleWeight& operator*=(const DoubleWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

protected:
	virtual double& normalize(double& val) { 
		return val;
	}
};

// Implements the Boolean type of semiring as described in Huang 2006
// +: logical or
// *: logical and
// 0: 0
// 1: 1
class BoolWeight : public SemiringWeight<BoolWeight, bool> {
public:
	BoolWeight(bool value) : SemiringWeight<BoolWeight, bool>(false, true) { this->value = normalize(value); }

	virtual BoolWeight& operator+=(const BoolWeight& rhs) {
		value = value || rhs.value;
		return *this;
	}
	virtual BoolWeight& operator*=(const BoolWeight& rhs) {
		value = value && rhs.value;
		return *this;
	}
	
protected:
	virtual bool& normalize(bool& val) { return val; }
};

// Implements the Viterbi type of semiring as described in Huang 2006
// +: max
// *: *
// 0: 0
// 1: 1
class ViterbiWeight : public SemiringWeight<ViterbiWeight, double> {
public:
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight, double>(0.0, 1.0) { this->value = normalize(value); }

	virtual ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	virtual ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

protected:
	virtual double& normalize(double& val)  { 
		if (val < 0.0) val = 0.0;
		else if (val > 1.0) val = 1.0;
		return val;
	}
};

// Implements the Inside type of semiring as described in Huang 2006
// +: +
// *: *
// 0: 0
// 1: 1
class InsideWeight : public SemiringWeight<InsideWeight, double> {
public:
	InsideWeight(double value) : SemiringWeight<InsideWeight, double>(0.0, 1.0) { this->value = normalize(value); }

	virtual InsideWeight& operator+=(const InsideWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	virtual InsideWeight& operator*=(const InsideWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

protected:
	virtual double& normalize(double& val) { 
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
	RealWeight(double value) : SemiringWeight<RealWeight, double>(INF, 0.0) { this->value = normalize(value); }

	virtual RealWeight& operator+=(const RealWeight& rhs) {
		value = std::min(value, rhs.value);
		return *this;
	}
	virtual RealWeight& operator*=(const RealWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}

protected:
	virtual double& normalize(double& val) { return val; }
};

// Implements the Inside type of semiring as described in Huang 2006
// +: min
// *: +
// 0: INF
// 1: 0
class TropicalWeight : public SemiringWeight<TropicalWeight, double> {
public:
	TropicalWeight(double value) : SemiringWeight<TropicalWeight, double>(INF, 0.0) { this->value = normalize(value); }

	virtual TropicalWeight& operator+=(const TropicalWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	virtual TropicalWeight& operator*=(const TropicalWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

protected:
	virtual double& normalize(double& val)  { 
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
	CountingWeight(int value) : SemiringWeight<CountingWeight, int>(0, 1) { this->value = normalize(value); }

	virtual CountingWeight& operator+=(const CountingWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	virtual CountingWeight& operator*=(const CountingWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

protected:
	virtual int& normalize(int& val) { 
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
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight, double>(-INF, 0.0) { this->value = normalize(value);}

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
class ProbWeight : public SemiringWeight<ProbWeight, double> {
public:
	ProbWeight(double value) : SemiringWeight<ProbWeight, double>(1.0, 0.0) { this->value = normalize(value);}

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
class HypergraphWeight : public SemiringWeight<HypergraphWeight, double> {
public:
	HypergraphWeight(double value) : SemiringWeight<HypergraphWeight, double>(0.0, 1.0) { this->value = normalize(value);}

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