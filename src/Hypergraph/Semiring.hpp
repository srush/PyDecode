
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>

#include <boost/serialization/strong_typedef.hpp>

#include "../common.h"



// A virtual base class of a weight with traits of a semiring
// including + and * operators, and annihlator/identity elements.
template<typename DerivedWeight>
class SemiringWeight {
public:
	operator double() const { return value; }
	DerivedWeight& operator=(double rhs) { 
		std::swap(this->value, rhs);
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

	const double one() const { return identity; }
	const double zero() const { return annihlator; }

	virtual DerivedWeight& operator+=(const DerivedWeight& rhs) = 0;
	virtual DerivedWeight& operator*=(const DerivedWeight& rhs) = 0;

	virtual bool is_zero() = 0;

protected:
	SemiringWeight() {}
	SemiringWeight(double ann, double id) : annihlator(ann), identity(id) {}
	double value;
	const double annihlator;
	const double identity;
};

// Implements the Viterbi type of semiring weight as described in Huang 2006
// +: max
// *: *
// 0: 0
// 1: 1
class ViterbiWeight : SemiringWeight<ViterbiWeight> {
public:
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight>(0.0, 1.0) { this->value = value; }

	virtual ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		this->value = std::max(this->value, rhs.value);
		return *this;
	}
	virtual ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		this->value = this->value + rhs.value;
		return *this;
	}

	virtual bool is_zero() { return this->value <= this->annihlator; }
};

// Implements the Boolean type of semiring weight as described in Huang 2006
// +: logical or
// *: logical and
// 0: 0
// 1: 1
class BoolWeight : SemiringWeight<BoolWeight> {
public:
	BoolWeight(bool value) : value(value), annihlator(false), identity(true) {}

/*  should really delete these, but only available in c++11
	operator double() const = delete;
	BoolWeight& operator=(double rhs) = delete;
	const double one() const = delete;
	const double zero() const = delete;
*/
	operator bool() const { return value; }
	BoolWeight& operator=(bool rhs) { 
		std::swap(this->value, rhs);
		return *this;
	}

	const bool one() const { return identity; }
	const bool zero() const { return annihlator; }

	virtual BoolWeight& operator+=(const BoolWeight& rhs) {
		this->value = this->value || rhs.value;
		return *this;
	}
	virtual BoolWeight& operator*=(const BoolWeight& rhs) {
		this->value = this->value && rhs.value;
		return *this;
	}

	virtual bool is_zero() { return this->value; }

protected:
	bool value;
	const bool annihlator /*= false  c++11*/;
	const bool identity /*= true c++11*/;

};

/* 

These two are how the python implemented the one and zero

// Implements the Viterbi type of semiring weight
// +: max
// *: plus
// 0: -infinity
// 1: 0.0
class ViterbiWeight : SemiringWeight<ViterbiWeight> {
public:
	ViterbiWeight(double value) : SemiringWeight<ViterbiWeight>(-INF, 0.0) { this->value = value; }

	virtual ViterbiWeight& operator+=(const ViterbiWeight& rhs) {
		this->value = std::max(this->value, rhs.value);
		return *this;
	}
	virtual ViterbiWeight& operator*=(const ViterbiWeight& rhs) {
		this->value = this->value + rhs.value;
		return *this;
	}

	virtual bool is_zero() { return this->value <= this->annihlator; }
};

// Implements the Probability type of semiring weight
// +: max
// *: plus
// 0: 1.0
// 1: 0.0
class ProbWeight : SemiringWeight<ProbWeight> {
public:
	ProbWeight(double value) : SemiringWeight<ProbWeight>(1.0, 0.0) { this->value = value; }

	virtual ProbWeight& operator+=(const ProbWeight& rhs) {
		this->value = std::max(this->value, rhs.value);
		return *this;
	}
	virtual ProbWeight& operator*=(const ProbWeight& rhs) {
		this->value = this->value + rhs.value;
		return *this;
	}

	virtual bool is_zero() { return this->value == this->annihlator; }
};


Not sure the intention of this semi-ring:

// Implements the Hypergraph type of semiring weight
// +: combine edge lists, forget nodes??
// *: combine node lists, forget edges??
// 0: empty object flagged as not zero?? 
// 1: empty object flagged as zero??
class HypergraphWeight : SemiringWeight<HypergraphWeight> {
public:
	HypergraphWeight(double value) : SemiringWeight<HypergraphWeight>(0.0, 1.0) { this->value = value; }

	virtual HypergraphWeight& operator+=(const HypergraphWeight& rhs) {
		this->value = std::max(this->value, rhs.value);
		return *this;
	}
	virtual HypergraphWeight& operator*=(const HypergraphWeight& rhs) {
		this->value = this->value + rhs.value;
		return *this;
	}

	virtual bool is_zero() { return this->value == this->annihlator; }
};
*/

#endif // HYPERGRAPH_SEMIRING_H_