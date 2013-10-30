
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>

#include "./common.h"


// A base class of a weight with traits of a semiring
// including + and * operators, and annihlator/identity elements.
template<typename ValType>
class SemiringWeight {
public:
	SemiringWeight(const SemiringWeight& other)
		: value(normalize(other.value)) {}
	SemiringWeight(ValType val) : value(normalize(val)) {}

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
class ViterbiWeight : public SemiringWeight<double> {
public:
	ViterbiWeight(double value) : SemiringWeight<double>(normalize(value)) { }
    ViterbiWeight() : SemiringWeight<double>(ViterbiWeight::zero()) {}
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

// Implements the Boolean type of semiring as described in Huang 2006
// +: logical or
// *: logical and
// 0: false
// 1: true
class BoolWeight : public SemiringWeight<bool> {
public:
	BoolWeight(bool value) : SemiringWeight<bool>(normalize(value)) { }

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
class InsideWeight : public SemiringWeight<double> {
public:
	InsideWeight(double value) : SemiringWeight<double>(normalize(value)) { }

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
class RealWeight : public SemiringWeight<double> {
public:
	RealWeight(double value) : SemiringWeight<double>(normalize(value)) { }

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
class TropicalWeight : public SemiringWeight<double> {
public:
	TropicalWeight(double value) : SemiringWeight<double>(normalize(value)) { }

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
class CountingWeight : public SemiringWeight<int> {
public:
	CountingWeight(int value) : SemiringWeight<int>(normalize(value)) { }

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

#endif // HYPERGRAPH_SEMIRING_H_
