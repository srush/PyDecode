// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>
#include "Hypergraph/Factory.h"
#include "Hypergraph/Hypergraph.h"
#include "./common.h"


/**
 * An untemplated base class for use in registering types
 */
class BaseSemiring {
public:
	BaseSemiring() : value(zero().value) {}
	BaseSemiring(double val) : value(val) {}
	virtual ~BaseSemiring() { };
	static BaseSemiring one() { return BaseSemiring(1.0); }
	static BaseSemiring zero() { return BaseSemiring(0.0); }
	BaseSemiring identity() { return one(); }
	BaseSemiring annihlator() { return zero(); }
	BaseSemiring& operator+=(const BaseSemiring& rhs) {
		value = value + rhs.value;
		return *this;
	}
	BaseSemiring& operator*=(const BaseSemiring& rhs) {
		value = value * rhs.value;
		return *this;
	}
	friend bool operator==(const BaseSemiring& lhs, const BaseSemiring& rhs);
	friend BaseSemiring operator+(BaseSemiring lhs, const BaseSemiring &rhs);
	friend BaseSemiring operator*(BaseSemiring lhs, const BaseSemiring &rhs);
protected:
	double value;
};


/**
 * A templated base class of a potential with traits of a semiring
 * including + and * operators, and annihlator/identity elements.
 */
template<typename ValType, typename SemiringPotential>
class BaseSemiringPotential : public BaseSemiring {
public:
	BaseSemiringPotential(const SemiringPotential& other)
		: value(other.value) {}
	BaseSemiringPotential(ValType val) : value(normalize(val)) {}
	BaseSemiringPotential() : value(zero()) {}

	operator ValType() const { return value; }

	SemiringPotential& operator=(SemiringPotential rhs) {
		normalize(rhs.value);
		std::swap(value, rhs.value);
		return *this;
	}

	SemiringPotential& operator=(ValType rhs) {
		normalize(rhs);
		std::swap(value, rhs);
		return *this;
	}

	static SemiringPotential add(SemiringPotential lhs, const SemiringPotential &rhs) {
		return lhs += rhs;
	}

	static SemiringPotential times(SemiringPotential lhs, const SemiringPotential &rhs) {
		return lhs *= rhs;
	}

	friend bool operator==(const SemiringPotential& lhs,const SemiringPotential& rhs) {
		return lhs.value == rhs.value;
	}

	friend SemiringPotential operator+(SemiringPotential lhs, const SemiringPotential &rhs) {
		lhs += rhs;
		return lhs;
	}
	friend SemiringPotential operator*(SemiringPotential lhs, const SemiringPotential &rhs) {
		lhs *= rhs;
		return lhs;
	}

	SemiringPotential& operator+=(const SemiringPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}
	SemiringPotential& operator*=(const SemiringPotential& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static SemiringPotential one() { return SemiringPotential(1.0); }
	static SemiringPotential zero() { return SemiringPotential(0.0); }
	static ValType randValue() { return dRand(zero(), one()); }

	// Determines range of acceptable values
	ValType& normalize(ValType& val) { return val; }

protected:
	ValType value;
};

/**
 * Implements the Viterbi type of semiring as described in Huang 2006.
 * +: max
 * *: *
 * 0: 0
 * 1: 1
 */
class ViterbiPotential : public BaseSemiringPotential<double, ViterbiPotential> {
public:
	ViterbiPotential(double value) : BaseSemiringPotential<double, ViterbiPotential>(normalize(value)) { }
	ViterbiPotential() : BaseSemiringPotential<double, ViterbiPotential>() { }

	ViterbiPotential& operator+=(const ViterbiPotential& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	ViterbiPotential& operator*=(const ViterbiPotential& rhs) {
		value = value * rhs.value;
		return *this;
	}

	double& normalize(double& val) {
		if (val < 0.0) val = 0.0;
		else if (val > 1.0) val = 1.0;
		return val;
	}

	static ViterbiPotential one() { return ViterbiPotential(1.0); }
	static ViterbiPotential zero() { return ViterbiPotential(0.0); }

// protected:
	REGISTER_TYPE_DECLARATION(ViterbiPotential);
};


/**
 * Implements the log-space Viterbi type of semiring.
 * +: max
 * *: +
 * 0: -INF
 * 1: 0
 */
class LogViterbiPotential : public BaseSemiringPotential<double, LogViterbiPotential> {
public:
	LogViterbiPotential(double value) :
		BaseSemiringPotential<double, LogViterbiPotential>(normalize(value)) { }
	LogViterbiPotential() :
		BaseSemiringPotential<double, LogViterbiPotential>(LogViterbiPotential::zero()) {}

	LogViterbiPotential& operator+=(const LogViterbiPotential& rhs) {
		value = std::max(value, rhs.value);
		return *this;
	}
	LogViterbiPotential& operator*=(const LogViterbiPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}

	double& normalize(double& val) {
		return val = val < -INF ? -INF : val;
	}

	static LogViterbiPotential one() { return LogViterbiPotential(0.0); }
	static LogViterbiPotential zero() { return LogViterbiPotential(-INF); }


// protected:
	REGISTER_TYPE_DECLARATION(LogViterbiPotential);
};


/**
 * Implements the Boolean type of semiring as described in Huang 2006
 * +: logical or
 * *: logical and
 * 0: false
 * 1: true
 */
class BoolPotential : public BaseSemiringPotential<bool, BoolPotential> {
public:
	BoolPotential(bool value) : BaseSemiringPotential<bool, BoolPotential>(normalize(value)) { }
	BoolPotential() : BaseSemiringPotential<bool, BoolPotential>() { }

	BoolPotential& operator+=(const BoolPotential& rhs) {
		value = value || rhs.value;
		return *this;
	}
	BoolPotential& operator*=(const BoolPotential& rhs) {
		value = value && rhs.value;
		return *this;
	}

	static BoolPotential one() { return BoolPotential(true); }
	static BoolPotential zero() { return BoolPotential(false); }
	static bool randValue() { return rand()/RAND_MAX > .5 ? true : false; }


// protected:
	REGISTER_TYPE_DECLARATION(BoolPotential);
};

/**
 * Implements the Inside type of semiring as described in Huang 2006
 * +: +
 * *: *
 * 0: 0
 * 1: 1
 */
class InsidePotential : public BaseSemiringPotential<double, InsidePotential> {
public:
	InsidePotential(double value) : BaseSemiringPotential<double, InsidePotential>(normalize(value)) { }
	InsidePotential() : BaseSemiringPotential<double, InsidePotential>(zero()) { }

	InsidePotential& operator+=(const InsidePotential& rhs) {
		value = value + rhs.value;
		return *this;
	}
	InsidePotential& operator*=(const InsidePotential& rhs) {
		value = value * rhs.value;
		return *this;
	}

		friend InsidePotential operator/(InsidePotential lhs, const InsidePotential &rhs) {
			lhs.value /= rhs.value;
			return lhs;
	}

	static InsidePotential one() { return InsidePotential(1.0); }
	static InsidePotential zero() { return InsidePotential(0.0); }

	double& normalize(double& val) {
		if (val < 0.0) val = 0.0;
		if (val >= 1.0) val = 1.0;
		return val;
	}

// protected:
	REGISTER_TYPE_DECLARATION(InsidePotential);
};

/**
 * Implements the Real type of semiring as described in Huang 2006
 * +: min
 * *: +
 * 0: INF
 * 1: 0
 */
class RealPotential : public BaseSemiringPotential<double, RealPotential> {
public:
	RealPotential(double value) : BaseSemiringPotential<double, RealPotential>(normalize(value)) { }
	RealPotential() : BaseSemiringPotential<double, RealPotential>(zero()) { }

	RealPotential& operator+=(const RealPotential& rhs) {
		value = std::min(value, rhs.value);
		return *this;
	}
	RealPotential& operator*=(const RealPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}

	static RealPotential one() { return RealPotential(0.0); }
	static RealPotential zero() { return RealPotential(INF); }
	static double randValue() { return dRand(one(), zero()); }


// protected:
	REGISTER_TYPE_DECLARATION(RealPotential);
};

/**
 * Implements the Inside type of semiring as described in Huang 2006
 * +: min
 * *: +
 * 0: INF
 * 1: 0
 */
class TropicalPotential : public BaseSemiringPotential<double, TropicalPotential> {
public:
	TropicalPotential(double value) : BaseSemiringPotential<double, TropicalPotential>(normalize(value)) { }
	TropicalPotential() : BaseSemiringPotential<double, TropicalPotential>(zero()) { }

	TropicalPotential& operator+=(const TropicalPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}
	TropicalPotential& operator*=(const TropicalPotential& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static TropicalPotential one() { return TropicalPotential(0.0); }
	static TropicalPotential zero() { return TropicalPotential(INF); }
	static double randValue() { return dRand(one(), zero()); }

	double& normalize(double& val) {
		if (val < 0.0) val = 0.0;
		return val;
	}

// protected:
	REGISTER_TYPE_DECLARATION(TropicalPotential);
};




/**
 * Implements the Counting type of semiring as described in Huang 2006
 * +: +
 * *: *
 * 0: 0
 * 1: 1
 */
class CountingPotential : public BaseSemiringPotential<int, CountingPotential> {
public:
	CountingPotential(int value) : BaseSemiringPotential<int, CountingPotential>(normalize(value)) { }
	CountingPotential() : BaseSemiringPotential<int, CountingPotential>(zero()) { }

	CountingPotential& operator+=(const CountingPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}
	CountingPotential& operator*=(const CountingPotential& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static CountingPotential one() { return CountingPotential(1); }
	static CountingPotential zero() { return CountingPotential(0); }
	static int randValue() { return rand(); }

	int& normalize(int& val) {
		if(val < 0) val = 0;
		return val;
	}

// protected:
	REGISTER_TYPE_DECLARATION(CountingPotential);
};

/**
 * Comparison pair. *Experimental*
 * Type (s, t) op (s', t')
 * +: if (s > s') then (s, t) else (s', t')
 * *: (s * s', t * t')
 * 0: (0, 0)
 * 1: (1, 1)
 */
// template<typename SemiringComp, typename SemiringOther>
// class CompPotential : public BaseSemiringPotential<std::pair<SemiringComp, SemiringOther>, CompPotential<SemiringComp, SemiringOther> > {
// public:
// 	typedef std::pair<SemiringComp, SemiringOther> MyVal;
// 	typedef CompPotential<SemiringComp, SemiringOther> MyClass;
// 	using BaseSemiringPotential<MyVal, MyClass>::value;

// 	CompPotential(MyVal value) : BaseSemiringPotential<MyVal, MyClass>(normalize(value)) { }
// 	CompPotential() : BaseSemiringPotential<MyVal, MyClass>(zero()) { }

// 	MyClass& operator+=(const MyClass& rhs) {
// 		if (value.first < rhs.value.first) value = rhs.value;
// 		return *this;
// 	}

// 	MyClass& operator*=(const MyClass& rhs) {
// 		value.first = value.first * rhs.value.first;
// 		value.second = value.second * rhs.value.second;
// 		return *this;
// 	}

// 	static const MyClass one() { return MyClass(MyVal(SemiringComp::one(), SemiringOther::one())); }
// 	static const MyClass zero() { return MyClass(MyVal(SemiringComp::zero(), SemiringOther::zero())); }

// 	MyVal& normalize(MyVal& val) {
// 		val.first = normalize(val.first);
// 		val.second = normalize(val.second);
// 		return val;
// 	}

// protected:
// 	REGISTER_TYPE_DECLARATION(CompPotential);
// };


typedef pair<int, int> SparsePair;
typedef vector<SparsePair> SparseVector;

/**
 * Sparse vector. *Experimental*
 *
 * +: Elementwise min
 * *: Elementwise +
 * 0: Empty Vector
 * 1: Empty Vector
 */
class SparseVectorPotential : public BaseSemiringPotential<SparseVector, SparseVectorPotential> {
public:
	SparseVectorPotential(const SparseVector vec) : BaseSemiringPotential<SparseVector, SparseVectorPotential>(vec) { }
	SparseVectorPotential() : BaseSemiringPotential<SparseVector, SparseVectorPotential>(SparseVectorPotential::zero()) { }

	SparseVectorPotential& operator+=(const SparseVectorPotential& rhs) {
		return *this;
	}

	SparseVectorPotential& operator*=(const SparseVectorPotential& rhs);

	static SparseVectorPotential one() { return SparseVectorPotential(SparseVector()); }
	static SparseVectorPotential zero() { return SparseVectorPotential(SparseVector()); }
	static SparseVector randValue();

// protected:
	REGISTER_TYPE_DECLARATION(SparseVectorPotential);
};

/**
 * Tree. *Experimental*
 *
 * +: No action
 * *: NULL if either is NULL. Otherwise create a new node with rhs.value and this->value as tails
 * 0: Empty Vector
 * 1: Empty Vector
 */
class TreePotential : public BaseSemiringPotential<Hypernode *, TreePotential> {
public:
TreePotential(Hypernode *value) : BaseSemiringPotential<Hypernode *, TreePotential>(normalize(value)) { }
TreePotential() : BaseSemiringPotential<Hypernode *, TreePotential>(zero()) { }

	TreePotential& operator+=(const TreePotential& rhs) {
		return *this;
	}

	TreePotential& operator*=(const TreePotential& rhs) {
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

	static TreePotential one() {
		return TreePotential(new Hypernode(""));
	}
	static const TreePotential zero() { return TreePotential(NULL); }

// protected:
	// REGISTER_TYPE_DECLARATION(TreePotential);
};


// Classes used to associate projections with Hypergraphs

class HypergraphProjection;

template<typename SemiringType>
class HypergraphPotentials {
 public:
	HypergraphPotentials(const Hypergraph *hypergraph,
										const vector<SemiringType> &potentials,
										SemiringType bias)
	: hypergraph_(hypergraph),
		potentials_(potentials),
		bias_(bias) {
			assert(potentials.size() == hypergraph->edges().size());
	}

	HypergraphPotentials(const Hypergraph *hypergraph)
		: hypergraph_(hypergraph),
			potentials_(hypergraph->edges().size(), SemiringType::one()),
			bias_(SemiringType::one()) {}

	SemiringType dot(const Hyperpath &path) const {
		path.check(*hypergraph_);
		SemiringType score = SemiringType::one();
		foreach (HEdge edge, path.edges()) {
			score *= potentials_[edge->id()];
		}
		return score * bias_;
	}

	SemiringType score(HEdge edge) const { return potentials_[edge->id()]; }
	const SemiringType& operator[] (HEdge edge) const {
		return potentials_[edge->id()];
	}
	SemiringType& operator[] (HEdge edge) {
		return potentials_[edge->id()];
	}

	const SemiringType &bias() const { return bias_; }
	SemiringType &bias() { return bias_; }

	HypergraphPotentials<SemiringType> *project_potentials(
		const HypergraphProjection &projection) const;

	/**
	 * Pairwise "times" with another set of potentials.
	 *
	 * @return New hypergraphpotentials.
	 */
	HypergraphPotentials <SemiringType> *times(
			const HypergraphPotentials<SemiringType> &potentials) const;

	void check(const Hypergraph &graph) const {
		if (!graph.same(*hypergraph_)) {
			throw HypergraphException("Hypergraph does not match potentials.");
		}
	}

	void check(const HypergraphPotentials<SemiringType> &potentials) const {
		if (!potentials.hypergraph_->same(*hypergraph_)) {
			throw HypergraphException("Hypergraph potentials do not match potentials.");
		}
	}

	const Hypergraph *hypergraph() const { return hypergraph_; }

 protected:
	const Hypergraph *hypergraph_;
	vector<SemiringType> potentials_;
	SemiringType bias_;
};


class HypergraphProjection {
 public:
	HypergraphProjection(const Hypergraph *original,
											 const Hypergraph *_new_graph,
											 const vector<HNode> *node_map,
											 const vector<HEdge> *edge_map)
			: original_graph(original),
			new_graph(_new_graph),
			node_map_(node_map),
			edge_map_(edge_map) {
				assert(node_map->size() == original_graph->nodes().size());
				assert(edge_map->size() == original_graph->edges().size());
#ifndef NDEBUG
				foreach (HNode node, *node_map) {
					assert(node == NULL ||
								 node->id() < (int)_new_graph->nodes().size());
				}
				foreach (HEdge edge, *edge_map) {
					assert(edge == NULL ||
								 edge->id() < (int)_new_graph->edges().size());
				}
#endif
			}

	~HypergraphProjection() {
		delete node_map_;
		delete edge_map_;
	}

	static HypergraphProjection *project_hypergraph(
			const Hypergraph *hypergraph,
			const HypergraphPotentials<BoolPotential> &edge_mask);

	HEdge project(HEdge original) const {
		return (*edge_map_)[original->id()];
	}

	HNode project(HNode original) const {
		return (*node_map_)[original->id()];
	}

	const Hypergraph *original_graph;
	const Hypergraph *new_graph;

 private:

	// Owned.
	const vector<HNode> *node_map_;
	const vector<HEdge> *edge_map_;
};

#endif // HYPERGRAPH_SEMIRING_H_
