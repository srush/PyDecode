// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>
#include "Hypergraph/Factory.h"
#include "Hypergraph/Hypergraph.h"
#include "./common.h"


class ViterbiPotential {
public:
	typedef double ValType;
	static inline ValType add(ValType lhs, const ValType &rhs) {
		lhs = std::max(lhs, rhs);
		return normalize(lhs);
	}

	static inline ValType times(ValType lhs, const ValType &rhs) {
		lhs *= rhs;
		return normalize(lhs);
	}

	static inline ValType& normalize(ValType& val) {
		if (val < 0.0) val = 0.0;
		else if (val > 1.0) val = 1.0;
		return val;
	}

	static inline ValType zero() { return 0.0; }
	static inline ValType one() { return 1.0; }

	static inline ValType randValue() { return dRand(0.0,1.0); }
};

class LogViterbiPotential : public ViterbiPotential {
public:
	static inline ValType add(ValType lhs, const ValType &rhs) {
		lhs = std::max(lhs, rhs);
		return normalize(lhs);
	}

	static inline ValType times(ValType lhs, const ValType &rhs) {
		lhs += rhs;
		return normalize(lhs);
	}

	static inline ValType zero() { return -INF; }
	static inline ValType one() { return 0.0; }

	static inline ValType& normalize(ValType& val) {
		return val = val < -INF ? -INF : val;
	}

	static inline ValType randValue() { return dRand(-INF, 0.0); }
};

class InsidePotential : public ViterbiPotential {
public:
	static inline ValType add(ValType lhs, const ValType &rhs) {
		lhs += rhs;
		return normalize(lhs);
	}

	static inline ValType times(ValType lhs, const ValType &rhs) {
		lhs *= rhs;
		return normalize(lhs);
	}

	static inline ValType zero() { return 0.0; }
	static inline ValType one() { return 1.0; }

	static inline ValType& normalize(ValType& val) {
		return val = val < 0.0 ? 0.0 : val;
	}

	static inline ValType randValue() { return dRand(0.0,1.0); }
};

class RealPotential : public ViterbiPotential {
public:
	static inline ValType add(ValType lhs, const ValType &rhs) {
		lhs = std::min(lhs, rhs);
		return normalize(lhs);
	}

	static inline ValType times(ValType lhs, const ValType &rhs) {
		lhs += rhs;
		return normalize(lhs);
	}

	static inline ValType zero() { return INF; }
	static inline ValType one() { return 0.0; }

	static inline ValType& normalize(ValType& val) {
		return val = val > INF ? INF : val;
	}

	static ValType randValue() { return dRand(0.0, INF); }
};

class TropicalPotential : public ViterbiPotential {
public:
	static inline ValType add(ValType lhs, const ValType &rhs) {
		lhs = std::min(lhs, rhs);
		return normalize(lhs);
	}

	static inline ValType times(ValType lhs, const ValType &rhs) {
		lhs += rhs;
		return normalize(lhs);
	}

	static inline ValType zero() { return INF; }
	static inline ValType one() { return 0.0; }

	static inline ValType& normalize(ValType& val) {
		return val = val > INF ? INF : val;
	}

	static ValType randValue() { return dRand(0.0, INF); }
};

class BoolPotential {
public:
	typedef bool ValType;
	static inline ValType add(const ValType& lhs, const ValType &rhs) {
		return lhs || rhs;
	}
	static inline ValType times(const ValType& lhs, const ValType &rhs) {
		return lhs && rhs;
	}

	static inline ValType& normalize(ValType& val) {
		return val;
	}

	static inline ValType one() { return true; }
	static inline ValType zero() { return false; }

	static inline ValType randValue() { return dRand(0.0,1.0) > .5; }
};

class CountingPotential {
public:
	typedef int ValType;
	static inline ValType add(ValType lhs, const ValType &rhs) {
		lhs += rhs;
		return normalize(lhs);
	}

	static inline ValType times(ValType lhs, const ValType &rhs) {
		lhs *= rhs;
		return normalize(lhs);
	}

	static inline ValType& normalize(ValType& val) {
		return val = val < 0 ? 0 : val;
	}

	static inline ValType one() { return 1; }
	static inline ValType zero() { return 0; }

	static inline ValType randValue() {
		return rand();
	}
};


// Classes used to associate projections with Hypergraphs

class HypergraphProjection;

template<typename SemiringType>
class HypergraphPotentials {
	typedef SemiringType S;
	typedef typename SemiringType::ValType V;
 public:
	HypergraphPotentials(const Hypergraph *hypergraph,
						const vector<V> &potentials,
						V bias)
	: hypergraph_(hypergraph),
		potentials_(potentials),
		bias_(bias) {
			assert(potentials.size() == hypergraph->edges().size());
	}

	HypergraphPotentials(const Hypergraph *hypergraph)
		: hypergraph_(hypergraph),
			potentials_(hypergraph->edges().size(), SemiringType::one()),
			bias_(SemiringType::one()) {}

	V dot(const Hyperpath &path) const {
		path.check(*hypergraph_);
		V score = SemiringType::one();
		foreach (HEdge edge, path.edges()) {
			score *= potentials_[edge->id()];
		}
		return score * bias_;
	}

	V score(HEdge edge) const { return potentials_[edge->id()]; }
	inline V operator[] (HEdge edge) const {
		return potentials_[edge->id()];
	}
	// inline V& operator[] (HEdge edge) {
	// 	return potentials_[edge->id()];
	// }

	void insert(const HEdge& edge, const V& val) {
		potentials_[edge->id()] = val;
	}

	const V &bias() const { return bias_; }
	V &bias() { return bias_; }

	HypergraphPotentials<S> *project_potentials(
		const HypergraphProjection &projection) const;

	/**
	 * Pairwise "times" with another set of potentials.
	 *
	 * @return New hypergraphpotentials.
	 */
	HypergraphPotentials<S> *times(
			const HypergraphPotentials<S> &potentials) const;

	void check(const Hypergraph &graph) const {
		if (!graph.same(*hypergraph_)) {
			throw HypergraphException("Hypergraph does not match potentials.");
		}
	}

	void check(const HypergraphPotentials<S> &potentials) const {
		if (!potentials.hypergraph_->same(*hypergraph_)) {
			throw HypergraphException("Hypergraph potentials do not match potentials.");
		}
	}

	const Hypergraph *hypergraph() const { return hypergraph_; }

 protected:
	const Hypergraph *hypergraph_;
	vector<V> potentials_;
	V bias_;
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
