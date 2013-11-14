
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>
#include "Hypergraph/Hypergraph.h"
#include "./common.h"

/**
 * A base class of a potential with traits of a semiring
 * including + and * operators, and annihlator/identity elements.
 */
template<typename ValType, typename SemiringPotential>
class BaseSemiringPotential {
public:
	BaseSemiringPotential(const SemiringPotential& other)
		: value(normalize(other.value)) {}
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
      return lhs + rhs;
    }

    static SemiringPotential times(SemiringPotential lhs, const SemiringPotential &rhs) {
      return lhs * rhs;
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

	static const SemiringPotential one() { return SemiringPotential(1.0); }
	static const SemiringPotential zero() { return SemiringPotential(0.0); }

	// Determines range of acceptable values
	ValType normalize(ValType val) { return val; };

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

	double normalize(double val) const {
		if (val < 0.0) val = 0.0;
		else if (val > 1.0) val = 1.0;
		return val;
	}

	static const ViterbiPotential one() { return ViterbiPotential(1.0); }
	static const ViterbiPotential zero() { return ViterbiPotential(0.0); }
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

     double normalize(double val) const {
       if (val < -INF) return -INF;
       return val;
     }

     static const LogViterbiPotential one() { return LogViterbiPotential(0.0); }
     static const LogViterbiPotential zero() { return LogViterbiPotential(-INF); }
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

	static const BoolPotential one() { return BoolPotential(true); }
	static const BoolPotential zero() { return BoolPotential(false); }

	bool normalize(bool val) const { return val; }
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

  InsidePotential() :
  BaseSemiringPotential<double, InsidePotential>(InsidePotential::zero()) {}

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

	static const InsidePotential one() { return InsidePotential(1.0); }
	static const InsidePotential zero() { return InsidePotential(0.0); }

	double normalize(double val) const {
		if (val < 0.0) val = 0.0;
        if (val >= 1.0) val = 1.0;
		return val;
	}
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

	RealPotential& operator+=(const RealPotential& rhs) {
		value = std::min(value, rhs.value);
		return *this;
	}
	RealPotential& operator*=(const RealPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}

	static const RealPotential one() { return RealPotential(0.0); }
	static const RealPotential zero() { return RealPotential(INF); }

	double normalize(double val) const { return val; }
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

	TropicalPotential& operator+=(const TropicalPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}
	TropicalPotential& operator*=(const TropicalPotential& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const TropicalPotential one() { return TropicalPotential(0.0); }
	static const TropicalPotential zero() { return TropicalPotential(INF); }

	double normalize(double val) const {
		if (val < 0.0) val = 0.0;
		return val;
	}
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

	CountingPotential& operator+=(const CountingPotential& rhs) {
		value = value + rhs.value;
		return *this;
	}
	CountingPotential& operator*=(const CountingPotential& rhs) {
		value = value * rhs.value;
		return *this;
	}

	static const CountingPotential one() { return CountingPotential(1); }
	static const CountingPotential zero() { return CountingPotential(0); }

	int normalize(int val) const {
		if(val < 0) val = 0;
		return val;
	}
};

/**
 * Comparison pair. *Experimental*
 * Type (s, t) op (s', t')
 * +: if (s > s') then (s, t) else (s', t')
 * *: (s * s', t * t')
 * 0: (0, 0)
 * 1: (1, 1)
 */
template<typename SemiringComp, typename SemiringOther>
class CompPotential : public BaseSemiringPotential<std::pair<SemiringComp, SemiringOther>, CompPotential<SemiringComp, SemiringOther> > {
public:
  typedef std::pair<SemiringComp, SemiringOther> MyVal;
  typedef CompPotential<SemiringComp, SemiringOther> MyClass;
  using BaseSemiringPotential<MyVal, MyClass>::value;

  CompPotential(MyVal value) : BaseSemiringPotential<MyVal, MyClass>(normalize(value)) { }

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

	SparseVectorPotential& operator*=(const SparseVectorPotential& rhs) {
      int i = 0, j = 0;
      SparseVector vec;
      while (i < value.size() || j < rhs.value.size()) {
        if (j >= rhs.value.size() || (i < value.size() && value[i].first < rhs.value[j].first)) {
          vec.push_back(pair<int, int>(value[i].first, value[i].second));
          ++i;
        } else if (i >= value.size() || (j < rhs.value.size() && value[i].first > rhs.value[j].first)) {
          vec.push_back(pair<int, int>(rhs.value[j].first, rhs.value[j].second));
          ++j;
        } else {
          vec.push_back(pair<int, int>(value[i].first, value[i].second + rhs.value[j].second));
          ++i;
          ++j;
        }
      }
      value = vec;
      return *this;
	}

	static const SparseVectorPotential one() { return SparseVectorPotential(SparseVector()); }
	static const SparseVectorPotential zero() { return SparseVectorPotential(SparseVector()); }

	int normalize(int val) const {
      return val;
	}
};



class TreePotential : public BaseSemiringPotential<Hypernode *, TreePotential> {
public:
TreePotential(Hypernode *value) : BaseSemiringPotential<Hypernode *, TreePotential>(normalize(value)) { }

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

	static const TreePotential one() {
      return TreePotential(new Hypernode(""));
    }
	static const TreePotential zero() { return TreePotential(NULL); }

	Hypernode *normalize(Hypernode *val) const {
		return val;
	}
};

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


template <>
inline double HypergraphPotentials<double>::dot(const Hyperpath &path) const {
  path.check(*hypergraph_);
  double score = 0.0;
  foreach (HEdge edge, path.edges()) {
    score += potentials_[edge->id()];
  }
  return score + bias_;
}

template<typename SemiringType>
HypergraphPotentials<SemiringType> *HypergraphPotentials<SemiringType>::times(const HypergraphPotentials<SemiringType> &other) const {
  check(other);
  vector<SemiringType> new_potentials(potentials_);
  for (uint i = 0; i < other.potentials_.size(); ++i) {
    new_potentials[i] *= other.potentials_[i];
  }
  return new HypergraphPotentials<SemiringType>(hypergraph_,
                               new_potentials,
                               bias_ * other.bias_);
}

template<typename SemiringType>
HypergraphPotentials<SemiringType> *HypergraphPotentials<SemiringType>::project_potentials(
    const HypergraphProjection &projection) const {
  vector<SemiringType> potentials(projection.new_graph->edges().size());
  foreach (HEdge edge, projection.original_graph->edges()) {
    HEdge new_edge = projection.project(edge);
    if (new_edge != NULL && new_edge->id() >= 0) {
      assert(new_edge->id() < projection.new_graph->edges().size());
      potentials[new_edge->id()] = score(edge);
    }
  }
  return new HypergraphPotentials<SemiringType>(projection.new_graph, potentials, bias_);
}



inline HypergraphProjection *HypergraphProjection::project_hypergraph(
    const Hypergraph *hypergraph,
    const HypergraphPotentials<BoolPotential> &edge_mask) {
  vector<HNode> *node_map =
      new vector<HNode>(hypergraph->nodes().size(), NULL);
  vector<HEdge> *edge_map =
      new vector<HEdge>(hypergraph->edges().size(), NULL);

  Hypergraph *new_graph = new Hypergraph();
  foreach (HNode node, hypergraph->nodes()) {
    if (node->terminal()) {
      // The node is a terminal, so just add it.
      (*node_map)[node->id()] =
          new_graph->add_terminal_node(node->label());
    } else {
      (*node_map)[node->id()] = new_graph->start_node(node->label());

      // Try to add each of the edges of the node.
      foreach (HEdge edge, node->edges()) {
        if (!(bool)edge_mask[edge]) continue;
        vector<HNode> tails;
        bool all_tails_exist = true;
        foreach (HNode tail_node, edge->tail_nodes()) {
          HNode new_tail_node = (*node_map)[tail_node->id()];
          if (new_tail_node == NULL) {
            // The tail node was pruned.
            all_tails_exist = false;
            break;
          } else {
            tails.push_back(new_tail_node);
          }
        }
        if (all_tails_exist) {
          HEdge new_edge = new_graph->add_edge(tails, edge->label());
          (*edge_map)[edge->id()] = new_edge;
        }
      }
      bool success = true;
      if (!new_graph->end_node()) {
        (*node_map)[node->id()] = NULL;
        success = false;
      }
      if (hypergraph->root()->id() == node->id()) {
        assert(success);
      }
    }
  }
  new_graph->finish();
  return new HypergraphProjection(hypergraph, new_graph,
                                  node_map, edge_map);
}

inline const HypergraphPotentials<LogViterbiPotential> *
pairwise_dot(const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
             const vector<double> &vec) {
  HypergraphPotentials<LogViterbiPotential> *potentials =
      new HypergraphPotentials<LogViterbiPotential>(sparse_potentials.hypergraph());
  foreach (HEdge edge, sparse_potentials.hypergraph()->edges()) {
    SparseVector edge_constraints =
        static_cast<SparseVector>(sparse_potentials.score(edge));
    foreach (SparsePair pair, edge_constraints) {
      (*potentials)[edge] *=
          LogViterbiPotential(pair.second * vec[pair.first]);
    }
  }
  SparseVector bias_constraints =
      static_cast<SparseVector>(sparse_potentials.bias());
  foreach (SparsePair pair, bias_constraints) {
    potentials->bias() *= LogViterbiPotential(pair.second * vec[pair.first]);
  }
  return potentials;
};

#endif // HYPERGRAPH_SEMIRING_H_
