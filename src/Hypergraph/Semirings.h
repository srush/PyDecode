
#ifndef HYPERGRAPH_SEMIRING_H_
#define HYPERGRAPH_SEMIRING_H_

#include <algorithm>
#include "Hypergraph/Hypergraph.h"
#include "./common.h"

/**
 * A base class of a weight with traits of a semiring
 * including + and * operators, and annihlator/identity elements.
 */
template<typename ValType, typename SemiringWeight>
class BaseSemiringWeight {
public:
	BaseSemiringWeight(const SemiringWeight& other)
		: value(normalize(other.value)) {}
	BaseSemiringWeight(ValType val) : value(normalize(val)) {}
    BaseSemiringWeight() : value(zero()) {}

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

    static SemiringWeight add(SemiringWeight lhs, const SemiringWeight &rhs) {
      return lhs + rhs;
    }

    static SemiringWeight times(SemiringWeight lhs, const SemiringWeight &rhs) {
      return lhs * rhs;
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

/**
 * Implements the Viterbi type of semiring as described in Huang 2006.
 * +: max
 * *: *
 * 0: 0
 * 1: 1
 */
class ViterbiWeight : public BaseSemiringWeight<double, ViterbiWeight> {
public:
ViterbiWeight(double value) : BaseSemiringWeight<double, ViterbiWeight>(normalize(value)) { }
  ViterbiWeight() : BaseSemiringWeight<double, ViterbiWeight>() { }
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

/**
 * Implements the log-space Viterbi type of semiring.
 * +: max
 * *: +
 * 0: -INF
 * 1: 0
 */
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


/**
 * Implements the Boolean type of semiring as described in Huang 2006
 * +: logical or
 * *: logical and
 * 0: false
 * 1: true
 */
class BoolWeight : public BaseSemiringWeight<bool, BoolWeight> {
public:
BoolWeight(bool value) : BaseSemiringWeight<bool, BoolWeight>(normalize(value)) { }
  BoolWeight() : BaseSemiringWeight<bool, BoolWeight>() { }

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

/**
 * Implements the Inside type of semiring as described in Huang 2006
 * +: +
 * *: *
 * 0: 0
 * 1: 1
 */
class InsideWeight : public BaseSemiringWeight<double, InsideWeight> {
public:
  InsideWeight(double value) : BaseSemiringWeight<double, InsideWeight>(normalize(value)) { }

  InsideWeight() :
  BaseSemiringWeight<double, InsideWeight>(InsideWeight::zero()) {}

	InsideWeight& operator+=(const InsideWeight& rhs) {
		value = value + rhs.value;
		return *this;
	}
	InsideWeight& operator*=(const InsideWeight& rhs) {
		value = value * rhs.value;
		return *this;
	}

    friend InsideWeight operator/(InsideWeight lhs, const InsideWeight &rhs) {
      lhs.value /= rhs.value;
      return lhs;
	}

	static const InsideWeight one() { return InsideWeight(1.0); }
	static const InsideWeight zero() { return InsideWeight(0.0); }

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

/**
 * Implements the Inside type of semiring as described in Huang 2006
 * +: min
 * *: +
 * 0: INF
 * 1: 0
 */
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




/**
 * Implements the Counting type of semiring as described in Huang 2006
 * +: +
 * *: *
 * 0: 0
 * 1: 1
 */
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

/**
 * Comparison pair. *Experimental*
 * Type (s, t) op (s', t')
 * +: if (s > s') then (s, t) else (s', t')
 * *: (s * s', t * t')
 * 0: (0, 0)
 * 1: (1, 1)
 */
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
class SparseVectorWeight : public BaseSemiringWeight<SparseVector, SparseVectorWeight> {
public:
SparseVectorWeight(const SparseVector vec) : BaseSemiringWeight<SparseVector, SparseVectorWeight>(vec) { }
SparseVectorWeight() : BaseSemiringWeight<SparseVector, SparseVectorWeight>(SparseVectorWeight::zero()) { }

  	SparseVectorWeight& operator+=(const SparseVectorWeight& rhs) {
		return *this;
	}

	SparseVectorWeight& operator*=(const SparseVectorWeight& rhs) {
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
          vec.push_back(pair<int, int>(i, value[i].second + rhs.value[j].second));
          ++i;
          ++j;
        }
      }
      value = vec;
      return *this;
	}

	static const SparseVectorWeight one() { return SparseVectorWeight(SparseVector()); }
	static const SparseVectorWeight zero() { return SparseVectorWeight(SparseVector()); }

	int normalize(int val) const {
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

class HypergraphProjection;

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

  HypergraphWeights(const Hypergraph *hypergraph)
    : hypergraph_(hypergraph),
      weights_(hypergraph->edges().size(), SemiringType::one()),
      bias_(SemiringType::one()) {}

 SemiringType dot(const Hyperpath &path) const {
   path.check(*hypergraph_);
   SemiringType score = SemiringType::one();
   foreach (HEdge edge, path.edges()) {
     score *= weights_[edge->id()];
   }
   return score + bias_;
 }

  SemiringType score(HEdge edge) const { return weights_[edge->id()]; }
  const SemiringType& operator[] (HEdge edge) const {
    return weights_[edge->id()];
  }
  SemiringType& operator[] (HEdge edge) {
    return weights_[edge->id()];
  }

  const SemiringType &bias() const { return bias_; }
  SemiringType &bias() { return bias_; }

  HypergraphWeights<SemiringType> *project_weights(
    const HypergraphProjection &projection) const;

  /**
   * Pairwise "times" with another set of weights.
   *
   * @return New hypergraphweights.
   */
  HypergraphWeights <SemiringType> *times(
      const HypergraphWeights<SemiringType> &weights) const;

  void check(const Hypergraph &graph) const {
    if (!graph.same(*hypergraph_)) {
      throw HypergraphException("Hypergraph does not match weights.");
    }
  }

  void check(const HypergraphWeights<SemiringType> &weights) const {
    if (!weights.hypergraph_->same(*hypergraph_)) {
      throw HypergraphException("Hypergraph weights do not match weights.");
    }
  }

  const Hypergraph *hypergraph() const { return hypergraph_; }

 protected:
  const Hypergraph *hypergraph_;
  vector<SemiringType> weights_;
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
      const HypergraphWeights<BoolWeight> &edge_mask);

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
inline double HypergraphWeights<double>::dot(const Hyperpath &path) const {
  path.check(*hypergraph_);
  double score = 0.0;
  foreach (HEdge edge, path.edges()) {
    score += weights_[edge->id()];
  }
  return score + bias_;
}

template<typename SemiringType>
HypergraphWeights<SemiringType> *HypergraphWeights<SemiringType>::times(const HypergraphWeights<SemiringType> &other) const {
  check(other);
  vector<SemiringType> new_weights(weights_);
  for (uint i = 0; i < other.weights_.size(); ++i) {
    new_weights[i] *= other.weights_[i];
  }
  return new HypergraphWeights<SemiringType>(hypergraph_,
                               new_weights,
                               bias_ * other.bias_);
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



inline HypergraphProjection *HypergraphProjection::project_hypergraph(
    const Hypergraph *hypergraph,
    const HypergraphWeights<BoolWeight> &edge_mask) {
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

inline const HypergraphWeights<LogViterbiWeight> *
pairwise_dot(const HypergraphWeights<SparseVectorWeight> &sparse_weights,
             const vector<double> &vec) {
  HypergraphWeights<LogViterbiWeight> *weights =
      new HypergraphWeights<LogViterbiWeight>(sparse_weights.hypergraph());
  foreach (HEdge edge, sparse_weights.hypergraph()->edges()) {
    SparseVector edge_constraints =
        static_cast<SparseVector>(sparse_weights.score(edge));
    foreach (SparsePair pair, edge_constraints) {
      (*weights)[edge] *=
          LogViterbiWeight(pair.second * vec[pair.first]);
    }
  }
  SparseVector bias_constraints =
      static_cast<SparseVector>(sparse_weights.bias());
  foreach (SparsePair pair, bias_constraints) {
    weights->bias() *= LogViterbiWeight(pair.second * vec[pair.first]);
  }
  return weights;
};

#endif // HYPERGRAPH_SEMIRING_H_
