// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_POTENTIALS_H_
#define HYPERGRAPH_POTENTIALS_H_

#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "./common.h"

// Classes used to associate projections with Hypergraphs
class HypergraphProjection;

template<typename SemiringType>
class HypergraphPotentials {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

  public:
    HypergraphPotentials(const Hypergraph *hypergraph,
                         V bias)
            : hypergraph_(hypergraph),
            bias_(bias) {}

    virtual V dot(const Hyperpath &path) const = 0;

    virtual V score(HEdge edge) const = 0;
    virtual inline V operator[] (HEdge edge) const = 0;

    const V &bias() const { return bias_; }
    V &bias() { return bias_; }

    HypergraphPotentials<S> *project_potentials(
        const HypergraphProjection &projection) const;

    virtual vector<V> &potentials() = 0;
    virtual const vector<V> &potentials() const = 0;

    /**
     * Pairwise "times" with another set of potentials.
     *
     * @return New hypergraph potentials.
     */
    virtual HypergraphPotentials<S> *times(
            const HypergraphPotentials<S> &potentials) const = 0;

    void check(const Hypergraph &graph) const {
        if (!graph.same(*hypergraph_)) {
            throw HypergraphException("Hypergraph does not match potentials.");
        }
    }

    void check(const HypergraphPotentials<S> &potentials) const {
        if (!potentials.hypergraph_->same(*hypergraph_)) {
            throw HypergraphException(
                "Hypergraph potentials do not match potentials.");
        }
    }

    // Create a copy of the weights.
    virtual HypergraphPotentials *clone() const = 0;

    // TODO(srush): This should be private. Fix template issue.
  public:
    const Hypergraph *hypergraph_;
    V bias_;
};


template<typename SemiringType>
  class HypergraphVectorPotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
    HypergraphVectorPotentials(const Hypergraph *hypergraph,
                               const vector<V> &potentials,
                               V bias)
            : HypergraphPotentials<SemiringType>(hypergraph, bias),
            potentials_(potentials) {
            assert(potentials.size() == hypergraph->edges().size());
    }
    vector<V> &potentials() { return potentials_; }
    const vector<V> &potentials() const { return potentials_; }

    explicit HypergraphVectorPotentials(const Hypergraph *hypergraph)
            : HypergraphPotentials<SemiringType>(hypergraph,
                                                 SemiringType::one()),
            potentials_(hypergraph->edges().size(), SemiringType::one()) {}

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            const vector<V> &potentials,
                            V bias) {
        return new HypergraphVectorPotentials<SemiringType>(
            hypergraph, potentials, bias);
    }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            score = SemiringType::times(score,
                                        potentials_[edge->id()]);
        }
        return SemiringType::times(score, this->bias_);
    }

    V score(HEdge edge) const { return potentials_[edge->id()]; }
    inline V operator[] (HEdge edge) const {
        return potentials_[edge->id()];
    }

    void insert(const HEdge& edge, const V& val) {
        potentials_[edge->id()] = val;
    }

    HypergraphPotentials<S> *times(
        const HypergraphPotentials<S> &potentials) const;

    HypergraphPotentials<S> *clone() const {
        return new HypergraphVectorPotentials(this->hypergraph_,
                                              potentials_,
                                              this->bias_);
    }

 protected:
    vector<V> potentials_;
};

template<typename SemiringType>
  class HypergraphMapPotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

 public:
   HypergraphMapPotentials(const Hypergraph *hypergraph,
                           const map<int, int> &potential_map,
                           const vector<V> &potentials,
                           V bias)
           : HypergraphPotentials<SemiringType>(hypergraph, bias),
            potential_map_(potential_map),
            potentials_(potentials) {
                //assert(potentials.size() == hypergraph->edges().size());
            }

    vector<V> &potentials() { return potentials_; }
    const vector<V> &potentials() const { return potentials_; }

    explicit HypergraphMapPotentials(const Hypergraph *hypergraph)
            : HypergraphPotentials<SemiringType>(hypergraph,
                                                 SemiringType::one()) {}

    static HypergraphPotentials<SemiringType> *
            make_potentials(const Hypergraph *hypergraph,
                            const map<int, int> &potentials_map,
                            const vector<V> &potentials,
                            V bias) {
        return new HypergraphMapPotentials<SemiringType>(
            hypergraph, potentials_map, potentials, bias);
    }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V cur_score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            cur_score = SemiringType::times(cur_score, score(edge));
        }
        return SemiringType::times(cur_score, this->bias_);
    }

    inline V score(HEdge edge) const {
        map<int, int>::const_iterator iter =
                potential_map_.find(edge->id());
        if (iter == potential_map_.end()) {
            return S::one();
        } else {
            return potentials_[iter->second];
        }
    }
    inline V operator[] (HEdge edge) const {
        return score(edge);
    }

    void insert(const HEdge& edge, const V& val) {
        int i = potentials_.size();
        potential_map_[edge->id()] = i;
        potentials_.push_back(val);
    }

    HypergraphPotentials<S> *times(
        const HypergraphPotentials<S> &potentials) const;

    HypergraphPotentials<S> *clone() const {
        return new HypergraphMapPotentials(this->hypergraph_,
                                           potential_map_,
                                           potentials_,
                                           this->bias_);
    }

 protected:
    map<int, int> potential_map_;
    vector<V> potentials_;
};

class HypergraphProjection {
 public:
    HypergraphProjection(const Hypergraph *original,
                         const Hypergraph *_new_graph,
                         const vector<HNode> *node_map,
                         const vector<HEdge> *edge_map,
                         bool bidirectional)
            : big_graph_(original),
            new_graph_(_new_graph),
            node_map_(node_map),
            edge_map_(edge_map),
            bidirectional_(bidirectional) {
                assert(node_map->size() == big_graph()->nodes().size());
                assert(edge_map->size() == big_graph()->edges().size());

                if (bidirectional) {
                    node_reverse_map_.resize(new_graph_->nodes().size(), NULL);
                    edge_reverse_map_.resize(new_graph_->edges().size(), NULL);
                    foreach (HNode node, original->nodes()) {
                        HNode mapped_node = (*node_map_)[node->id()];
                        if (mapped_node == NULL || mapped_node->id() < 0) continue;
                        node_reverse_map_[(*node_map_)[node->id()]->id()] = node;
                    }
                    foreach (HEdge edge, original->edges()) {
                        HEdge mapped_edge = (*edge_map_)[edge->id()];
                        if (mapped_edge == NULL || mapped_edge->id() < 0) continue;
                        edge_reverse_map_[mapped_edge->id()] = edge;
                    }
                }

#ifndef NDEBUG

                foreach (HNode node, *node_map) {
                    assert(node == NULL ||
                           node->id() < _new_graph->nodes().size());
                }
                foreach (HEdge edge, *edge_map) {
                    assert(edge == NULL ||
                           edge->id() < _new_graph->edges().size());
                }
#endif
            }

    ~HypergraphProjection() {

        delete node_map_;
        delete edge_map_;
        node_map_ = NULL;
        edge_map_ = NULL;
    }

    static HypergraphProjection *compose_projections(
        const HypergraphProjection *projection1, bool reverse1,
        const HypergraphProjection *projection2) {
        const Hypergraph *big_graph = projection1->big_graph();
        if (reverse1) big_graph = projection1->new_graph();

        vector<HEdge> *edge_map =
                new vector<HEdge>(big_graph->edges().size(), NULL);
        vector<HNode> *node_map =
                new vector<HNode>(big_graph->nodes().size(), NULL);
        foreach (HEdge edge, big_graph->edges()) {
            HEdge proj;
            if (reverse1) {
                proj = projection1->unproject(edge);
            } else {
                proj = projection1->project(edge);
            }
            if (proj != NULL && proj->id() >= 0) {
                (*edge_map)[edge->id()] = projection2->project(proj);
            }
        }
        foreach (HNode node, big_graph->nodes()) {
            HNode proj;
            if (reverse1) {
                proj = projection1->unproject(node);
            } else {
                proj = projection1->project(node);
            }
            if (proj != NULL && proj->id() >= 0) {
                (*node_map)[node->id()] = projection2->project(proj);
            }
        }

        return new HypergraphProjection(big_graph,
                                        projection2->new_graph(),
                                        node_map, edge_map, false);
    }


    static HypergraphProjection *project_hypergraph(
            const Hypergraph *hypergraph,
            const HypergraphPotentials<BoolPotential> &edge_mask);

    Hyperpath *project(const Hyperpath &original_path) const {
        original_path.check(*big_graph());
        vector<HEdge> edges;
        foreach (HEdge edge, original_path.edges()) {
            edges.push_back(project(edge));
        }
        return new Hyperpath(new_graph(), edges);
    }

    HEdge project(HEdge original) const {
        assert(original->id() < edge_map_->size());
        return (*edge_map_)[original->id()];
    }

    HNode project(HNode original) const {
        assert(original->id() < node_map_->size());
        return (*node_map_)[original->id()];
    }

    HNode unproject(HNode original) const {
        assert(bidirectional_);
        assert(original->id() < node_reverse_map_.size());
        return node_reverse_map_[original->id()];
    }

    HEdge unproject(HEdge original) const {
        assert(bidirectional_);
        assert(original->id() < edge_reverse_map_.size());
        return edge_reverse_map_[original->id()];
    }

    const Hypergraph *big_graph() const {
        return big_graph_;
    }
    const Hypergraph *new_graph() const {
        return new_graph_;
    }

 private:
    const Hypergraph *big_graph_;
    const Hypergraph *new_graph_;

    // Owned.
    const vector<HNode> *node_map_;
    const vector<HEdge> *edge_map_;
    vector<HNode> node_reverse_map_;
    vector<HEdge> edge_reverse_map_;

    bool bidirectional_;
};

template<typename SemiringType>
  class HypergraphProjectedPotentials :
  public HypergraphPotentials<SemiringType> {
    typedef SemiringType S;
    typedef typename SemiringType::ValType V;

  public:
    HypergraphProjectedPotentials(
        HypergraphPotentials<S> *base_potentials,
        const HypergraphProjection *projection)
            : HypergraphPotentials<SemiringType>(
                projection->big_graph(),
                base_potentials->bias()),
            base_potentials_(base_potentials),
            projection_(projection) {
                base_potentials->check(*projection->new_graph());
            }

    static HypergraphPotentials<SemiringType> *
            make_potentials(HypergraphPotentials<S> *base_potentials,
                            const HypergraphProjection *projection) {
        return new HypergraphProjectedPotentials<SemiringType>(
            base_potentials, projection);
    }

    vector<V> &potentials() { return base_potentials_->potentials(); }
    const vector<V> &potentials() const {
        return base_potentials_->potentials(); }

    V dot(const Hyperpath &path) const {
        path.check(*this->hypergraph_);
        V base_score = SemiringType::one();
        foreach (HEdge edge, path.edges()) {
            base_score = SemiringType::times(base_score,
                                             score(edge));
        }
        return SemiringType::times(base_score, this->bias_);
    }

    V score(HEdge edge) const {
        HEdge new_edge = projection_->project(edge);
        return base_potentials_->score(new_edge);
    }
    inline V operator[] (HEdge edge) const {
        return score(edge);
    }

    HypergraphPotentials<S> *times(
        const HypergraphPotentials<S> &other) const {
        vector<typename SemiringType::ValType> new_potentials(
            projection_->new_graph()->edges().size());
        int i = -1;
        foreach (HEdge edge, projection_->new_graph()->edges()) {
            i++;
            new_potentials[i] = SemiringType::times(
                base_potentials_->score(edge),
                other.score(edge));
        }
        return new HypergraphProjectedPotentials<SemiringType>(
            new HypergraphVectorPotentials<SemiringType>(
                projection_->new_graph(),
                new_potentials,
                SemiringType::times(this->bias_, other.bias())),
            projection_);
    }

    HypergraphPotentials<S> *clone() const {
        return new HypergraphProjectedPotentials(
            base_potentials_->clone(),
            projection_);
    }


 protected:
    HypergraphPotentials<S> *base_potentials_;
    const HypergraphProjection *projection_;
};

void
pairwise_dot(
    const HypergraphPotentials<SparseVectorPotential> &sparse_potentials,
    const vector<double> &vec,
    HypergraphPotentials<LogViterbiPotential> *weights);

#endif  // HYPERGRAPH_POTENTIALS_H_
