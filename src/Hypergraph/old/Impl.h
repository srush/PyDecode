// Copyright [2013] Alexander Rush
#ifndef HYPERGRAPH_IMPL_H_
#define HYPERGRAPH_IMPL_H_

#include <set>
#include <vector>
#include <string>

#include "./common.h"
#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Weights.h"
using namespace std;

class HypernodeImpl;

class HyperedgeImpl {
 public:
  virtual ~HyperedgeImpl() { delete _features; }

  HyperedgeImpl(const string &label, const string &features,
                int id, vector <HypernodeImpl *>tail_nodes,
                HypernodeImpl *head_node)
      : _id(id),
      _label(label), _tail_nodes(tail_nodes),
    _head_node(head_node), _features(load_weights_from_string(features)) {}

  const HypernodeImpl &tail_node(unsigned int i) const {
    return * (_tail_nodes[i]);
  }

  uint num_nodes() const { return _tail_nodes.size(); }

  const wvector & fvector() const {
    return *_features;
  }

  const string &feature_string() const {
    return _feature_str;
  }

  void set_feature_string(const string &feat_str) {
    _feature_str = feat_str;
  }

  const HypernodeImpl & head_node() const {
    return (*_head_node);
  }

  const vector <HypernodeImpl*> & tail_nodes() const {
    return _tail_nodes;
  }

  uint id() const {
    return _id;
  }

  string label() const {
    return _label;
  }

  void reid(int new_id) {
    _id = new_id;
  }

  const string _label;
  vector <HypernodeImpl *> _tail_nodes;
  HypernodeImpl * _head_node;

 private:
  int _id;
  str_vector * _features;
  string _feature_str;
};


class HypernodeImpl {
 public:

  HypernodeImpl(const string & label, int id)
      : _id(id),
      _label(label) {}

  void add_edge(HyperedgeImpl *edge) {
    _edges.push_back(edge);
  }

  void add_in_edge(HyperedgeImpl *edge) {
    _in_edges.push_back(edge);
  }

  bool is_terminal() const {
    return _edges.size() == 0;
  }

  void prune_edges(const set<int> & keep_edges ) {
    vector < HyperedgeImpl *> new_edges;
    vector < HyperedgeImpl *> new_in_edges;
    foreach (HyperedgeImpl * edge, _edges) {
      if (keep_edges.find(edge->id()) != keep_edges.end()) {
        new_edges.push_back(edge);
      }
    }
    foreach (HyperedgeImpl * edge, _in_edges) {
      if (keep_edges.find(edge->id()) != keep_edges.end()) {
        new_in_edges.push_back(edge);
      }
    }
    _edges = new_edges;
    _in_edges = new_in_edges;
  }

  const HyperedgeImpl & edge(uint i) const {
    return *_edges[i];
  }

  unsigned int num_edges() const {
    return _edges.size();
  }

  unsigned int num_in_edges() const {
    return _in_edges.size();
  }

  const HyperedgeImpl & in_edge(uint i) const {
    return *_in_edges[i];
  }

  uint id() const {
    return _id;
  }

  const vector <HyperedgeImpl*> & edges() const {
    return _edges;
  }

  const vector <HyperedgeImpl*> & in_edges() const {
    return _in_edges;
  }

  void reid(int new_id) {
    _id = new_id;
  }

  string label() const {
    return _label;
  }

 private:
  int _id;

 public:
  vector<HyperedgeImpl *> _edges;

  string _label;

 private:

  vector <HyperedgeImpl *> _in_edges;
};

class HypergraphImpl {
 public:
  HypergraphImpl() {}

  HypergraphImpl(vector<HypernodeImpl *> nodes,
                 vector<HyperedgeImpl *> edges,
                 HypernodeImpl *root)
      : _nodes(nodes), _edges(edges), _root(root) {}

  ~HypergraphImpl() {
    for (int i = 0; i < _nodes.size(); ++i)
      delete _nodes[i];

    for (int i = 0; i < _edges.size(); ++i)
      delete _edges[i];
  }

  const HypernodeImpl & root() const {
    return *_root;
  }

  unsigned int num_edges() const {
    return _edges.size();
  }

  uint num_nodes() const {
    return _nodes.size();
  }

  const HypernodeImpl & get_node(unsigned int i) const {
    return *_nodes[i];
  }

  const HyperedgeImpl & get_edge(uint i) const {
    return *_edges[i];
  }


  const vector <HypernodeImpl*> &nodes() const {
    return _nodes;
  }
  const vector <HyperedgeImpl*> &edges() const {
    return _edges;
  }

  void prune(const HypergraphPrune & prune);

 protected:
  HypernodeImpl *_root;
  vector<HypernodeImpl *> _nodes;
  vector<HyperedgeImpl *> _edges;

  virtual void print() const {}
};

#endif  // HYPERGRAPH_IMPL_H_
