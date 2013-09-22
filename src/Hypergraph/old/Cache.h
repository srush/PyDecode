// Copyright [2013] Alexander Rush
#ifndef HYPERGRAPH_CACHE_H_
#define HYPERGRAPH_CACHE_H_

#include <assert.h>
#include <vector>

using namespace std;

template <class C, class V>
class Cache {
 public:
  vector <V> store;
  vector <bool> has_value;

  explicit Cache(int size) {
    store.resize(size);
    has_value.resize(size);
  }

  int size() {
    return has_value.size();
  }

  const V &get(const C &edge) const {
    int id = edge.id();
    assert(has_value[id]);
    return store[id];
  }

  V &get_no_check(const C &edge) {
    int id = edge.id();
    has_value[id] = true;
    return store[id];
  }

  V &get(const C &edge) {
    int id = edge.id();
    assert(has_value[id]);
    return store[id];
  }

  const V &get_default(const C &edge, const V &def) const {
    int id = edge.id();
    if (has_value[id]) {
      return store[id];
    } else {
      return def;
    }
  }

  V get_value(const C &edge) const {
    int id = edge.id();
    assert(has_value[id]);
    return store[id];
  }

  V get_by_key(int id) const {
    return store[id];
  }

  void set_value(const C & edge, V val) {
    int id = edge.id();
    assert(id < store.size());
    has_value[id]= true;
    store[id] = val;
  }

  bool has_key(const C & edge) const {
    return has_value[edge.id()];
  }

  bool has_key(int k) const {
    return has_value[k];
  }
};

#endif  // HYPERGRAPH_CACHE_H_
