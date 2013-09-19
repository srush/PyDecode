#ifndef HYPERGRAPHIMPL_H_
#define HYPERGRAPHIMPL_H_

#include "Hypergraph.h"
#include "hypergraph.pb.h"
#include "features.pb.h"

#include "Weights.h"
#include "../common.h"
#include <vector>
#include <set>
using namespace std;

namespace Scarab {
namespace HG {

class HypernodeImpl;

class HyperedgeImpl: public Hyperedge {
public:
  virtual ~HyperedgeImpl(){ delete _features; }
  HyperedgeImpl(const string & label, str_vector * features, 
                int id, vector <Hypernode *> tail_nodes, 
                Hypernode * head_node):
  _id(id),
    _label(label), _tail_nodes(tail_nodes), 
    _head_node(head_node), _features(features) {
    
  }
  

  const Hypernode & tail_node(unsigned int i) const {
    return * (_tail_nodes[i]);
  }

  uint num_nodes() const{
    return _tail_nodes.size();
  }  
  
  const wvector & fvector() const {
    return *_features;
  }

  const string &feature_string() const {
    return _feature_str;
  }

  void set_feature_string(const string &feat_str) {
    _feature_str = feat_str;
  }

  const Scarab::HG::Hypernode & head_node() const { 
    return (*_head_node);
  }

  const vector <Scarab::HG::Hypernode*> & tail_nodes() const  {
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
  vector <Hypernode *> _tail_nodes;
  Hypernode * _head_node;

private:
  int _id;
  str_vector * _features;
  string _feature_str;
};


class HypernodeImpl: public Hypernode {
public:
  ~HypernodeImpl(){ delete _features; }
  HypernodeImpl(const string & label, int id, wvector * features) :
  _id(id),
  _label(label), _features(features) {}
 
  
  void add_edge(Hyperedge *edge) {
    _edges.push_back(edge);
  }

  void add_in_edge(Hyperedge *edge) {
    _in_edges.push_back(edge);
  }

  bool is_terminal() const {
    return _edges.size() == 0;
  }


  void prune_edges(const set<int> & keep_edges ) {
    vector < Hyperedge *> new_edges; 
    vector < Hyperedge *> new_in_edges; 
    foreach (Hyperedge * edge, _edges) {
      if (keep_edges.find(edge->id()) != keep_edges.end()) 
        new_edges.push_back(edge);
    }
    foreach (Hyperedge * edge, _in_edges) {
      if (keep_edges.find(edge->id()) != keep_edges.end()) 
        new_in_edges.push_back(edge); 
    }

    _edges = new_edges;
    _in_edges = new_in_edges;    
  }

  const Hyperedge & edge(uint i ) const {
    return *_edges[i];
  }
  
  unsigned int num_edges() const{
    return _edges.size();
  }

  unsigned int num_in_edges() const{
    return _in_edges.size();
  }

  const Hyperedge & in_edge(uint i) const {
    return *_in_edges[i];
  }

  uint id() const {
    return _id;
  }

  const vector <Hyperedge*> & edges() const {
    return _edges;
  }

  const vector <Hyperedge*> & in_edges() const {
    return _in_edges;
  } 
  
  void reid(int new_id) {
    _id = new_id;
  }

  string label() const {
    return _label; 
  }

private:  
   int  _id;
 public:
  vector <Hyperedge *> _edges; 
  string _label;


private:  
  wvector *_features;
  
  vector <Hyperedge *> _in_edges; 
};

class HypergraphImpl : public HGraph {
 public:
  HypergraphImpl(){}

  HypergraphImpl(vector <Hypernode*> nodes, vector <Hyperedge*> edges, Hypernode* root):
  _nodes(nodes), _edges(edges), _root(root){}
  
  ~HypergraphImpl(){
    for (int i = 0; i < _nodes.size(); ++i) {
      delete _nodes[i];
    }
    for (int i = 0; i < _edges.size(); ++i) {
      delete _edges[i];
    }
  }
  //HypergraphImpl(const char* filename);//const Hypergraph & pb);
      
  const Hypernode & root() const {
    return *_root;
  }

  unsigned int num_edges() const{
    return _edges.size();
  }

  uint num_nodes() const{
    return _nodes.size();
  }

  const Hypernode & get_node(unsigned int i) const {
    //const ForestNode & node =* (ForestNode*) _nodes[i];
    //assert (node.id() == i);
    return *_nodes[i];
  }


  const Hyperedge & get_edge(uint i) const {
    //const Hyperedge & edge =
    //assert (edge.id() == i);
    return *_edges[i];
  }
  
  
  void build_from_file(const char * file_name);
  void build_from_proto(Hypergraph *hgraph);

  const vector <Hypernode*> & nodes() const {
    return _nodes;
  }
  const vector <Hyperedge*> & edges() const {
    return _edges;
  }

  void prune(const HypergraphPrune & prune );
  Hypergraph write_to_proto(const HypergraphPrune &prune);
 protected:
  ::Hypergraph * hgraph;
  Hypernode * _root;
  vector <Hypernode *> _nodes;
  vector <Hyperedge *> _edges; 

  virtual Hypernode * make_node(const Hypergraph_Node & node,  wvector * features) {
    //    cout << "bad make node called!" << endl;
    return new HypernodeImpl(node.label(), node.id(), features); 
  }

  virtual void make_edge(const Hypergraph_Edge & edge, const Hyperedge * our_edge) {}
  virtual void convert_edge(const Hyperedge * our_edge, Hypergraph_Edge * edge, int id ) {} 
/* ;{ */
/*     edge->set_id(id); */
/*     edge->set_label(our_edge->label()); */
/*   } */
  virtual void convert_node(const Hypernode * our_node, Hypergraph_Node * node, int id ) { 
    node->set_id(id);
    node->set_label(our_node->label());
  }
  virtual void set_up(const Hypergraph & hgraph) {}
  virtual void print() const {}
  
};


}
}
#endif
