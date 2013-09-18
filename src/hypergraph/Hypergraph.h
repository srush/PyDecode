#ifndef HYPERGRAPH_H_
#define HYPERGRAPH_H_
#include "Weights.h"
#include <vector>
#include <set>
#include <string>
using namespace std;

namespace Scarab {
namespace HG {


// Hypergraph interface for internal use. 
// This interface is entirely immutable! Every function is const
// See Cache.h for implementing state on top on hypergraphs.  

class Hypernode;
typedef const Hypernode *HNode; 
typedef vector <const Hypernode * > HNodes;  

// Base class for weighted hyperedge 
class Hyperedge {
 public:
  //Hyperedge(unsigned int id): _id(id){}
   virtual ~Hyperedge() {}

  /** 
   * Get edge id
   * 
   * @return The id of this edge in a fixed hypergraph
   */
   virtual unsigned int id() const = 0; 
  
   virtual string label() const = 0;

  // These are deprecated use iterator

  /** 
   * Get a node in this edges tail
   * @deprecated
   * @param i The node position, in tail_node order 
   * 
   * @return The hypernode
   */
  virtual const Hypernode & tail_node(unsigned int i) const =0;

  /** 
   * The number of tail nodes in this edge.
   * @deprecated
   * 
   * @return The length
   */
  virtual unsigned int num_nodes() const =0 ;
  
  /** 
   * The feature vector associated with this node. 
   * TODO: remove this from the representation
   * @deprecated
   * 
   * @return Feature vector
   */
  virtual const svector<int, double> & fvector() const = 0;
  
  /** 
   * Get the node at the head of this hyperedge
   * 
   * @return Head node
   */
  virtual const Hypernode & head_node() const  = 0;

  // TODO: These should be iterators, figuring that part out
  /** 
   * Get all the nodes in the tail of the hyperedge
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to nodes.
   */
  virtual const vector <Hypernode *> & tail_nodes() const =0; 


  // protected: 
  //unsigned int _id;
};


typedef const Hyperedge * HEdge;
typedef vector <HEdge > HEdges;  

/**
 * Hypernode - Constant representation of a hypernode in a hypergraph. 
 * Accessors for edges above and below. 
 */
class Hypernode {
 public:

  /** 
   * WARNING: Private except to Hypergraph
   */
  //Hypernode(unsigned int id): _id(id){}
   virtual ~Hypernode() {}
  /** 
   * The unique identifier of the Hypernode 
   * @deprecated
   * 
   * @return The number 
   */
   virtual unsigned int id() const = 0;// {return _id;}

  // This interface is deprecated, use iterators below instead

  /** 
   * Get the number of hyperedges that have this node as head
   * @deprecated
   * 
   * @return The number 
   */
  virtual unsigned int num_edges() const = 0;
  
  /** 
   * Get the number of hyperedges that have this node as tail
   * @deprecated
   * 
   * @return The number
   */
  virtual unsigned int num_in_edges() const =0;


  /** 
   * Get the hyperedge  this node as head
   * @deprecated
   * @param i Local local id of edge
   * 
   * @return The hyperedge
   */
  virtual const Hyperedge & edge(unsigned int i ) const = 0;   
  

  /** 
   * Get the hyperedge this node as tail
   * @deprecated
   * @param i Local local id of edge
   * 
   * @return The hyperedge
   */
  virtual const Hyperedge & in_edge(unsigned int i) const = 0;



  /** 
   * Is this node a terminal nodes. (no children) 
   * 
   * @return True if terminal (assert is_terminal() == (num_edges() == 0))
   */
  virtual bool is_terminal() const =0;
 
  // TODO: These should be (lazy) iterators, figure that part out
  /** 
   * Get all hyperedges with this hypernode as head.
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to edges.
   */
  virtual const vector <Hyperedge*> &edges() const =0; 

  /** 
   * Get all hyperedges with this hypernode as a tail.
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to edges.
   */
  virtual const vector <Hyperedge *> &in_edges() const =0; 

  virtual string label() const { return ""; }; 
};

class HGraph {
 public:
  virtual ~HGraph() {}

  /** 
   * Display the hypergraph for debugging.
   */
  virtual void print() const = 0;
  
  /** 
   * Get the root of the hypergraph
   * 
   * @return Hypernode at root
   */
  virtual const Hypernode &root() const = 0;

  
  // TODO: remove these 
  virtual unsigned int num_edges() const =0;
  virtual unsigned int num_nodes() const = 0;
    
  virtual const Hypernode &get_node(unsigned int i) const = 0; 
  virtual const Hyperedge &get_edge(unsigned int i) const = 0;
  
  // Switching to iterator interface
  /** 
   * Get all hypernodes in the hypergraph. (Assume unordered)
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to hypernodes in hypergraph.
   */
  virtual const vector <Hypernode* > &nodes() const = 0; 

  /** 
   * Get all hyperedges in the hypergraph. (Assume unordered)
   * WARNING: Treat this as a const iterator.
   * @return Const iterator to edges in hypergraph .
   */
  virtual const vector <Hyperedge*> & edges() const =0; 
};

struct HypergraphPrune {
  HypergraphPrune(const HGraph &hgraph_) : hgraph(hgraph_) {}
  set <int> nodes;
  set <int> edges;
  const HGraph & hgraph;
};


}
}
#endif
