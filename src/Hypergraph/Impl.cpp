#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "Hypergraph/Impl.h"
#include "Hypergraph/Cache.h"
#include "Weights.h"
#include "../interface/hypergraph/gen-cpp/hypergraph.pb.h"
#include "../interface/hypergraph/gen-cpp/features.pb.h"

using namespace google::protobuf::io;

// void HypergraphImpl::prune(const HypergraphPrune & prune) {
//   vector <Hypernode *> new_nodes;
//   vector <Hyperedge *> new_edges;

//   int node_count = 0;
//   foreach (Hypernode * tmp_node, nodes()) {
//     HypernodeImpl * node = (HypernodeImpl *)tmp_node;
//     if (prune.nodes.find(node->id()) == prune.nodes.end()) {
//       new_nodes.push_back((Hypernode*)node);
//       node->prune_edges(prune.edges);
//       node->reid(node_count);
//       node_count++;
//     }
//   }

//   int edge_count = 0;
//   foreach (Hyperedge * tmp_edge, edges()) {
//     HyperedgeImpl * edge = (HyperedgeImpl *)tmp_edge;
//     if (prune.edges.find(edge->id()) == prune.edges.end()) {
//       new_edges.push_back((Hyperedge*)edge);
//       edge->reid(edge_count);
//       edge_count++;
//     }
//   }

//   _nodes = new_nodes;
//   _edges = new_edges;
//}


  Hypergraph HypergraphImpl::write_to_proto(const HypergraphPrune &prune) {
    Hypergraph hgraph;

    Cache<Hypernode, int> renumbering(num_nodes());
    int num_new_nodes = 0;
    int num_new_edges = 0;
    foreach (HNode my_node, nodes()) {
      if (my_node->id() == root().id() ||
          prune.nodes.find(my_node->id()) == prune.nodes.end()) {
        renumbering.set_value(*my_node, num_new_nodes);
        num_new_nodes++;
      }
      assert(my_node->id() < num_nodes());

    }
    foreach (HNode my_node, nodes()) {
      if (!renumbering.has_key(*my_node)) {
        continue;
      }
      assert(my_node->id() < num_nodes());
      int node_id = renumbering.get(*my_node);
      Hypergraph_Node * node = hgraph.add_node();

      convert_node(my_node, node, node_id);

      foreach (HEdge my_edge, my_node->edges()) {
        int edge_id = num_new_edges;
        bool has_all_children = true;
        foreach (HNode sub_node, my_edge->tail_nodes()) {
          has_all_children &= renumbering.has_key(*sub_node);
        }

        if (prune.edges.find(my_edge->id()) == prune.edges.end() && has_all_children) {
          num_new_edges++;
        } else {
          continue;
        }
        assert(my_node->id() < num_nodes());
        assert(my_edge->id() < num_edges());
        Hypergraph_Edge *edge = node->add_edge();
        str_vector * features;

        edge->SetExtension(edge_fv, ((HyperedgeImpl *)my_edge)->feature_string());

        foreach (HNode sub_node, my_edge->tail_nodes()) {
          //int id = edge.tail_node_ids(k);
          assert(renumbering.has_key(*sub_node));
          edge->add_tail_node_ids(renumbering.get(*sub_node)); //push_back( _nodes[id]);
        }
        convert_edge(my_edge, edge, edge_id);
      }
    }
    //assert (_nodes.size() == (uint)hgraph->node_size());
    hgraph.set_root(renumbering.get(root()));
//     cerr << "num nodes " << num_nodes() << " " << hgraph.node_size() << endl;
//     cerr << "num edges " << num_edges() << " " << num_new_edges << endl;
    //cerr << "size " << num_nodes() << " " << num_edges() << endl;
    //   {
    //     fstream output(file_name, ios::out | ios::binary);
    //     if (!hgraph.SerializeToOstream(&output)) {
    //       assert (false);
    //     }
    //   }
    return hgraph;
  }
void HypergraphImpl::build_from_file(const char * file_name) {
  hgraph = new ::Hypergraph();
  {
    //int fd = open(file_name, O_RDONLY);
    fstream input(file_name, ios::in | ios::binary);
    google::protobuf::io::IstreamInputStream fs(&input);

    google::protobuf::io::CodedInputStream coded_fs(&fs);
    coded_fs.SetTotalBytesLimit(1000000000, -1);
    hgraph->ParseFromCodedStream(&coded_fs);

    //if (!hgraph->ParseFromIstream(&input)) {
    //assert (false);
    //}
  }
  build_from_proto(hgraph);
}

void HypergraphImpl::build_from_proto(Hypergraph *hgraph) {
  set_up(*hgraph);

  assert (hgraph->node_size() > 0);

  _nodes.resize(hgraph->node_size());
  for (int i = 0; i < hgraph->node_size(); i++) {
    const Hypergraph_Node & node = hgraph->node(i);

    string feat_str = node.GetExtension(node_fv);
    str_vector * features = svector_from_str<int, double>(feat_str);

    //
    Hypernode *forest_node = make_node(node, features);

    //assert (forest_node->
    assert (node.id() < hgraph->node_size());
    _nodes[node.id()] = forest_node;
    //assert(_nodes[forest_node->id()]->id() == forest_node->id());
  }

  int edge_id = 0;
  for (int i = 0; i < hgraph->node_size(); i++) {
    const Hypergraph_Node& node = hgraph->node(i);
    //assert (node.id()  == i);

    for (int j=0; j < node.edge_size(); j++) {
      const Hypergraph_Edge& edge = node.edge(j);
      str_vector * features;
      string feat_str = "";
      if (edge.HasExtension(edge_fv)) {
        feat_str = edge.GetExtension(edge_fv);
        features = svector_from_str<int, double>(feat_str);
      } else {
        features = new svector<int, double>();
      }

      vector <Scarab::HG::Hypernode *> tail_nodes;
      for (int k =0; k < edge.tail_node_ids_size(); k++ ){
        int id = edge.tail_node_ids(k);
        tail_nodes.push_back( _nodes[id]);
      }



      Scarab::HG::Hyperedge * forest_edge = new Scarab::HG::HyperedgeImpl(edge.label(),
                                                                          features,
                                                                          edge_id,
                                                                          tail_nodes,
                                                                          _nodes[node.id()]);
      ((HyperedgeImpl *)forest_edge)->set_feature_string(feat_str);
      make_edge(edge, forest_edge);
      for (int k =0; k < edge.tail_node_ids_size(); k++){
        int id = edge.tail_node_ids(k);
        ((HypernodeImpl*)_nodes[id])->add_in_edge(forest_edge);
      }
      ((HypernodeImpl*)_nodes[node.id()])->add_edge(forest_edge);

      edge_id++;
      //int for_edge_id = forest_edge->id();
      _edges.push_back(forest_edge);//[for_edge_id] = forest_edge;
    }
    //cout << node.id() << " "<<  _nodes[node.id()]->num_edges() << " " << node.edge_size() << " " << _nodes[node.id()]->is_word() << endl;
    //assert (_nodes[node.id()]->num_edges() == (uint)node.edge_size() );
  }
  assert (_nodes.size() == (uint)hgraph->node_size());
  int root_num = hgraph->root();
  _root = _nodes[hgraph->root()];//_nodes[_nodes.size()-1];
  //delete hgraph;
}
