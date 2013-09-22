// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_WEIGHTS_H_
#define HYPERGRAPH_WEIGHTS_H_

#include <string>

#include "Hypergraph/Weights.h"

DEFINE_string(weight_file, "", "svector weight file with translation params");

wvector * load_weights_from_file(const char * file) {
  fstream input(file, ios::in);

  char buf[1000];
  input.getline(buf, 100000);
  string s(buf);
  return svector_from_str<int, double>(s);
}

wvector * load_weights_from_str(const string &feat_str) {
  return svector_from_str<int, double>(feat_str);
}

wvector * cmd_weights() {
  return load_weights_from_file(FLAGS_weight_file.c_str());
}

#endif
