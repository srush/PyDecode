// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_WEIGHTS_H_
#define HYPERGRAPH_WEIGHTS_H_

// Weights.h - helper methods for using David Chiang's sparse vector
#include <iostream>
#include <fstream>
#include <string>
#include "svector.hpp"

using namespace std;

// Weight vector
typedef svector<int, double> wvector;

typedef svector<int, double> str_vector;

/**
 * @param file Name of file to read from.
 * @return Weight Vector from the file
 */
wvector *load_weights_from_file(const char * file);
wvector *load_weights_from_string(const string &str);

wvector *cmd_weights();
#endif  // HYPERGRAPH_WEIGHTS_H_
