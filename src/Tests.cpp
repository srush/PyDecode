//test.cpp
#include <string>
#include <gtest/gtest.h>
#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "./common.h"

TEST(Decode, TestHypergraph) {
  Hypergraph test;
  vector<HNode> nodes;
  nodes.push_back(test.add_terminal_node("one"));
  nodes.push_back(test.add_terminal_node("two"));
  nodes.push_back(test.add_terminal_node("three"));

  test.start_node("root");
  test.add_edge(nodes, "Edgy");
  test.end_node();

  test.finish();
  ASSERT_EQ(test.nodes().size(), 4);
  ASSERT_EQ(test.edges().size(), 1);
}

TEST(Decode, ViterbiPotential) {
  // Test for correct normalization
  ViterbiPotential high = ViterbiPotential(1.1);
  ASSERT_EQ(high, ViterbiPotential(1.0));
  ViterbiPotential low = ViterbiPotential(-0.1);
  ASSERT_EQ(low, ViterbiPotential(0.0));

  // Test assignment
  high = 0.5;
  ASSERT_EQ(high, ViterbiPotential(0.5));

  // Test Operators
  high += low;
  ASSERT_EQ(high, ViterbiPotential(0.5));
  low += high;
  ASSERT_EQ(low, ViterbiPotential(0.5));
  high *= low;
  ASSERT_EQ(high, ViterbiPotential(0.25));
  high *= ViterbiPotential::one();
  ASSERT_EQ(high, ViterbiPotential(0.25));
  high *= ViterbiPotential::zero();
  ASSERT_EQ(high, ViterbiPotential::zero());

  // Test casting
  double d = 0.5;
  high = 0.25;
  d = (double)high + d;
  ASSERT_EQ(d, 0.75);
}

TEST(Semirings, PropertyTests) {
  vector<create_type_fnptr> creators = BaseSemiringFactory::retrieve_classes();
  foreach (create_type_fnptr fnptr, creators) {
    // BaseSemiring* = (*fnptr)();
    // // ASSERT_EQ()
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
