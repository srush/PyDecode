//test.cpp
#include "gtest/gtest.h"
#include "Hypergraph/Hypergraph.h"

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

TEST(Decode, ViterbiWeight) {
  // Test for correct normalization
  ViterbiWeight high = ViterbiWeight(1.1);
  std::cerr << high;
  ASSERT_EQ(high, ViterbiWeight(1.0));
  ViterbiWeight low = ViterbiWeight(-0.1);
  ASSERT_EQ(low, ViterbiWeight(0.0));

  // Test assignment
  high = 0.5;
  ASSERT_EQ(high, ViterbiWeight(0.5));

  // Test Operators
  high += low;
  ASSERT_EQ(high, ViterbiWeight(0.5));
  low += high;
  ASSERT_EQ(low, ViterbiWeight(0.5));
  high *= low;
  ASSERT_EQ(high, ViterbiWeight(0.25));
  high *= ViterbiWeight::one();
  ASSERT_EQ(high, ViterbiWeight(0.25));
  high *= ViterbiWeight::zero();
  ASSERT_EQ(high, ViterbiWeight::zero());

  // Test casting
  double d = 0.5;
  high = 0.25;
  d = (double)high + d;
  ASSERT_EQ(d, 0.75);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
