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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
