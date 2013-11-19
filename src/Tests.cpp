//test.cpp
#include <string>
#include <gtest/gtest.h>
#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "./common.h"

#ifndef NUM_LOOPS
#define NUM_LOOPS 1
#endif

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

TEST(Decode, SemiringPropertyTests) {
  srand(time(NULL));
  
  vector<BaseRegistry<BaseSemiring*>::creator_fnptr> creators = 
      BaseRegistry<BaseSemiring*>::retrieve_classes();
  ASSERT_GT(creators.size(), 0);

  foreach (BaseRegistry<BaseSemiring*>::creator_fnptr fnptr, creators) {
    // std::cout << "Beginning " << typeid(*(*fnptr)()).name() << std::endl;
    for(int i = 0; i < NUM_LOOPS; i++) {
      BaseSemiring* a = (*fnptr)();
      BaseSemiring* b = (*fnptr)();
      BaseSemiring* c = (*fnptr)();
      *c = *a + *b;
      ASSERT_EQ(*c, *a + *b);
      ASSERT_EQ(*a * a->annihlator(), a->annihlator());
      ASSERT_EQ(*a * a->identity(), *a);
      ASSERT_EQ(*a + a->annihlator(), *a);
      *c = *a;
      ASSERT_EQ(*a + *b, *c += *b);
      *c = *a;
      ASSERT_EQ(*a * *b, *c *= *b);
    }
  }
}

TEST(Decode, StaticSemiringTests) {
  srand(time(NULL));
  
  // vector<BaseRegistry<StaticBaseSemiringPotential*>::creator_fnptr> creators = 
  //     BaseRegistry<StaticBaseSemiringPotential*>::retrieve_classes();
  // ASSERT_GT(creators.size(), 0);

  // foreach (BaseRegistry<StaticBaseSemiringPotential*>::creator_fnptr fnptr, creators) {
  //   std::cout << "Beginning " << typeid(*(*fnptr)()).name() << std::endl;
  //   for(int i = 0; i < NUM_LOOPS; i++) {
  //     StaticBaseSemiringPotential* sb = (*fnptr)();
  //     // std::cout << *sb->randValue() << std::endl;
  //     // std::cout << "Type: " << typeid(*sb).name() << std::endl;
  //     std::cout << "ValType: " << typeid(*(*sb).randValue()).name() << std::endl;
  //     ValType *a = (*sb).randValue();
  //     ValType *b = (*sb).randValue();
  //     ValType *c = (*sb).randValue();

  //     *c = *(*sb).add(*a,*b);
  //     ASSERT_EQ(*c, *(*sb).add(*a,*b));

  //     ASSERT_EQ(*(*sb).times(*a,*(*sb).zero()), *(*sb).zero());
  //     ASSERT_EQ(*(*sb).times(*a,*(*sb).one()), *a);

  //     ASSERT_EQ(*(*sb).add(*a,*(*sb).zero()), *a);
  //   }
  // }

}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
