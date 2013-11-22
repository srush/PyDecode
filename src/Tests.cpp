//test.cpp
#include <string>
#include <gtest/gtest.h>
#include "Hypergraph/Hypergraph.h"
#include "Hypergraph/Semirings.h"
#include "./common.h"

#ifndef NUM_LOOPS
#define NUM_LOOPS 10000
#endif

#ifndef SEMIRINGTEST
#define SEMIRINGTEST(TYPE) \
do { \
    TYPE::ValType a = TYPE::randValue(); \
    TYPE::ValType b = TYPE::randValue(); \
    TYPE::ValType c = TYPE::safe_add(a, b); \
    TYPE::ValType d = TYPE::safe_times(a, b); \
    /* Test consistency */ \
    ASSERT_EQ(a, TYPE::normalize(a)); \
    ASSERT_EQ(c, TYPE::safe_add(a,b)); \
    ASSERT_EQ(d, TYPE::safe_times(a,b)); \
    /* Test properties */ \
    ASSERT_EQ(TYPE::safe_add(a, TYPE::zero()), a); \
    ASSERT_EQ(TYPE::safe_times(a, TYPE::one()), a); \
    ASSERT_EQ(TYPE::safe_times(a, TYPE::zero()), TYPE::zero()); \
} while(0)
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

TEST(Decode, SemiringTests) {
    srand(time(NULL));  
    typedef CompPotential<ViterbiPotential, LogViterbiPotential> CVL;
    for(uint i = 0; i < NUM_LOOPS; ++i) {
        SEMIRINGTEST(ViterbiPotential);
        SEMIRINGTEST(LogViterbiPotential);
        SEMIRINGTEST(InsidePotential);
        SEMIRINGTEST(RealPotential);
        SEMIRINGTEST(TropicalPotential);
        SEMIRINGTEST(BoolPotential);
        SEMIRINGTEST(CountingPotential);
        SEMIRINGTEST(CVL);
        // SEMIRINGTEST(SparseVectorPotential);
        // SEMIRINGTEST(TreePotential);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
