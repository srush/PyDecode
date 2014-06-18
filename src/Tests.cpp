//test.cpp
#include <string>
#include <gtest/gtest.h>
#include "Hypergraph/Hypergraph.hh"
#include "Hypergraph/Semirings.hh"
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

/*
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
*/

TEST(Decode, BinarySemiringTests) {
	srand(time(NULL));
    typedef CompPotential<ViterbiPotential, LogViterbiPotential> CVL;
    for(uint i = 0; i < NUM_LOOPS; ++i) {
		BinaryVectorPotential::ValType a = BinaryVectorPotential::randValue();
		ASSERT_EQ(BinaryVectorPotential::times(a, BinaryVectorPotential::one()), a);
		ASSERT_EQ(BinaryVectorPotential::times(a, BinaryVectorPotential::zero()), BinaryVectorPotential::zero());
		ASSERT_EQ(BinaryVectorPotential::add(a, BinaryVectorPotential::zero()), a);
	}

	BinaryVectorPotential::ValType a = BinaryVectorPotential::ValType(0xfa);
	BinaryVectorPotential::ValType b = BinaryVectorPotential::ValType(0x05);
	BinaryVectorPotential::ValType c = BinaryVectorPotential::ValType(0x15);
	ASSERT_EQ(BinaryVectorPotential::ValType(0xff), BinaryVectorPotential::times(a, b));
	ASSERT_EQ(BinaryVectorPotential::ValType(0xff), BinaryVectorPotential::times(a, b));
	ASSERT_TRUE(BinaryVectorPotential::valid(a, b));
	ASSERT_FALSE(BinaryVectorPotential::valid(a, c));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
