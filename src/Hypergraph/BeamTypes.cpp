#include "Hypergraph/BeamTypes.hh"
#include <utility>
#include <vector>


bool valid_binary_vectors(const bitset<BITMAPSIZE> &lhs,
                          const bitset<BITMAPSIZE> &rhs) {
    return ((lhs & rhs).none());
}
