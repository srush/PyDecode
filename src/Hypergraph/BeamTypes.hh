#include <bitset>
#include <vector>

using namespace std;

#define BITMAPSIZE 500
typedef bitset<BITMAPSIZE> binvec;



#define VALID_BINARY_VECTORS(X, Y)                \
    (X & Y).none()

/**
 * Binary vector. *Experimental*
 *
 * +: Bitwise AND.
 * *: Bitwise OR.
 * 0: All one bitset.
 * 1: All zero bitset.
 */

class BinaryVectorBeam {
  public:
    typedef bitset<BITMAPSIZE> ValType;

    static inline ValType add(ValType lhs, const ValType& rhs) {
        lhs &= rhs;
        return lhs;
    }

    static inline ValType times(ValType value, const ValType& rhs) {
        value |= rhs;
        return value;
    }

    static inline ValType one() {
        ValType vec = ValType(0x0);
        return vec;
    }

    static inline ValType zero() {
        ValType vec = ValType(0x0);
        vec.set();
        return vec;
    }

    static inline bool valid(const ValType& lhs,
                             const ValType& rhs) {
        return ((lhs & rhs).none());
    }
};


class AlphabetBeam {
  public:
    typedef vector<int> ValType;
    static const int kSize = 54;
    static inline ValType add(ValType lhs, const ValType& rhs) {
        // not used.
        return lhs;
    }

    /* static inline bool equals(const ValType &lhs, const ValType &rhs) { */
    /*     // not used. */
    /*     return lhs; */
    /* } */

    static inline ValType times(ValType value, const ValType& rhs) {
        for (int i = 0; i < kSize; ++i) {
            if (rhs[i] != -1)
                value[i] = rhs[i];
        }
        return value;
    }

    static inline ValType one() {
        ValType vec(kSize, -1);
        return vec;
    }

    static inline ValType zero() {
        // not used.
        ValType vec(kSize, -1);
        return vec;
    }

    static inline bool valid(const ValType& lhs,
                             const ValType& rhs) {
        for (int i = 0; i < kSize; ++i) {
            if (lhs[i] != -1 && rhs[i] != -1 && lhs[i] != rhs[i]) {
                return false;
            }
            /* for (int j = 0; j < kSize; ++j) { */
            /*     if (i != j && lhs[j] != -1 && rhs[i] != -1 && lhs[j] == rhs[i]) { */
            /*         return false; */
            /*     } */
            /* } */
        }
        return true;
    }
};

bool valid_binary_vectors(const bitset<BITMAPSIZE> &lhs,
                          const bitset<BITMAPSIZE> &rhs);

struct ParsingElement {
    ParsingElement *last;
    int edge;
    int position;
};

class ParsingBeam {
  public:
    typedef vector<int> ValType;
    static const int kSize = 54;
    static inline ValType add(ValType lhs, const ValType& rhs) {
        // not used.
        return lhs;
    }

    /* static inline bool equals(const ValType &lhs, const ValType &rhs) { */
    /*     // not used. */
    /*     return lhs; */
    /* } */

    static inline ValType times(ValType value, const ValType& rhs) {
        for (int i = 0; i < kSize; ++i) {
            if (rhs[i] != -1)
                value[i] = rhs[i];
        }
        return value;
    }

    static inline ValType one() {
        ValType vec(kSize, -1);
        return vec;
    }

    static inline ValType zero() {
        // not used.
        ValType vec(kSize, -1);
        return vec;
    }

    static inline ValType randValue() {
        return ValType(0, 0);
    }

    static inline ValType &normalize(ValType &val) {
        return val;
    }

    static inline bool valid(const ValType& lhs,
                             const ValType& rhs) {
        for (int i = 0; i < kSize; ++i) {
            if (lhs[i] != -1 && rhs[i] != -1 && lhs[i] != rhs[i]) {
                return false;
            }
            /* for (int j = 0; j < kSize; ++j) { */
            /*     if (i != j && lhs[j] != -1 && rhs[i] != -1 && lhs[j] == rhs[i]) { */
            /*         return false; */
            /*     } */
            /* } */
        }
        return true;
    }
};
