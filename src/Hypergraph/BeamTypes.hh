#ifndef HYPERGRAPH_BEAMTYPES_H_
#define HYPERGRAPH_BEAMTYPES_H_

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

    static inline bool equals(const ValType &lhs, const ValType& rhs) {
        return lhs == rhs;
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

    static inline bool equals(const ValType &lhs, const ValType& rhs) {
        return lhs == rhs;
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
    // ParsingElement(int edge_, int position_, int total_size_)
    //         : edge(edge_), position(position_), total_size(total_size_){}

    ParsingElement() : up(NULL), position(-1), rolling_hash(1) {}
    const ParsingElement *up;
    int edge;
    int position;
    int total_size;

    // Rolling hash used for equality check.
    bitset<10000> hash;
    int rolling_hash;
    bool blank() const { return position == -1; }
    void recompute_hash() {
        if (up == NULL) {
            hash[edge] = 1;
            rolling_hash = edge % 1000000;
        } else {
            hash = up->hash;
            rolling_hash = (up->rolling_hash * edge) % 1000000;
            hash[edge] = 1;
        }
    }
    bool operator<(const ParsingElement &other) const {
        return rolling_hash < other.rolling_hash;
    }

};

class ParsingBeam {
  public:
    typedef ParsingElement ValType;

    static inline ValType times(ValType value, const ValType& rhs) {
        // Not symmetric.
        if (rhs.blank()) return value;
        if (value.blank()) return rhs;
        if (value.edge == rhs.edge) {
            if (rhs.position + 1 == value.position) {
                //value.position += 1;
                if (value.position == value.total_size) {
                    if (rhs.up != NULL){
                        value = (*rhs.up);
                    }
                } else {
                    value.up = rhs.up;
                    value.recompute_hash();
                }
            }
        } else {
            //ValType *v = new ValType(rhs);
            value.up = &rhs;
            value.recompute_hash();
        }
        return value;
    }

    static inline ValType one() {
        ParsingElement vec;
        vec.position = -1;
        return vec;
    }

    static inline bool valid(const ParsingElement& lhs,
                             const ParsingElement& rhs) {
        if (lhs.blank()) return true;
        if (rhs.blank()) return true;
        if (lhs.edge == rhs.edge) {
            if (rhs.position + 1 != lhs.position) return false;
            else return true;
        }
        if (lhs.position != 0) {
            return false;
        }
        return true;
    }


    struct equal_to
            : std::binary_function<ValType, ValType, bool>
    {
        bool operator() (const ValType &lhs1, const ValType& rhs1) const {
            return (lhs1.hash == rhs1.hash);
        }
    };
    struct hash
            : std::unary_function<ValType, int>
    {
        int operator() (const ValType &lhs) const {
            return lhs.rolling_hash;
        }
    };

    // static inline int hash(const ValType &lhs1) {
    //     return lhs1.rolling_hash;
    // }

    static inline bool equals(const ValType &lhs1, const ValType& rhs1) {
        return (lhs1.hash == rhs1.hash);
        // const ValType *lhs = &lhs1;
        // const ValType *rhs = &rhs1;
        // while(true) {
        //     if (lhs->hash != rhs->hash) return false;
        //     if (lhs->edge != rhs->edge || lhs->position != rhs->position) {
        //         return false;
        //     }
        //     if (lhs->up == NULL && rhs->up == NULL) {
        //         return true;
        //     }
        //     if ((lhs->up == NULL && rhs->up != NULL) ||
        //         (lhs->up != NULL && rhs->up == NULL)) {
        //         return false;
        //     }
        //     lhs = lhs->up;
        //     rhs = rhs->up;
        // }
    }
};

#endif  // HYPERGRAPH_BEAMTYPES_H_
