cdef extern from "Hypergraph/Automaton.hh":
    cdef cppclass CDFA "DFA":
        CDFA(int num_states, int num_symbols,
            const vector[map[int, int] ] &transition,
            const set[int] &final)
        bool final(int state)
        int transition(int state, int symbol)
        int valid_transition(int state, int symbol)


cdef extern from "Hypergraph/Algorithms.hh":
    CHypergraph *cmake_lattice "make_lattice"(
        int width, int height,
        const vector[vector[int] ] transitions,
        vector[CLatticeLabel ] *transitions) except +

    cdef cppclass CLatticeLabel "LatticeLabel":
        int i
        int j

    cdef cppclass CDFALabel "DFANode":
        int left_state
        int right_state


cdef class DFALabel:
    cdef CDFALabel label
    cdef _core
    cdef init(DFALabel self, CDFALabel label, core)

cdef class DFA:
    cdef const CDFA *thisptr


cdef class LatticeLabel:
    cdef CLatticeLabel label
    cdef init(LatticeLabel self, CLatticeLabel label)

cdef void _fill_trellis(float[:, :] emissions,
                        float[:] transitions,
                        int n_labels,
                        int[:] words,
                        float[:, :] trellis,
                        int[:,:] path)
