# Cython template hack.
cdef extern from "<bitset>" namespace "std":
    cdef cppclass cbitset "bitset<1600>":
        void set(int, int)
        bool& operator[](int)

cdef class Bitset:
    cdef cbitset data

    cdef init(self, cbitset data):
        self.data = data
        return self

    def __setitem__(self, int position, bool val):
        self.data.set(position, val)

    def __getitem__(self, int position):
        return self.data[position]
