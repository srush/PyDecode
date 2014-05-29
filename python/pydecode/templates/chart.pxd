cdef class DPChartBuilder:
    cdef CHypergraph *_hg_ptr
    cdef vector[const CHypernode *] *_chart
    cdef _label_chart
    cdef _semiring
    cdef _scorer
    cdef bool _done
    cdef int _last
    cdef bool _debug
    cdef bool _strict
    cdef _hasher
    cdef int _max_arity
    cdef _data

cdef class SizedTupleHasher:
    cdef vector[int] _multipliers
    cdef int _max_size
    cdef _np_multi
    cpdef hasher(self, t)

cdef make_quartet(int a, int b, int c, int d)
cdef class Quartet:
    cdef int a
    cdef int b
    cdef int c
    cdef int d

cdef class QuartetHash:
    cdef Quartet _multipliers
    cdef int _max_size
    cpdef int hasher(self, Quartet t)
