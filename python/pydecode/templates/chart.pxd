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

cdef class IntTuple4:
    cdef int a, b, c, d

cdef class IntTuple4Hasher:
    cdef IntTuple4 _multipliers
    cdef int _max_size
    cpdef int hasher(self, IntTuple4 t)
