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



cdef class SizedTupleHasher:
    cdef vector[int] _multipliers
    cdef int _max_size
